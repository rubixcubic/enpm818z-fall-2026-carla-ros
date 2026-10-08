#!/usr/bin/env python3
"""Semantic segmentation of the AV's front camera, by a trained network.

Slides: 'Semantic segmentation: a class for every pixel' and 'A class for
every pixel, in CARLA'. On an AV no camera gives the classes: a network
computes them from the image, every frame. This node is that network.

    in   /carla/ego_vehicle/rgb_front/image   the L2 bridge's camera, 1280 x 720

    out  /l5/seg/labels     mono8   one class per pixel, as CARLA tags (1 is road,
                                    2 is sidewalk, ...; classes.py has the list)
         /l5/seg/classes    bgr8    the same, in CARLA's colors (CityScapes palette)
         /l5/seg/overlay    bgr8    the camera image with the classes on top,
                                    and a legend of the classes in the frame
         /l5/seg/legend     String  every class: tag, name, color (latched JSON)
         /l5/seg/instances  bgr8    with instances:=true, YOLOv8s-seg's masks,
                                    one color per object (reading slides)

With rig:=true it also segments l5_bev_demo's four roof cameras and publishes
/l5/<camera>/semantic_net (mono8, CARLA tags), which semantic_bev splats with
labels:=network.

The node takes the newest image each time it is free; while the network runs,
older images are dropped. The log says how many it kept.
"""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from l5_seg_demo.classes import CARLA_TAGS, COLORS_BGR, network_to_tags, tag_name
from l5_seg_demo.ros_util import spin
from l5_seg_demo.segformer import MODEL_ID, WEIGHTS_DIR, SegFormer

RIG_CAMERAS = ("front", "left", "right", "rear")
# Instance colors (BGR): one per object, cycled. Chosen to differ from each other.
INSTANCE_COLORS = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
                   (148, 103, 189), (140, 86, 75), (227, 119, 194), (23, 190, 207),
                   (188, 189, 34), (127, 127, 127)]


def image_to_bgr(msg: Image) -> np.ndarray:
    """sensor_msgs/Image (bgra8, bgr8 or rgb8) to an H x W x 3 BGR array."""
    channels = {"bgra8": 4, "bgr8": 3, "rgb8": 3, "rgba8": 4}.get(msg.encoding)
    if channels is None:
        raise ValueError(f"unsupported image encoding '{msg.encoding}'")
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    arr = arr[:, :msg.width * channels].reshape(msg.height, msg.width, channels)
    if msg.encoding.startswith("rgb"):
        return np.ascontiguousarray(arr[:, :, 2::-1])
    return np.ascontiguousarray(arr[:, :, :3])


def array_msg(arr: np.ndarray, header, encoding: str) -> Image:
    msg = Image()
    msg.header = header
    msg.height, msg.width = arr.shape[:2]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = arr.strides[0]
    msg.data = np.ascontiguousarray(arr).tobytes()
    return msg


def draw_legend(img: np.ndarray, tags: np.ndarray, min_share: float = 0.002) -> None:
    """A legend, top left, of the classes that cover at least min_share of the frame."""
    counts = np.bincount(tags.ravel(), minlength=256)
    present = [t for t in np.argsort(-counts) if counts[t] >= min_share * tags.size]
    scale = img.shape[1] / 1280.0
    row = int(26 * scale)
    font = 0.6 * scale
    x0, y0 = int(10 * scale), int(10 * scale)
    width = int(230 * scale)
    cv2.rectangle(img, (x0 - 4, y0 - 4), (x0 + width, y0 + row * len(present) + 4),
                  (255, 255, 255), -1)
    for k, t in enumerate(present):
        y = y0 + k * row
        color = tuple(int(c) for c in COLORS_BGR[t])
        cv2.rectangle(img, (x0, y + 3), (x0 + int(30 * scale), y + row - 5), color, -1)
        share = 100.0 * counts[t] / tags.size
        cv2.putText(img, f"{tag_name(t)}  {share:.1f}%", (x0 + int(38 * scale), y + row - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), max(1, int(scale)), cv2.LINE_AA)


class SegNode(Node):

    def __init__(self) -> None:
        super().__init__("seg_node")
        self.declare_parameter("image_topic", "/carla/ego_vehicle/rgb_front/image")
        self.declare_parameter("model_id", MODEL_ID)
        self.declare_parameter("weights_dir", WEIGHTS_DIR)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("half", True)
        self.declare_parameter("overlay_alpha", 0.5)
        self.declare_parameter("instances", False)
        self.declare_parameter("yolo_weights", "~/.cache/enpm818z-weights/yolov8s-seg.pt")
        self.declare_parameter("yolo_conf", 0.25)
        self.declare_parameter("rig", False)
        self.declare_parameter("report_every", 10.0)

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        self.alpha = float(p("overlay_alpha"))
        self.net = SegFormer(p("model_id"), str(p("weights_dir")), p("device"), bool(p("half")))
        self.get_logger().info(
            f"{p('model_id')} on {self.net.device}"
            f"{' (float16)' if self.net.half else ''}")

        self.yolo = None
        if p("instances"):
            import os
            from ultralytics import YOLO
            self.yolo = YOLO(os.path.expanduser(p("yolo_weights")))
            self.yolo_conf = float(p("yolo_conf"))

        sensor_qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_labels = self.create_publisher(Image, "/l5/seg/labels", 2)
        self.pub_classes = self.create_publisher(Image, "/l5/seg/classes", 2)
        self.pub_overlay = self.create_publisher(Image, "/l5/seg/overlay", 2)
        self.pub_instances = self.create_publisher(Image, "/l5/seg/instances", 2)
        legend = self.create_publisher(String, "/l5/seg/legend", latched)
        legend.publish(String(data=json.dumps(
            [{"tag": t, "name": n, "rgb": list(c)} for t, (n, c) in CARLA_TAGS.items()])))

        topic = str(p("image_topic"))
        if topic:
            self.create_subscription(Image, topic, self._on_front, sensor_qos)
        self.pub_rig = {}
        if p("rig"):
            for name in RIG_CAMERAS:
                self.pub_rig[name] = self.create_publisher(
                    Image, f"/l5/{name}/semantic_net", sensor_qos)
                self.create_subscription(
                    Image, f"/l5/{name}/rgb",
                    lambda m, n=name: self._on_rig(n, m), sensor_qos)

        self.times = {"front": [], "rig": [], "yolo": []}
        self.create_timer(float(p("report_every")), self._report)

    # ---------------------------------------------------------- the network
    def _segment(self, bgr: np.ndarray, key: str) -> np.ndarray:
        t0 = time.perf_counter()
        train_ids = self.net(bgr[:, :, ::-1])            # the network takes RGB
        self.net.sync()
        self.times[key].append(time.perf_counter() - t0)
        return network_to_tags(train_ids)

    def _on_front(self, msg: Image) -> None:
        bgr = image_to_bgr(msg)
        tags = self._segment(bgr, "front")
        if not self.context.ok():
            return
        self.pub_labels.publish(array_msg(tags, msg.header, "mono8"))
        colors = COLORS_BGR[tags]
        self.pub_classes.publish(array_msg(colors, msg.header, "bgr8"))
        overlay = cv2.addWeighted(bgr, 1.0 - self.alpha, colors, self.alpha, 0.0)
        draw_legend(overlay, tags)
        self.pub_overlay.publish(array_msg(overlay, msg.header, "bgr8"))
        if self.yolo is not None:
            self.pub_instances.publish(array_msg(self._instances(bgr), msg.header, "bgr8"))

    def _on_rig(self, name: str, msg: Image) -> None:
        tags = self._segment(image_to_bgr(msg), "rig")
        if self.context.ok():
            self.pub_rig[name].publish(array_msg(tags, msg.header, "mono8"))

    def _instances(self, bgr: np.ndarray) -> np.ndarray:
        """YOLOv8s-seg: each object's own mask, its own color, its box and class."""
        t0 = time.perf_counter()
        result = self.yolo.predict(bgr, retina_masks=True, conf=self.yolo_conf,
                                   verbose=False)[0]
        self.times["yolo"].append(time.perf_counter() - t0)
        out = bgr.copy()
        if result.masks is None:
            return out
        masks = result.masks.data.cpu().numpy() > 0.5
        for k, (mask, box, cls, conf) in enumerate(zip(
                masks, result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy())):
            color = np.array(INSTANCE_COLORS[k % len(INSTANCE_COLORS)], dtype=np.float32)
            out[mask] = (0.45 * out[mask] + 0.55 * color).astype(np.uint8)
            x1, y1, x2, y2 = (int(v) for v in box)
            c = tuple(int(v) for v in color)
            cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
            cv2.putText(out, f"{result.names[int(cls)]} {conf:.2f}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
        return out

    def _report(self) -> None:
        parts = []
        for key, label in (("front", "front camera"), ("rig", "rig cameras"),
                           ("yolo", "YOLOv8s-seg")):
            t = self.times[key]
            if t:
                parts.append(f"{label}: {len(t)} images, {1000 * np.mean(t):.1f} ms each")
            self.times[key] = []
        if parts:
            self.get_logger().info("; ".join(parts))
        else:
            self.get_logger().info("waiting for images (is l2_carla_demo running?)")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SegNode()
    try:
        spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
