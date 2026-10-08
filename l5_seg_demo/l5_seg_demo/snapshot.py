#!/usr/bin/env python3
"""Save one frame as a picture: camera, network, CARLA's truth, where they agree.

Waits for the three images of one tick (same time stamp), skips the first
`skip` such ticks, draws the four panels and exits:

    1  the bridge's front camera, what the network sees
    2  the network's classes (seg_node), CARLA's colors
    3  CARLA's true classes (seg_truth), same colors
    4  per pixel: green where the two agree, red where they differ, gray where
       the pixel is not graded (a CARLA class Cityscapes does not grade).
       Lane markings count as road, as in seg_eval.

It also writes the frame's IoU per class to a JSON file of the same name, and
the three images to an .npz file, so the picture can be drawn again offline:

    ros2 run l5_seg_demo snapshot --ros-args -p out:=seg.png -p skip:=20
    ros2 run l5_seg_demo snapshot --ros-args -p redraw:=seg.npz -p out:=seg2.png
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, QoSReliabilityPolicy  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402

from l5_seg_demo.classes import (CITYSCAPES, COLORS_RGB, IGNORE, tag_name,  # noqa: E402
                                 tags_to_graded)
from l5_seg_demo.ros_util import spin, stamp_key  # noqa: E402
from l5_seg_demo.seg_eval import ConfusionMatrix  # noqa: E402
from l5_seg_demo.seg_node import image_to_bgr  # noqa: E402
from l5_seg_demo.seg_eval_node import mono8  # noqa: E402

AGREE, DIFFER, NOT_GRADED = (46, 160, 67), (214, 39, 40), (200, 200, 200)


class Snapshot(Node):

    def __init__(self) -> None:
        super().__init__("seg_snapshot")
        self.declare_parameter("out", "seg.png")
        self.declare_parameter("skip", 20)
        self.declare_parameter("image_topic", "/carla/ego_vehicle/rgb_front/image")
        self.declare_parameter("redraw", "")
        self.out = self.get_parameter("out").value
        self.skip = int(self.get_parameter("skip").value)
        self.frames: dict[int, dict] = {}
        self.matched = 0
        self.done = False
        camera_qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        ours = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)
        for kind, topic, qos in (("rgb", self.get_parameter("image_topic").value, camera_qos),
                                 ("net", "/l5/seg/labels", ours),
                                 ("truth", "/l5/seg/truth_tags", ours)):
            self.create_subscription(Image, topic, lambda m, k=kind: self._on(k, m), qos)

    def _on(self, kind: str, msg: Image) -> None:
        if self.done:
            return
        key = stamp_key(msg.header.stamp)
        img = image_to_bgr(msg)[:, :, ::-1].copy() if kind == "rgb" else mono8(msg).copy()
        frame = self.frames.setdefault(key, {})
        frame[kind] = img
        for old in sorted(self.frames)[:-60]:
            del self.frames[old]
        if len(frame) < 3:
            return
        self.matched += 1
        del self.frames[key]
        if self.matched <= self.skip:
            return
        t = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        np.savez_compressed(self.out.rsplit(".", 1)[0] + ".npz", rgb=frame["rgb"],
                            net=frame["net"], truth=frame["truth"], t=t)
        stats = draw(frame["rgb"], frame["net"], frame["truth"], self.out, t)
        with open(self.out.rsplit(".", 1)[0] + ".json", "w") as f:
            json.dump(stats, f, indent=1)
        self.get_logger().info(f"wrote {self.out}: mIoU {stats['miou']:.3f} on this frame")
        self.done = True
        raise SystemExit


def draw(rgb: np.ndarray, net: np.ndarray, truth: np.ndarray, out: str, t: float) -> dict:
    g_truth, g_net = tags_to_graded(truth), tags_to_graded(net)
    graded = g_truth != IGNORE
    agree = np.empty(truth.shape + (3,), dtype=np.uint8)
    agree[:] = NOT_GRADED
    agree[graded & (g_truth == g_net)] = AGREE
    agree[graded & (g_truth != g_net)] = DIFFER
    cm = ConfusionMatrix()
    cm.add(g_truth, g_net)
    stats = cm.report()
    stats["sim_time_s"] = round(t, 3)
    stats["agree_share_of_graded"] = round(float((g_truth == g_net)[graded].mean()), 4)
    stats["graded_share_of_image"] = round(float(graded.mean()), 4)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.4))
    panels = [(rgb, "1. The front camera: what the network sees"),
              (COLORS_RGB[net], "2. The network's classes (SegFormer-B0, Cityscapes)"),
              (COLORS_RGB[truth], "3. CARLA's true classes (semantic camera)"),
              (agree, "4. Where 2 and 3 agree (lane marking counted as road)")]
    for ax, (img, title) in zip(axes.ravel(), panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.set_axis_off()

    share = (np.bincount(net.ravel(), minlength=256)
             + np.bincount(truth.ravel(), minlength=256)) / (2.0 * net.size)
    tags = [t for t in np.argsort(-share) if share[t] >= 0.001]
    handles = [Patch(facecolor=COLORS_RGB[t] / 255.0, edgecolor="0.3", label=tag_name(t))
               for t in tags]
    handles += [Patch(facecolor=np.array(c) / 255.0, edgecolor="0.3", label=lab)
                for c, lab in ((AGREE, "agree"), (DIFFER, "differ"),
                               (NOT_GRADED, "not graded"))]
    rows = -(-len(handles) // 9)
    fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=10.5, frameon=False,
               borderaxespad=0.3)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.965, bottom=0.035 + 0.035 * rows,
                        wspace=0.03, hspace=0.12)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    stats["classes_in_legend"] = [tag_name(t) for t in tags]
    stats["cityscapes_classes"] = list(CITYSCAPES)
    return stats


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Snapshot()
    redraw = node.get_parameter("redraw").value
    if redraw:
        d = np.load(redraw)
        stats = draw(d["rgb"], d["net"], d["truth"], node.out, float(d["t"]))
        print(f"wrote {node.out}: mIoU {stats['miou']:.3f} on this frame")
        node.destroy_node()
        rclpy.shutdown()
        return
    try:
        spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
