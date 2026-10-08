#!/usr/bin/env python3
"""The network's classes, painted onto the LiDAR points.

Slides: 'Semantic segmentation: a class for every pixel' and the Fusion
section's projection step. The camera knows what each pixel is; the LiDAR knows
where each point is. Project every LiDAR point into the camera image, read the
network's class at that pixel, and the point gets the class: a 3D map of
classes. This is the idea of PointPainting (Vora et al., CVPR 2020), done by
geometry alone.

    in   /carla/ego_vehicle/lidar                      the L2 bridge's LiDAR
         /l5/seg/labels                                seg_node's classes (mono8,
                                                       CARLA tags)
         /carla/ego_vehicle/rgb_front/camera_info      K, the intrinsics
         tf  ego_vehicle/lidar -> ego_vehicle/rgb_front_optical   the extrinsics

    out  /l5/seg/lidar_classes   PointCloud2 (x, y, z, rgb) in the LiDAR's frame:
                                 points the camera sees take the class's color,
                                 the rest stay gray

Pairing: the bridge ticks the simulator at 20 Hz, and the LiDAR turns once per
tick (rotation_frequency = 1 / delta), so each sweep and each image carry the
stamp of the same tick. Each label image is painted onto the sweep with the
same stamp; the network drops images while it is busy, so most sweeps are
never painted.

What goes wrong, on purpose visible: the LiDAR sits on the roof and the camera
at the windshield, so a point hidden from the camera behind a nearer object
still projects onto that object's pixels and takes its class. Points on the
ground just behind a car turn "car" at its edges.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from tf2_ros import Buffer, TransformException, TransformListener

from l5_seg_demo.classes import CARLA_TAGS
from l5_seg_demo.ros_util import spin, stamp_key

# CARLA tag -> packed RGB, the float32 layout RViz's RGB8 transformer reads.
_RGB = np.zeros(256, dtype=np.uint32)
for _tag, (_name, (_r, _g, _b)) in CARLA_TAGS.items():
    _RGB[_tag] = (_r << 16) | (_g << 8) | _b
UNSEEN_RGB = (110 << 16) | (110 << 8) | 110       # gray: outside the camera's view

FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
]


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """A unit quaternion as a 3 x 3 rotation matrix."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def paint(points: np.ndarray, rot: np.ndarray, trans: np.ndarray, k: np.ndarray,
          labels: np.ndarray, min_depth: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Give each LiDAR point the class of the pixel it projects to.

    points   N x 3, in the LiDAR's frame
    rot, trans   the LiDAR frame into the camera's optical frame (x right,
                 y down, z forward): p_cam = rot @ p + trans
    k        3 x 3 intrinsics, for an image of labels' size
    labels   H x W, one CARLA tag per pixel

    Returns the tag of every point (255 where the camera does not see it) and
    a mask of the points it sees: in front of the camera by min_depth meters,
    and inside the image.
    """
    cam = points @ rot.T + trans
    z = cam[:, 2]
    seen = z > min_depth
    u = np.full(len(points), -1, dtype=np.int64)
    v = np.full(len(points), -1, dtype=np.int64)
    u[seen] = np.floor(k[0, 0] * cam[seen, 0] / z[seen] + k[0, 2]).astype(np.int64)
    v[seen] = np.floor(k[1, 1] * cam[seen, 1] / z[seen] + k[1, 2]).astype(np.int64)
    h, w = labels.shape
    seen &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    tags = np.full(len(points), 255, dtype=np.uint8)
    tags[seen] = labels[v[seen], u[seen]]
    return tags, seen


class SegLidar(Node):
    def __init__(self) -> None:
        super().__init__("seg_lidar")
        self.declare_parameter("lidar_topic", "/carla/ego_vehicle/lidar")
        self.declare_parameter("labels_topic", "/l5/seg/labels")
        self.declare_parameter("camera_info_topic", "/carla/ego_vehicle/rgb_front/camera_info")
        self.declare_parameter("report_every", 10.0)
        p = lambda name: self.get_parameter(name).value  # noqa: E731

        sensor_qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(PointCloud2, "/l5/seg/lidar_classes", 2)
        self.create_subscription(PointCloud2, p("lidar_topic"), self._on_lidar, sensor_qos)
        self.create_subscription(Image, p("labels_topic"), self._on_labels, 2)
        self.create_subscription(CameraInfo, p("camera_info_topic"), self._on_info, latched)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.sweeps: deque = deque(maxlen=20)       # (stamp key, header, N x 3), 1 s at 20 Hz
        self.info = None
        self.extrinsic = None
        self.painted = 0
        self.unpaired = 0
        self.seen_share = []
        self.create_timer(float(p("report_every")), self._report)

    def _on_info(self, msg: CameraInfo) -> None:
        self.info = msg

    def _on_lidar(self, msg: PointCloud2) -> None:
        step = msg.point_step // 4
        pts = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, step)[:, :3]
        self.sweeps.append((stamp_key(msg.header.stamp), msg.header, pts))

    def _lookup(self, lidar_frame: str, camera_frame: str) -> bool:
        if self.extrinsic is not None:
            return True
        try:
            t = self.tf_buffer.lookup_transform(camera_frame, lidar_frame, Time())
        except TransformException as exc:
            self.get_logger().warn(f"no transform {lidar_frame} -> {camera_frame} yet: {exc}",
                                   throttle_duration_sec=5.0)
            return False
        q, tr = t.transform.rotation, t.transform.translation
        self.extrinsic = (quat_to_matrix(q.x, q.y, q.z, q.w),
                          np.array([tr.x, tr.y, tr.z]))
        self.get_logger().info(
            f"extrinsic {lidar_frame} -> {camera_frame}: translation "
            f"({tr.x:.2f}, {tr.y:.2f}, {tr.z:.2f}) m")
        return True

    def _on_labels(self, msg: Image) -> None:
        if self.info is None or not self.sweeps:
            return
        key = stamp_key(msg.header.stamp)
        match = next((s for s in self.sweeps if s[0] == key), None)
        if match is None:
            self.unpaired += 1
            return
        _, header, pts = match
        if not self._lookup(header.frame_id, msg.header.frame_id):
            return
        labels = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width]
        # K is for the camera's image; scale it if the labels come at another size.
        k = np.array(self.info.k, dtype=float).reshape(3, 3)
        k[0] *= msg.width / self.info.width
        k[1] *= msg.height / self.info.height
        tags, seen = paint(pts, *self.extrinsic, k, labels)
        rgb = np.where(seen, _RGB[tags], UNSEEN_RGB).astype(np.uint32)
        cloud = np.empty((len(pts), 4), dtype=np.float32)
        cloud[:, :3] = pts
        cloud[:, 3] = rgb.view(np.float32)

        out = PointCloud2()
        out.header = header
        out.height, out.width = 1, len(pts)
        out.fields = FIELDS
        out.is_bigendian = False
        out.point_step, out.row_step = 16, 16 * len(pts)
        out.is_dense = True
        out.data = cloud.tobytes()
        self.pub.publish(out)
        self.painted += 1
        self.seen_share.append(float(seen.mean()))

    def _report(self) -> None:
        if not self.painted:
            self.get_logger().info(
                f"nothing painted yet ({len(self.sweeps)} sweeps buffered, "
                f"camera info {'in' if self.info else 'missing'}, "
                f"{self.unpaired} label images with no sweep of the same tick)")
            return
        self.get_logger().info(
            f"painted {self.painted} sweeps; the camera sees "
            f"{100 * np.mean(self.seen_share):.0f}% of each sweep's points; "
            f"{self.unpaired} label images had no sweep of the same tick")
        self.painted, self.unpaired, self.seen_share = 0, 0, []


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SegLidar()
    spin(node)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
