#!/usr/bin/env python3
"""A LiDAR detector with no learning: one detection per cluster of points.

The tracker needs detections; this node makes them from the L2 bridge's LiDAR,
in the simplest way that works, so every step can be read:

  Step 1. Move the points into the AV's frame and drop the ones that hit the
          AV itself (its roof and hood are inside the lowest beams).
  Step 2. Drop the ground: keep points between ground_max_z and
          obstacle_max_z above the road (flat roads only).
  Step 3. Mark every cell of a 2D grid (cell_size, 0.4 m by default) that holds
          a point, and join touching cells, diagonals included: each connected
          group of cells is one cluster.
  Step 4. One detection per cluster: its centroid, the mean of its points, and
          its extent. Clusters with too few points, or wider than max_extent
          (walls, buildings), are dropped.
  Step 5. Move the centroid into the fixed frame 'map', the frame the tracker
          works in, with the AV's pose at the sweep's time stamp.

The centroid falls short of the object's center: the LiDAR only sees the
faces turned toward the AV (L5, Frustum Association, Step 3). Every
detection carries the same R, measurement_sigma in x and y.

Subscribes: /carla/ego_vehicle/lidar (sensor_msgs/PointCloud2), TF.
Publishes:  /l5/detections          vision_msgs/Detection3DArray, frame map
            /l5/detections/markers  visualization_msgs/MarkerArray
"""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from scipy import ndimage
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from l5_tracking_demo.ros_util import make_detection, spin, yaw_from_quaternion


def matrix_from_transform(tf) -> np.ndarray:
    """4 x 4 homogeneous matrix from a geometry_msgs/Transform."""
    q, t = tf.rotation, tf.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    M = np.eye(4)
    M[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    M[:3, 3] = [t.x, t.y, t.z]
    return M


class Detector(Node):
    """Steps 1 to 5 above. l5_box_demo's box_detector subclasses it: same
    clusters (_clusters), a rotated box instead of a centroid (_detect)."""

    def __init__(self, name: str = "detector", out_topic: str = "/l5/detections") -> None:
        super().__init__(name)
        p = self.declare_parameter
        p("lidar_topic", "/carla/ego_vehicle/lidar")
        p("output_topic", out_topic)
        p("fixed_frame", "map")
        p("ego_frame", "ego_vehicle")
        p("ego_length", 4.8)
        p("ego_width", 2.2)
        p("ground_max_z", 0.3)
        p("obstacle_max_z", 2.5)
        p("max_range", 40.0)
        p("cell_size", 0.4)
        p("min_points", 5)
        p("max_extent", 7.0)
        p("measurement_sigma", 0.5)
        g = lambda name: self.get_parameter(name).value
        self.fixed, self.ego_frame = g("fixed_frame"), g("ego_frame")
        self.ego_half = (g("ego_length") / 2.0 + 0.2, g("ego_width") / 2.0 + 0.2)
        self.z_lo, self.z_hi = g("ground_max_z"), g("obstacle_max_z")
        self.max_range, self.cell = g("max_range"), g("cell_size")
        self.min_points, self.max_extent = int(g("min_points")), g("max_extent")
        self.sigma = g("measurement_sigma")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.ego_from_lidar = None
        self.pending: list[PointCloud2] = []       # sweeps waiting for their TF
        qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(PointCloud2, g("lidar_topic"), self._on_cloud, qos)
        out_topic = g("output_topic")
        self.pub = self.create_publisher(Detection3DArray, out_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, out_topic + "/markers", 10)
        self.create_timer(0.02, self._drain)
        self.sweeps = 0
        self.total = 0

    # ------------------------------------------------------------------ input
    def _on_cloud(self, msg: PointCloud2) -> None:
        if not self.context.ok():
            return
        self.pending.append(msg)
        self._drain()

    def _drain(self) -> None:
        """Process sweeps in order, as soon as the AV's pose at their stamp is known.

        The bridge sends the pose (TF map -> ego_vehicle) right after each tick,
        so a sweep can arrive a moment before the pose that goes with it. Using
        the latest pose instead would place every detection one tick off: 0.5 m
        at 10 m/s.
        """
        while self.pending and self.context.ok():
            msg = self.pending[0]
            if self.ego_from_lidar is None:
                try:
                    tf = self.tf_buffer.lookup_transform(self.ego_frame, msg.header.frame_id, Time())
                except Exception:
                    self.get_logger().info("waiting for the LiDAR mount in TF",
                                           throttle_duration_sec=5.0)
                    return
                self.ego_from_lidar = matrix_from_transform(tf.transform)
            stamp = Time.from_msg(msg.header.stamp)
            if self.tf_buffer.can_transform(self.fixed, self.ego_frame, stamp):
                tf = self.tf_buffer.lookup_transform(self.fixed, self.ego_frame, stamp)
                self.pending.pop(0)
                self._detect(msg, matrix_from_transform(tf.transform),
                             yaw_from_quaternion(tf.transform.rotation))
            elif len(self.pending) > 10:         # its pose never came: give up on it
                self.pending.pop(0)
            else:
                return

    # -------------------------------------------------------------- detection
    def _clusters(self, msg: PointCloud2) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Steps 1 to 3: (x, y, z) of each cluster's points, in the AV's frame."""
        pts = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, msg.point_step // 4)
        ego = (self.ego_from_lidar @ np.c_[pts[:, :3], np.ones(len(pts))].T).T
        x, y, z = ego[:, 0], ego[:, 1], ego[:, 2]
        # Step 1 and 2: not the AV itself, not the ground, not too far.
        own = (np.abs(x) < self.ego_half[0]) & (np.abs(y) < self.ego_half[1])
        keep = (~own) & (z > self.z_lo) & (z < self.z_hi) & (np.hypot(x, y) < self.max_range)
        x, y, z = x[keep], y[keep], z[keep]

        # Step 3: occupied cells, then connected groups of cells.
        n = int(np.ceil(2 * self.max_range / self.cell))
        i = np.clip(((x + self.max_range) / self.cell).astype(int), 0, n - 1)
        j = np.clip(((y + self.max_range) / self.cell).astype(int), 0, n - 1)
        occupied = np.zeros((n, n), dtype=bool)
        occupied[i, j] = True
        labels, _ = ndimage.label(occupied, structure=np.ones((3, 3)))
        point_label = labels[i, j]
        order = np.argsort(point_label)
        bounds = np.flatnonzero(np.diff(point_label[order])) + 1
        return [(x[m], y[m], z[m]) for m in np.split(order, bounds) if len(m)]

    def _detect(self, msg: PointCloud2, map_from_ego: np.ndarray, ego_yaw: float) -> None:
        # Step 4 and 5: one detection per cluster, in the fixed frame.
        out = Detection3DArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.fixed
        for cx, cy, cz in self._clusters(msg):
            if len(cx) < self.min_points:
                continue
            ex, ey = np.ptp(cx), np.ptp(cy)
            if max(ex, ey) > self.max_extent:
                continue
            centroid = map_from_ego @ np.array([cx.mean(), cy.mean(), 0.0, 1.0])
            out.detections.append(make_detection(
                centroid[0], centroid[1], centroid[2] + cz.max() / 2.0,
                (max(ex, 0.2), max(ey, 0.2), cz.max()), "object", self.sigma, ego_yaw))
        self.pub.publish(out)
        self.pub_markers.publish(self._markers(out))
        self.sweeps += 1
        self.total += len(out.detections)
        if self.sweeps % 100 == 1:
            self.get_logger().info(
                f"sweep {self.sweeps}: {len(out.detections)} detections "
                f"(mean {self.total / self.sweeps:.1f} per sweep)")

    def _markers(self, arr: Detection3DArray) -> MarkerArray:
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        for k, det in enumerate(arr.detections):
            m = Marker()
            m.header = arr.header
            m.ns, m.id, m.type, m.action = "detections", k, Marker.CUBE, Marker.ADD
            m.pose = det.bbox.center
            m.scale.x, m.scale.y, m.scale.z = det.bbox.size.x, det.bbox.size.y, det.bbox.size.z
            m.color.r, m.color.g, m.color.b, m.color.a = 0.3, 0.6, 1.0, 0.35
            ma.markers.append(m)
        return ma


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Detector()
    try:
        spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
