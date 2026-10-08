#!/usr/bin/env python3
"""A detector you control: CARLA's true objects, with noise, misses and clutter added.

The LiDAR detector (detector_node.py) gives real detections, and real
detections mix several effects at once. This node lets you turn one effect up
at a time and watch the tracker:

  noise_sigma     Gaussian noise added to each true position, in x and y (m)
  miss_prob       chance that an object in range gives no detection this frame
  clutter_rate    mean number of false detections per frame (Poisson),
                  spread evenly over the disc of radius max_range
  flip_prob       chance that a detection carries the wrong class label:
                  'vehicle', 'bicycle' or 'other' instead of its own. With the
                  tracker's reset_on_class_change on, this is Tempe (L5).

Every detection reports R = noise_sigma^2 I, so the filter's R is right; set
reported_sigma to something else to give the tracker a wrong R on purpose.
Objects: every vehicle (and, with include_walkers, every pedestrian) within
max_range of the AV, seen or not: there is no occlusion here.

Publishes /l5/detections (vision_msgs/Detection3DArray, frame map), stamped
with the simulation time of the tick it describes, and /l5/detections/markers.
Run it instead of the LiDAR detector:

    ros2 launch l5_tracking_demo tracking.launch.py source:=truth
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from l5_tracking_demo.carla_truth import TruthWatcher
from l5_tracking_demo.ros_util import make_detection, sec_to_stamp, spin

LABELS = ("vehicle", "bicycle", "other")


class TruthDetector(Node):

    def __init__(self) -> None:
        super().__init__("truth_detector")
        p = self.declare_parameter
        p("host", "localhost")
        p("port", 2000)
        p("role_name", "ego")
        p("fixed_frame", "map")
        p("max_range", 40.0)
        p("publish_every", 2)          # every 2nd tick: 10 Hz with the bridge's 20 Hz
        p("noise_sigma", 0.3)
        p("reported_sigma", -1.0)      # < 0: report noise_sigma
        p("miss_prob", 0.1)
        p("clutter_rate", 0.5)
        p("flip_prob", 0.0)
        p("include_walkers", False)
        p("seed", 1)
        g = lambda name: self.get_parameter(name).value
        self.frame = g("fixed_frame")
        self.max_range = g("max_range")
        self.every = max(1, int(g("publish_every")))
        self.sigma = g("noise_sigma")
        self.reported = g("reported_sigma") if g("reported_sigma") > 0 else self.sigma
        self.miss, self.clutter, self.flip = g("miss_prob"), g("clutter_rate"), g("flip_prob")
        self.walkers = g("include_walkers")
        self.rng = np.random.default_rng(int(g("seed")))
        self.truth = TruthWatcher(g("host"), int(g("port")), g("role_name"))
        self.pub = self.create_publisher(Detection3DArray, "/l5/detections", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/l5/detections/markers", 10)
        self.create_timer(0.01, self._publish_new)
        self.ticks = 0
        self.get_logger().info(
            f"noise {self.sigma} m, miss {self.miss}, clutter {self.clutter}/frame, "
            f"flip {self.flip}, range {self.max_range} m")

    def _publish_new(self) -> None:
        for t, truth in self.truth.take_new():
            self.ticks += 1
            if self.ticks % self.every or not self.context.ok():
                continue
            ego = next((o for o in truth if o[0] == self.truth.ego_id), None)
            if ego is None:
                self.get_logger().warn("no AV (role_name ego) in the world yet",
                                       throttle_duration_sec=5.0)
                continue
            self._publish(t, ego, truth)

    def _publish(self, t: float, ego: tuple, truth: list) -> None:
        out = Detection3DArray()
        out.header.stamp = sec_to_stamp(t)
        out.header.frame_id = self.frame
        ex, ey = ego[1], ego[2]
        for oid, x, y, _vx, _vy, yaw, kind in truth:
            if oid == ego[0] or (kind == "pedestrian" and not self.walkers):
                continue
            if math.hypot(x - ex, y - ey) > self.max_range or self.rng.random() < self.miss:
                continue
            label = kind
            if self.rng.random() < self.flip:
                label = str(self.rng.choice([lab for lab in LABELS if lab != kind]))
            nx, ny = self.rng.normal(0.0, self.sigma, 2)
            size = (4.5, 1.9, 1.5) if kind == "vehicle" else (0.6, 0.6, 1.8)
            out.detections.append(make_detection(x + nx, y + ny, size[2] / 2, size, label,
                                                 self.reported, yaw))
        for _ in range(self.rng.poisson(self.clutter)):
            r = self.max_range * math.sqrt(self.rng.random())      # even over the disc
            a = 2 * math.pi * self.rng.random()
            out.detections.append(make_detection(ex + r * math.cos(a), ey + r * math.sin(a),
                                                 0.5, (1.0, 1.0, 1.0), "other", self.reported))
        self.pub.publish(out)
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        for k, det in enumerate(out.detections):
            m = Marker()
            m.header = out.header
            m.ns, m.id, m.type, m.action = "detections", k, Marker.CUBE, Marker.ADD
            m.pose = det.bbox.center
            m.scale.x, m.scale.y, m.scale.z = det.bbox.size.x, det.bbox.size.y, det.bbox.size.z
            m.color.r, m.color.g, m.color.b, m.color.a = 0.3, 0.6, 1.0, 0.35
            ma.markers.append(m)
        self.pub_markers.publish(ma)

    def destroy_node(self) -> None:
        self.truth.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TruthDetector()
    try:
        spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
