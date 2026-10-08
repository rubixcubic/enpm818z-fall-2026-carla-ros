#!/usr/bin/env python3
"""3D boxes from the LiDAR, with no learning: clusters, then L-shape fitting.

The 3D Detection section of L5 says a detector reports, per object, 7 numbers:
center (x, y, z), length, width, height and heading. This node produces them in
the simplest way that works, so every step can be read:

  Steps 1 to 3 (l5_tracking_demo's detector, shared code): points in the AV's
          frame, the AV's own points and the ground dropped, clusters on a 2D
          grid of touching cells.
  Step 4. For each cluster, fit a rotated rectangle seen from above by L-shape
          fitting (lshape.py: Zhang et al., IEEE IV 2017, the closeness
          criterion). Length = the longer side; the heading runs along it.
  Step 5. Height = the highest point of the cluster above the road (the box
          stands on the road); center z = half of it.
  Step 6. Move the box into the fixed frame 'map' with the AV's pose at the
          sweep's time stamp.

The heading is known modulo 180 degrees: one sweep cannot tell the front of a
car from its back. Every published yaw is in [-90, 90) degrees, and the RViz
label says "heading mod 180".

With evaluate:=true the node also grades every sweep against CARLA's true
vehicle boxes at the same simulation time (box_eval.py, the nuScenes
definitions), and writes a report.

Subscribes: /carla/ego_vehicle/lidar, TF.
Publishes:  /l5/boxes          vision_msgs/Detection3DArray, frame map: center,
                               size (l, w, h), orientation (yaw mod 180 deg)
            /l5/boxes/markers  visualization_msgs/MarkerArray
            /l5/boxes/debug    std_msgs/String (JSON): this sweep's points,
                               boxes and centroids in the AV's frame, for the
                               snapshot tool
"""

from __future__ import annotations

import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from l5_tracking_demo.detector_node import Detector
from l5_tracking_demo.ros_util import make_detection, spin, stamp_to_sec

from l5_box_demo.box_eval import BoxScore
from l5_box_demo.lshape import Box2D, corners, fit_lshape, wrap_half_turn


class BoxDetector(Detector):

    def __init__(self) -> None:
        super().__init__("box_detector", "/l5/boxes")
        p = self.declare_parameter
        p("angle_step_deg", 1.0)
        p("d0", 0.1)
        p("d_max", 0.4)
        p("evaluate", False)
        p("eval_range", 30.0)
        p("visible_min_points", 5)
        p("include_parked", True)
        p("carla_host", "localhost")
        p("carla_port", 2000)
        p("out", "")
        p("debug_every", 5)
        g = lambda name: self.get_parameter(name).value
        self.step, self.d0, self.d_max = g("angle_step_deg"), g("d0"), g("d_max")
        self.pub_debug = self.create_publisher(String, "/l5/boxes/debug", 10)
        self.debug_every = int(g("debug_every"))
        self.evaluate = bool(g("evaluate"))
        self.score, self.truth = None, None
        if self.evaluate:
            from l5_box_demo.box_truth import BoxTruth
            self.truth = BoxTruth(g("carla_host"), int(g("carla_port")),
                                  include_parked=bool(g("include_parked")))
            self.score = BoxScore()
            self.eval_range = g("eval_range")
            self.visible_min = int(g("visible_min_points"))
            self.out = g("out")
            self.create_timer(10.0, self._report)
            self.get_logger().info(f"evaluating against CARLA; {len(self.truth.parked)} "
                                   "parked map vehicles added to the truth")

    # ------------------------------------------------------------- detection
    def _detect(self, msg: PointCloud2, map_from_ego: np.ndarray, ego_yaw: float) -> None:
        out = Detection3DArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.fixed
        ego_boxes, all_pts = [], []
        for cx, cy, cz in self._clusters(msg):
            all_pts.append(np.c_[cx, cy])
            if len(cx) < self.min_points:
                continue
            box = fit_lshape(np.c_[cx, cy], self.step, self.d0, self.d_max)
            if box.length > self.max_extent:          # walls, buildings
                continue
            height = float(cz.max())
            ego_boxes.append((box, height, float(cx.mean()), float(cy.mean())))
            center = map_from_ego @ np.array([box.cx, box.cy, height / 2.0, 1.0])
            yaw = wrap_half_turn(box.yaw + ego_yaw)
            out.detections.append(make_detection(
                center[0], center[1], center[2],
                (max(box.length, 0.1), max(box.width, 0.1), max(height, 0.1)),
                "object", self.sigma, yaw))
        self.pub.publish(out)
        self.pub_markers.publish(self._box_markers(out))
        self.sweeps += 1
        self.total += len(out.detections)
        if self.debug_every and self.sweeps % self.debug_every == 0:
            self._publish_debug(msg, ego_boxes, all_pts, map_from_ego, ego_yaw)
        if self.evaluate:
            self._grade(msg, ego_boxes, all_pts, map_from_ego, ego_yaw)
        if self.sweeps % 100 == 1:
            self.get_logger().info(
                f"sweep {self.sweeps}: {len(out.detections)} boxes "
                f"(mean {self.total / self.sweeps:.1f} per sweep)")

    # ------------------------------------------------------------ evaluation
    def _truth_in_ego(self, t: float, map_from_ego: np.ndarray, ego_yaw: float):
        boxes = self.truth.boxes_at(t)
        if boxes is None:
            return None
        ego_from_map = np.linalg.inv(map_from_ego)
        out = []
        for o in boxes:
            p = ego_from_map @ np.array([o["x"], o["y"], 0.0, 1.0])
            if math.hypot(p[0], p[1]) > self.eval_range:
                continue
            out.append({"id": o["id"], "x": float(p[0]), "y": float(p[1]),
                        "yaw": wrap_half_turn(o["yaw"] - ego_yaw), "size": o["size"]})
        return out

    def _grade(self, msg, ego_boxes, all_pts, map_from_ego, ego_yaw) -> None:
        truth = self._truth_in_ego(stamp_to_sec(msg.header.stamp), map_from_ego, ego_yaw)
        if truth is None:
            return
        pts = np.vstack(all_pts) if all_pts else np.zeros((0, 2))
        for o in truth:              # visible: enough kept points inside its footprint
            c, s = math.cos(o["yaw"]), math.sin(o["yaw"])
            d = pts - np.array([o["x"], o["y"]])
            u, v = d @ np.array([c, s]), d @ np.array([-s, c])
            inside = (np.abs(u) <= o["size"][0] / 2 + 0.3) & (np.abs(v) <= o["size"][1] / 2 + 0.3)
            o["visible"] = int(inside.sum()) >= self.visible_min
        boxes = [{"x": b.cx, "y": b.cy, "yaw": b.yaw, "size": (b.length, b.width, h),
                  "centroid": (mx, my)}
                 for b, h, mx, my in ego_boxes if math.hypot(b.cx, b.cy) <= self.eval_range]
        self.score.add(truth, boxes)

    def _report(self) -> None:
        if not self.score or not self.score.frames:
            return
        s = self.score.summary()
        self.get_logger().info(json.dumps(s))
        if self.out:
            with open(self.out, "w") as f:
                json.dump(s, f, indent=1)

    # --------------------------------------------------------------- outputs
    def _publish_debug(self, msg, ego_boxes, all_pts, map_from_ego, ego_yaw) -> None:
        d = {"t": stamp_to_sec(msg.header.stamp),
             "points": np.round(np.vstack(all_pts), 2).tolist() if all_pts else [],
             "boxes": [{"corners": np.round(corners(b), 3).tolist(), "cx": b.cx, "cy": b.cy,
                        "length": b.length, "width": b.width, "height": h, "yaw": b.yaw,
                        "centroid": [mx, my]} for b, h, mx, my in ego_boxes]}
        if self.truth is not None:
            tr = self._truth_in_ego(d["t"], map_from_ego, ego_yaw) or []
            d["truth"] = [{"corners": np.round(corners(Box2D(o["x"], o["y"], o["size"][0],
                                                             o["size"][1], o["yaw"])), 3).tolist()}
                          for o in tr]
        self.pub_debug.publish(String(data=json.dumps(d)))

    def _box_markers(self, arr: Detection3DArray) -> MarkerArray:
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        for k, det in enumerate(arr.detections):
            c = det.bbox.center
            box = Marker()
            box.header = arr.header
            box.ns, box.id, box.type, box.action = "boxes", k, Marker.CUBE, Marker.ADD
            box.pose = c
            box.scale.x, box.scale.y, box.scale.z = det.bbox.size.x, det.bbox.size.y, det.bbox.size.z
            box.color.r, box.color.g, box.color.b, box.color.a = 0.52, 0.72, 0.89, 0.45
            ma.markers.append(box)
            # the heading axis, both ways: front and back cannot be told apart
            q = c.orientation
            yaw = math.atan2(2.0 * q.w * q.z, 1.0 - 2.0 * q.z * q.z)
            hl = det.bbox.size.x / 2.0 + 0.6
            axis = Marker()
            axis.header = arr.header
            axis.ns, axis.id, axis.type, axis.action = "heading", k, Marker.LINE_LIST, Marker.ADD
            axis.scale.x = 0.08
            axis.color.r, axis.color.g, axis.color.b, axis.color.a = 0.0, 0.0, 0.6, 1.0
            p0, p1 = Point(), Point()
            p0.x, p0.y = c.position.x - hl * math.cos(yaw), c.position.y - hl * math.sin(yaw)
            p1.x, p1.y = c.position.x + hl * math.cos(yaw), c.position.y + hl * math.sin(yaw)
            p0.z = p1.z = c.position.z + det.bbox.size.z / 2.0
            axis.points = [p0, p1]
            ma.markers.append(axis)
            label = Marker()
            label.header = arr.header
            label.ns, label.id, label.type, label.action = "size", k, Marker.TEXT_VIEW_FACING, Marker.ADD
            label.pose.position.x, label.pose.position.y = c.position.x, c.position.y
            label.pose.position.z = c.position.z + det.bbox.size.z / 2.0 + 0.6
            label.scale.z = 0.5
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = (f"{det.bbox.size.x:.1f} x {det.bbox.size.y:.1f} x {det.bbox.size.z:.1f} m, "
                          f"heading {math.degrees(yaw):.0f} deg mod 180")
            ma.markers.append(label)
        return ma

    def destroy_node(self) -> None:
        if self.truth is not None:
            self.truth.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BoxDetector()
    try:
        spin(node)
    finally:
        if node.evaluate:
            node._report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
