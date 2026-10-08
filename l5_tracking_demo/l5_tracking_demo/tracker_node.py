#!/usr/bin/env python3
"""The multi-object tracker of L5, as a ROS 2 node.

Every detection message is one frame of the loop on the slide 'Every frame,
the same loop':
  1. predict every track to the frame's time stamp (constant velocity, L3);
  2. associate: epsilon = nu^T S^-1 nu for every track and detection, gate at
     9.21, then GNN (Hungarian) or NN (each track in turn, by ID);
  3. update the matched tracks;
  4. manage the rest: tentative -> confirmed after M of N, confirmed ->
     coasting on a miss, coasting -> deleted after max_coast_time.
The algorithm itself is in tracker_core.py; this file only converts messages.

Subscribes: /l5/detections (vision_msgs/Detection3DArray, frame map)
Publishes:  /l5/tracks           vision_msgs/Detection3DArray: the tracks the
                                 planner is told about (confirmed and coasting),
                                 id = track ID
            /l5/tracks/json      std_msgs/String: every track, with its status,
                                 velocity and covariance, plus the frame's counts
            /l5/tracks/markers   visualization_msgs/MarkerArray for RViz:
                                 gray tentative, green confirmed, orange coasting;
                                 1-sigma position ellipse, velocity arrow (1 s
                                 of travel), ID
"""

from __future__ import annotations

import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

from l5_tracking_demo.ros_util import detections_from_msg, make_detection, stamp_to_sec, spin
from l5_tracking_demo.tracker_core import (
    COASTING, CONFIRMED, TENTATIVE, MultiTracker)

COLORS = {TENTATIVE: (0.6, 0.6, 0.6), CONFIRMED: (0.1, 0.8, 0.2), COASTING: (1.0, 0.55, 0.0)}


class TrackerNode(Node):

    def __init__(self) -> None:
        super().__init__("tracker")
        p = self.declare_parameter
        p("detections_topic", "/l5/detections")
        p("association", "gnn")            # gnn or nn
        p("nn_order", "ascending")         # NN only: which track chooses first, by ID
        p("cost", "epsilon")               # epsilon, or epsilon + ln|S| (see README)
        p("confirmed_first", True)         # confirmed tracks choose before tentative ones
        p("gate", 9.21)
        p("confirm_m", 3)
        p("confirm_n", 5)
        p("max_coast_time", 1.0)
        p("init_vel_sigma", 5.0)
        p("accel_sigma", 3.0)
        p("measurement_sigma", 0.5)        # used when a detection carries no R
        p("reset_on_class_change", False)
        p("log_every", 50)
        g = lambda name: self.get_parameter(name).value
        self.tracker = MultiTracker(
            association=g("association"), nn_order=g("nn_order"), gate=g("gate"),
            confirm_m=int(g("confirm_m")), confirm_n=int(g("confirm_n")),
            max_coast_time=g("max_coast_time"), init_vel_sigma=g("init_vel_sigma"),
            accel_sigma=g("accel_sigma"), reset_on_class_change=g("reset_on_class_change"),
            cost=g("cost"), confirmed_first=g("confirmed_first"))
        self.default_sigma = g("measurement_sigma")
        self.log_every = int(g("log_every"))
        self.create_subscription(Detection3DArray, g("detections_topic"), self._on_detections, 10)
        self.pub_tracks = self.create_publisher(Detection3DArray, "/l5/tracks", 10)
        self.pub_json = self.create_publisher(String, "/l5/tracks/json", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/l5/tracks/markers", 10)
        self.frames = 0
        self.get_logger().info(
            f"association {g('association')}"
            + (f" ({g('nn_order')} IDs first)" if g("association") == "nn" else "")
            + f", cost {g('cost')}, confirmed first {g('confirmed_first')}, gate {g('gate')}, confirm {g('confirm_m')} of {g('confirm_n')}, "
              f"coast {g('max_coast_time')} s, reset_on_class_change {g('reset_on_class_change')}")

    def _on_detections(self, msg: Detection3DArray) -> None:
        if not self.context.ok():
            return
        t = stamp_to_sec(msg.header.stamp)
        if self.tracker.t is not None and t <= self.tracker.t:
            return                                # out of order or repeated
        rep = self.tracker.step(detections_from_msg(msg, self.default_sigma), t)
        self.frames += 1
        counts = self.tracker.counts()
        self._publish(msg, t, rep, counts)
        if self.frames % self.log_every == 1:
            self.get_logger().info(
                f"frame {self.frames} (t = {t:.2f} s): {rep.detections} detections, "
                f"{rep.matched} matched, {rep.started} started, {rep.deleted} deleted"
                + (f", {rep.resets} restarted on a class change" if rep.resets else "")
                + f" | tracks: {counts[TENTATIVE]} tentative, {counts[CONFIRMED]} confirmed, "
                  f"{counts[COASTING]} coasting")

    # ---------------------------------------------------------------- output
    def _publish(self, msg, t, rep, counts) -> None:
        tracks = []
        told = Detection3DArray()
        told.header = msg.header
        for trk in self.tracker.tracks:
            x, y, vx, vy = (float(v) for v in trk.kf.x)
            P = trk.kf.P[:2, :2]
            tracks.append({"id": trk.id, "status": trk.status, "class": trk.label,
                           "x": x, "y": y, "vx": vx, "vy": vy,
                           "P": [[float(P[0, 0]), float(P[0, 1])], [float(P[1, 0]), float(P[1, 1])]],
                           "age": trk.age, "detections": trk.detections})
            if trk.status in (CONFIRMED, COASTING):
                det = make_detection(x, y, 0.75, (1.0, 1.0, 1.5), trk.label,
                                     math.sqrt(max(P[0, 0], 1e-9)), math.atan2(vy, vx))
                det.id = str(trk.id)
                told.detections.append(det)
        frame = {"stamp": t, "frame_id": msg.header.frame_id,
                 "detections": rep.detections, "matched": rep.matched,
                 "started": rep.started, "deleted": rep.deleted, "resets": rep.resets,
                 "counts": counts, "tracks": tracks}
        self.pub_json.publish(String(data=json.dumps(frame)))
        self.pub_tracks.publish(told)
        self.pub_markers.publish(self._markers(msg.header))

    def _markers(self, header) -> MarkerArray:
        ma = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        k = 0
        for trk in self.tracker.tracks:
            r, g, b = COLORS[trk.status]
            x, y, vx, vy = trk.kf.x

            def marker(kind):
                nonlocal k
                m = Marker()
                m.header = header
                m.ns, m.id, m.type, m.action = "tracks", k, kind, Marker.ADD
                m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 1.0
                m.pose.orientation.w = 1.0
                k += 1
                ma.markers.append(m)
                return m

            dot = marker(Marker.SPHERE)
            dot.pose.position.x, dot.pose.position.y, dot.pose.position.z = x, y, 0.5
            dot.scale.x = dot.scale.y = dot.scale.z = 0.6
            # 1-sigma ellipse of the position, from the eigenvectors of P.
            ell = marker(Marker.LINE_STRIP)
            ell.scale.x = 0.08
            vals, vecs = np.linalg.eigh(trk.kf.P[:2, :2])
            axes = vecs * np.sqrt(np.maximum(vals, 0.0))
            for a in np.linspace(0, 2 * np.pi, 37):
                d = axes @ np.array([math.cos(a), math.sin(a)])
                ell.points.append(Point(x=float(x + d[0]), y=float(y + d[1]), z=0.3))
            # Velocity: an arrow as long as 1 s of travel.
            if trk.status != TENTATIVE:
                arrow = marker(Marker.ARROW)
                arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.15, 0.4, 0.4
                arrow.points = [Point(x=float(x), y=float(y), z=0.5),
                                Point(x=float(x + vx), y=float(y + vy), z=0.5)]
            text = marker(Marker.TEXT_VIEW_FACING)
            text.pose.position.x, text.pose.position.y, text.pose.position.z = x, y, 2.0
            text.scale.z = 1.0
            text.color.r = text.color.g = text.color.b = 1.0
            text.text = f"{trk.id}"
        return ma


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrackerNode()
    try:
        spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
