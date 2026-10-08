#!/usr/bin/env python3
"""Grade the detector and the tracker against CARLA's ground truth.

A real AV has no ground truth; CARLA does, so we can measure what the
lecture only describes. For every frame, the objects that count are CARLA's
vehicles (and pedestrians, with include_walkers) within eval_range of the AV,
seen or hidden, the AV itself excluded. They are compared with the
detections and tracks stamped at the same simulation time.

Detector:
  detection rate   the share of those objects with a detection within
                   match_dist (one detection per object, Hungarian pairing)
Tracker, using the tracks the planner is told about (confirmed and coasting):
  recall           the share of those objects with a track within match_dist
  false tracks     tracks within eval_range with no object within match_dist.
                   With the LiDAR detector this includes real things that are
                   not CARLA actors: parked cars that are part of the map,
                   poles, trees, walls.
  ID switches      times an object's matched track ID changed from the one
                   it had the last time it was matched (as in MOTA)
  position error   distance from a matched track to its object's center, m
  speed error      |track velocity - true velocity|, m/s, for matched tracks

Prints a report every report_period seconds and writes it to `out` (JSON).

    ros2 run l5_tracking_demo evaluate --ros-args -p out:=eval.json
"""

from __future__ import annotations

import json
import math

import numpy as np
import rclpy
from rclpy.node import Node
from scipy.optimize import linear_sum_assignment
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray

from l5_tracking_demo.carla_truth import TruthWatcher
from l5_tracking_demo.ros_util import stamp_to_sec, spin


def pair(a: np.ndarray, b: np.ndarray, max_dist: float) -> list[tuple[int, int, float]]:
    """Closest one-to-one pairs between two point sets, each within max_dist."""
    if len(a) == 0 or len(b) == 0:
        return []
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(np.where(d <= max_dist, d, 1e6))
    return [(int(r), int(c), float(d[r, c])) for r, c in zip(rows, cols) if d[r, c] <= max_dist]


class Evaluate(Node):

    def __init__(self) -> None:
        super().__init__("evaluate")
        p = self.declare_parameter
        p("host", "localhost")
        p("port", 2000)
        p("role_name", "ego")
        p("eval_range", 30.0)
        p("match_dist", 3.0)
        p("include_walkers", False)
        p("report_period", 10.0)
        p("out", "")
        g = lambda name: self.get_parameter(name).value
        self.range, self.match = g("eval_range"), g("match_dist")
        self.walkers = g("include_walkers")
        self.out = g("out")
        self.truth = TruthWatcher(g("host"), int(g("port")), g("role_name"))
        self.create_subscription(Detection3DArray, "/l5/detections", self._on_det, 10)
        self.create_subscription(String, "/l5/tracks/json", self._on_tracks, 10)
        self.create_timer(0.05, self._drain)
        self.create_timer(float(g("report_period")), self._report)
        self.pending: list[tuple[str, float, object, int]] = []
        self.m = {"det_frames": 0, "det_objects": 0, "det_found": 0, "det_unmatched": 0,
                  "trk_frames": 0, "trk_objects": 0, "trk_found": 0, "false_tracks": 0,
                  "id_switches": 0, "pos_err_sum": 0.0, "speed_err_sum": 0.0}
        self.last_track: dict[int, int] = {}       # object id -> track id last matched
        self.tries = 0

    # --------------------------------------------------------------- inputs
    def _on_det(self, msg: Detection3DArray) -> None:
        self.pending.append(("det", stamp_to_sec(msg.header.stamp), msg, 0))

    def _on_tracks(self, msg: String) -> None:
        frame = json.loads(msg.data)
        self.pending.append(("trk", frame["stamp"], frame, 0))

    def _drain(self) -> None:
        keep = []
        for kind, t, data, tries in self.pending:
            truth = self.truth.at(t)
            if truth is None:
                if tries < 40:                         # wait up to 2 s for the tick
                    keep.append((kind, t, data, tries + 1))
                continue
            objects = self._objects(truth)
            if objects is None:
                continue
            (self._grade_det if kind == "det" else self._grade_trk)(objects, data)
        self.pending = keep

    def _objects(self, truth: list):
        ego = next((o for o in truth if o[0] == self.truth.ego_id), None)
        if ego is None:
            return None
        objs = [o for o in truth if o[0] != ego[0]
                and (o[6] == "vehicle" or (self.walkers and o[6] == "pedestrian"))
                and math.hypot(o[1] - ego[1], o[2] - ego[2]) <= self.range]
        return ego, objs

    # -------------------------------------------------------------- grading
    def _grade_det(self, objects, msg: Detection3DArray) -> None:
        ego, objs = objects
        pts = np.array([[d.bbox.center.position.x, d.bbox.center.position.y]
                        for d in msg.detections]).reshape(-1, 2)
        near = pts[np.hypot(pts[:, 0] - ego[1], pts[:, 1] - ego[2]) <= self.range] \
            if len(pts) else pts
        truth = np.array([[o[1], o[2]] for o in objs]).reshape(-1, 2)
        pairs = pair(truth, near, self.match)
        self.m["det_frames"] += 1
        self.m["det_objects"] += len(objs)
        self.m["det_found"] += len(pairs)
        self.m["det_unmatched"] += len(near) - len(pairs)

    def _grade_trk(self, objects, frame: dict) -> None:
        ego, objs = objects
        told = [trk for trk in frame["tracks"] if trk["status"] in ("confirmed", "coasting")
                and math.hypot(trk["x"] - ego[1], trk["y"] - ego[2]) <= self.range + self.match]
        truth = np.array([[o[1], o[2]] for o in objs]).reshape(-1, 2)
        pts = np.array([[trk["x"], trk["y"]] for trk in told]).reshape(-1, 2)
        pairs = pair(truth, pts, self.match)
        self.m["trk_frames"] += 1
        self.m["trk_objects"] += len(objs)
        self.m["trk_found"] += len(pairs)
        self.m["false_tracks"] += len(told) - len(pairs)
        for r, c, dist in pairs:
            oid, tid = objs[r][0], told[c]["id"]
            if oid in self.last_track and self.last_track[oid] != tid:
                self.m["id_switches"] += 1
            self.last_track[oid] = tid
            self.m["pos_err_sum"] += dist
            self.m["speed_err_sum"] += math.hypot(told[c]["vx"] - objs[r][3],
                                                  told[c]["vy"] - objs[r][4])

    # --------------------------------------------------------------- report
    def summary(self) -> dict:
        m = self.m
        ratio = lambda a, b: round(a / b, 3) if b else None
        return {
            "detector_frames": m["det_frames"],
            "objects_per_frame": ratio(m["det_objects"], m["det_frames"]),
            "detection_rate": ratio(m["det_found"], m["det_objects"]),
            "unmatched_detections_per_frame": ratio(m["det_unmatched"], m["det_frames"]),
            "tracker_frames": m["trk_frames"],
            "recall": ratio(m["trk_found"], m["trk_objects"]),
            "false_tracks_per_frame": ratio(m["false_tracks"], m["trk_frames"]),
            "id_switches": m["id_switches"],
            "mean_position_error_m": ratio(m["pos_err_sum"], m["trk_found"]),
            "mean_speed_error_mps": ratio(m["speed_err_sum"], m["trk_found"]),
        }

    def _report(self) -> None:
        s = self.summary()
        self.get_logger().info(json.dumps(s))
        if self.out:
            with open(self.out, "w") as f:
                json.dump(s, f, indent=1)

    def destroy_node(self) -> None:
        self.truth.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Evaluate()
    try:
        spin(node)
    finally:
        node._report()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
