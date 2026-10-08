#!/usr/bin/env python3
"""Save one frame of the tracker as a PNG: a top view around the AV.

Waits for `after_frames` tracker frames (so tracks have had time to start),
then draws the next frame and exits:
  gray outlines   CARLA's vehicles at that instant (ground truth), with their IDs
  blue crosses    this frame's detections
  dots            tracks: gray tentative, green confirmed, orange coasting,
                  each labeled with its track ID
  dashed ellipse  each track's gate: the points where epsilon = 9.21, with
                  S = P + R (R from measurement_sigma)
  arrows          track velocity, drawn as 1 s of travel (confirmed, coasting)
The view is the AV's frame: x forward to the right, y left up the page, with
the AV's center at (0, 0).

    ros2 run l5_tracking_demo snapshot --ros-args -p out:=tracking.png
"""

from __future__ import annotations

import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402
import rclpy                                                      # noqa: E402
from matplotlib.patches import Ellipse, Polygon                   # noqa: E402
from rclpy.executors import ExternalShutdownException             # noqa: E402
from rcl_interfaces.msg import ParameterDescriptor                # noqa: E402
from rclpy.node import Node                                       # noqa: E402
from std_msgs.msg import String                                   # noqa: E402
from vision_msgs.msg import Detection3DArray                      # noqa: E402

from l5_tracking_demo.carla_truth import TruthWatcher             # noqa: E402
from l5_tracking_demo.ros_util import stamp_to_sec, time_key      # noqa: E402

COLORS = {"tentative": "0.55", "confirmed": "#1a9e3a", "coasting": "#f08c00"}


class Snapshot(Node):

    def __init__(self) -> None:
        super().__init__("tracking_snapshot")
        p = self.declare_parameter
        anyval = ParameterDescriptor(dynamic_typing=True)    # 3 or 3.0 both work
        p("host", "localhost")
        p("port", 2000)
        p("role_name", "ego")
        p("out", "tracking_snapshot.png")
        p("after_frames", 100)
        p("half_extent", 35.0, anyval)
        p("measurement_sigma", 0.5, anyval)
        p("min_ego_speed", 0.0, anyval)        # m/s: wait until the AV moves at least this fast
        g = lambda name: self.get_parameter(name).value
        self.out, self.after = g("out"), int(g("after_frames"))
        self.half = float(g("half_extent"))
        self.R = float(g("measurement_sigma")) ** 2 * np.eye(2)
        self.min_speed = float(g("min_ego_speed"))
        self.truth = TruthWatcher(g("host"), int(g("port")), g("role_name"))
        self.extents: dict[int, tuple] = {}
        self.dets: dict[int, np.ndarray] = {}
        self.frames = 0
        self.done = False
        self.create_subscription(Detection3DArray, "/l5/detections", self._on_det, 10)
        self.create_subscription(String, "/l5/tracks/json", self._on_tracks, 10)

    def _on_det(self, msg: Detection3DArray) -> None:
        self.dets[time_key(stamp_to_sec(msg.header.stamp))] = np.array(
            [[d.bbox.center.position.x, d.bbox.center.position.y] for d in msg.detections]
        ).reshape(-1, 2)

    def _extent(self, oid: int) -> tuple:
        if oid not in self.extents:
            actor = self.truth.world.get_actor(oid)
            e = actor.bounding_box.extent if actor is not None else None
            self.extents[oid] = (e.x, e.y) if e is not None else (2.25, 0.95)
        return self.extents[oid]

    def _on_tracks(self, msg: String) -> None:
        if self.done:
            return
        self.frames += 1
        frame = json.loads(msg.data)
        truth = self.truth.at(frame["stamp"])
        dets = self.dets.get(time_key(frame["stamp"]))
        if self.frames < self.after or truth is None or dets is None:
            return
        ego = next((o for o in truth if o[0] == self.truth.ego_id), None)
        if ego is None or math.hypot(ego[3], ego[4]) < self.min_speed:
            return
        self._draw(frame, truth, dets, ego)
        self.done = True

    def _draw(self, frame: dict, truth: list, dets: np.ndarray, ego: tuple) -> None:
        c, s = math.cos(-ego[5]), math.sin(-ego[5])
        Rot = np.array([[c, -s], [s, c]])          # map -> AV frame rotation

        def to_ego(x, y):
            return Rot @ np.array([x - ego[1], y - ego[2]])

        fig, ax = plt.subplots(figsize=(10.5, 6.2))
        ax.set_aspect("equal")
        ax.set_xlim(-self.half * 0.6, self.half)
        ax.set_ylim(-self.half * 0.6, self.half * 0.6)
        ax.grid(color="0.92")
        # Ground truth: every CARLA vehicle, as its box.
        for oid, x, y, _vx, _vy, yaw, kind in truth:
            if kind != "vehicle":
                continue
            ex, ey = self._extent(oid)
            cy, sy = math.cos(yaw), math.sin(yaw)
            corners = [(x + cy * a - sy * b, y + sy * a + cy * b)
                       for a, b in ((ex, ey), (-ex, ey), (-ex, -ey), (ex, -ey))]
            pts = np.array([to_ego(*p) for p in corners])
            if oid == ego[0]:
                ax.add_patch(Polygon(pts, closed=True, fc="black", ec="black"))
                ax.annotate("AV", (0, 0), xytext=(0, -2.6), ha="center", fontsize=9)
            else:
                ax.add_patch(Polygon(pts, closed=True, fc="none", ec="0.45", lw=1.0))
        # This frame's detections.
        if len(dets):
            d = np.array([to_ego(*p) for p in dets])
            ax.plot(d[:, 0], d[:, 1], "x", color="#1f5fbf", ms=6, mew=1.5)
        # Tracks: gate, velocity, ID.
        for trk in frame["tracks"]:
            p = to_ego(trk["x"], trk["y"])
            if abs(p[0]) > self.half or abs(p[1]) > self.half:
                continue
            col = COLORS[trk["status"]]
            S = Rot @ (np.array(trk["P"]) + self.R) @ Rot.T
            vals, vecs = np.linalg.eigh(S)
            ang = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
            w, h = 2 * np.sqrt(9.21 * vals[::-1])
            ax.add_patch(Ellipse(p, w, h, angle=ang, fill=False, ls="--", lw=0.9, ec=col))
            ax.plot(*p, "o", color=col, ms=6)
            if trk["status"] != "tentative":
                v = Rot @ np.array([trk["vx"], trk["vy"]])
                ax.annotate("", xy=p + v, xytext=p,
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
            ax.annotate(str(trk["id"]), p, xytext=(4, 4), textcoords="offset points",
                        fontsize=8, color=col)
        counts = frame["counts"]
        speed = math.hypot(ego[3], ego[4])
        ax.set_title(f"AV at {speed:.1f} m/s. {frame['detections']} detections; tracks "
                     f"{counts['tentative']} tentative, {counts['confirmed']} confirmed, "
                     f"{counts['coasting']} coasting", fontsize=10)
        ax.set_xlabel("x, forward (m)")
        ax.set_ylabel("y, left (m)")
        handles = [
            plt.Line2D([], [], color="0.45", lw=1, label="CARLA vehicle (truth)"),
            plt.Line2D([], [], marker="x", ls="", color="#1f5fbf", label="detection"),
            plt.Line2D([], [], marker="o", ls="", color=COLORS["tentative"], label="tentative track"),
            plt.Line2D([], [], marker="o", ls="", color=COLORS["confirmed"], label="confirmed track"),
            plt.Line2D([], [], marker="o", ls="", color=COLORS["coasting"], label="coasting track"),
            plt.Line2D([], [], ls="--", color="0.3", label="gate, epsilon = 9.21"),
        ]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
        fig.tight_layout()
        fig.savefig(self.out, dpi=150)
        self.get_logger().info(f"wrote {self.out}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Snapshot()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if node.context.ok():
            raise
    finally:
        node.truth.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
