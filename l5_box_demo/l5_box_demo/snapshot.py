#!/usr/bin/env python3
"""Save one sweep's boxes as a top view PNG, in the AV's frame (x forward, to the right).

The sweep's data goes next to it as <stem>_sweep.json (boxes.png gives
boxes_sweep.json), so it never overwrites box_detector's report, which the
README saves as out:=boxes.json.

Listens to /l5/boxes/debug (published by box_detector every debug_every
sweeps), keeps the skip-th message, draws it and exits:
    gray dots      the LiDAR points the clusters were made of (ground removed)
    blue boxes     the fitted boxes, with their heading axis (mod 180 deg)
    red crosses    each cluster's centroid, the bare-centroid detector's answer
    green dashes   CARLA's true vehicle boxes, when box_detector runs with evaluate:=true

    ros2 run l5_box_demo snapshot --ros-args -p out:=boxes.png -p skip:=3
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from std_msgs.msg import String  # noqa: E402

from l5_tracking_demo.ros_util import spin  # noqa: E402


class Snapshot(Node):

    def __init__(self) -> None:
        super().__init__("box_snapshot")
        self.declare_parameter("out", "boxes.png")
        self.declare_parameter("skip", 3)
        self.declare_parameter("half_width", 30.0)
        self.out = self.get_parameter("out").value
        self.skip = int(self.get_parameter("skip").value)
        self.half = float(self.get_parameter("half_width").value)
        self.seen = 0
        self.done = False
        self.create_subscription(String, "/l5/boxes/debug", self._on_debug, 10)

    def _on_debug(self, msg: String) -> None:
        if self.done:
            return
        self.seen += 1
        if self.seen <= self.skip:
            return
        data = self.out.rsplit(".", 1)[0] + "_sweep.json"    # to redraw later
        with open(data, "w") as f:
            f.write(msg.data)
        draw(json.loads(msg.data), self.out, self.half)
        self.get_logger().info(f"wrote {self.out} and {data}")
        self.done = True
        raise SystemExit


def draw(d: dict, out: str, half: float) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    pts = np.array(d["points"]).reshape(-1, 2)
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], s=1.5, color="0.55", zorder=1, label="LiDAR points")
    for k, o in enumerate(d.get("truth", [])):
        c = np.array(o["corners"] + [o["corners"][0]])
        ax.plot(c[:, 0], c[:, 1], "--", color="#2e7d32", lw=1.2, zorder=2,
                label="CARLA's true box" if k == 0 else None)
    for k, b in enumerate(d["boxes"]):
        c = np.array(b["corners"] + [b["corners"][0]])
        ax.fill(c[:, 0], c[:, 1], color="#85B7E2", alpha=0.6, zorder=3,
                label="fitted box" if k == 0 else None)
        ax.plot(c[:, 0], c[:, 1], color="#0b2e8a", lw=1.2, zorder=4)
        h = b["length"] / 2.0 + 0.5
        dx, dy = h * np.cos(b["yaw"]), h * np.sin(b["yaw"])
        ax.plot([b["cx"] - dx, b["cx"] + dx], [b["cy"] - dy, b["cy"] + dy], color="#0b2e8a",
                lw=0.8, zorder=4, label="heading axis (mod 180 deg)" if k == 0 else None)
        ax.plot(*b["centroid"], "x", color="#c62828", ms=6, mew=1.6, zorder=5,
                label="cluster centroid" if k == 0 else None)
    ax.plot(0, 0, marker=(3, 0, -90), ms=12, color="black", zorder=6, linestyle="none",
            label="the AV (ego frame origin), facing +x")
    ax.set_xlim(-half, half)
    ax.set_ylim(-half * 0.7, half * 0.7)
    ax.set_aspect("equal")
    ax.set_xlabel("x, forward (m)")
    ax.set_ylabel("y, left (m)")
    ax.grid(color="0.92")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, frameon=False)
    ax.set_title(f"l5_box_demo: one LiDAR sweep, t = {d['t']:.2f} s (simulation time)")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Snapshot()
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
