#!/usr/bin/env python3
"""Grade the network against CARLA's true classes: IoU per class and mIoU.

    in   /l5/seg/labels      the network's classes (seg_node), CARLA tags
         /l5/seg/truth_tags  CARLA's true classes (seg_truth), same camera

Two images are compared only when they come from the same simulation tick:
the same time stamp. The network skips ticks while it is busy, so not every
truth image is graded. Both images are mapped to the 19 Cityscapes classes
(classes.py): CARLA's lane markings count as road, and CARLA classes that
Cityscapes does not grade are left out.

Every report_every seconds the node logs the IoU of each class seen so far,
the mIoU and the pixel accuracy, summed over all graded frames, and what the
network called CARLA's lane-marking pixels. With out:=seg.json the report is
also saved.

The topics are parameters, so the same node grades l5_bev_demo's front roof
camera too:
    ros2 run l5_seg_demo seg_eval --ros-args \\
        -p truth_topic:=/l5/front/semantic -p labels_topic:=/l5/front/semantic_net \\
        -p reliable:=false
"""

from __future__ import annotations

import json

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image

from l5_seg_demo.classes import ROADLINE, tag_name, tags_to_graded
from l5_seg_demo.ros_util import spin, stamp_key
from l5_seg_demo.seg_eval import ConfusionMatrix

KEEP = 60        # images kept per side while waiting for their pair (3 s at 20 Hz)


def mono8(msg: Image) -> np.ndarray:
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    return arr[:, :msg.width]


class SegEval(Node):

    def __init__(self) -> None:
        super().__init__("seg_eval")
        self.declare_parameter("truth_topic", "/l5/seg/truth_tags")
        self.declare_parameter("labels_topic", "/l5/seg/labels")
        self.declare_parameter("report_every", 10.0)
        self.declare_parameter("out", "")
        # Must match the publishers: seg_node and seg_truth publish reliably;
        # l5_bev_demo's roof cameras (and seg_node's rig output) publish best effort.
        self.declare_parameter("reliable", True)
        self.out = str(self.get_parameter("out").value)
        self.cm = ConfusionMatrix()
        self.truth: dict[int, np.ndarray] = {}
        self.labels: dict[int, np.ndarray] = {}
        self.roadline = np.zeros(256, dtype=np.int64)  # network's answer on lane markings
        self.received = {"truth": 0, "labels": 0}

        qos = QoSProfile(depth=10, reliability=(
            QoSReliabilityPolicy.RELIABLE if self.get_parameter("reliable").value
            else QoSReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(Image, self.get_parameter("truth_topic").value,
                                 lambda m: self._store("truth", m), qos)
        self.create_subscription(Image, self.get_parameter("labels_topic").value,
                                 lambda m: self._store("labels", m), qos)
        self.create_timer(float(self.get_parameter("report_every").value), self._report)

    def _store(self, side: str, msg: Image) -> None:
        self.received[side] += 1
        key = stamp_key(msg.header.stamp)
        mine = self.truth if side == "truth" else self.labels
        other = self.labels if side == "truth" else self.truth
        img = mono8(msg).copy()
        if key in other:
            pair = other.pop(key)
            truth, labels = (img, pair) if side == "truth" else (pair, img)
            self._grade(truth, labels)
        else:
            mine[key] = img
            for old in sorted(mine)[:-KEEP]:       # never paired: the network skipped it
                del mine[old]

    def _grade(self, truth_tags: np.ndarray, net_tags: np.ndarray) -> None:
        if truth_tags.shape != net_tags.shape:
            self.get_logger().warn(f"sizes differ: truth {truth_tags.shape}, "
                                   f"network {net_tags.shape}", throttle_duration_sec=5.0)
            return
        self.cm.add(tags_to_graded(truth_tags), tags_to_graded(net_tags))
        self.roadline += np.bincount(net_tags[truth_tags == ROADLINE], minlength=256)

    def report(self) -> dict:
        rep = self.cm.report()
        rep["images_received"] = dict(self.received)
        total = int(self.roadline.sum())
        if total:
            top = np.argsort(-self.roadline)[:3]
            rep["lane_marking_pixels"] = {
                "count": total,
                "network_called_them": {tag_name(t): round(float(self.roadline[t] / total), 4)
                                        for t in top if self.roadline[t] > 0}}
        return rep

    def _report(self) -> None:
        if self.cm.frames == 0:
            self.get_logger().info(
                f"no pair yet (received {self.received['truth']} truth, "
                f"{self.received['labels']} network images; are seg_node and seg_truth "
                "running?)")
            return
        rep = self.report()
        lines = [f"{rep['frames']} frames graded "
                 f"(of {self.received['truth']} truth images): "
                 f"mIoU {rep['miou']:.3f}, pixel accuracy {rep['pixel_accuracy']:.3f}",
                 f"  {'class':14s} {'IoU':>6s} {'truth share':>12s}   the network called it"]
        for name, c in sorted(rep["classes"].items(), key=lambda kv: -kv[1]["truth_share"]):
            called = ", ".join(f"{n} {100 * v:.0f}%" for n, v in c["network_called_it"].items())
            share = 100 * c["truth_share"]
            lines.append(f"  {name:14s} {c['iou']:6.3f} {share:11.2f}%   {called}")
        if "lane_marking_pixels" in rep:
            called = ", ".join(f"{n} {100 * v:.1f}%" for n, v in
                               rep["lane_marking_pixels"]["network_called_them"].items())
            lines.append(f"  CARLA's lane-marking pixels, the network called them: {called}")
        self.get_logger().info("\n".join(lines))
        if self.out:
            with open(self.out, "w") as f:
                json.dump(rep, f, indent=1)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SegEval()
    try:
        spin(node)
    finally:
        if node.cm.frames and node.out:
            with open(node.out, "w") as f:
                json.dump(node.report(), f, indent=1)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
