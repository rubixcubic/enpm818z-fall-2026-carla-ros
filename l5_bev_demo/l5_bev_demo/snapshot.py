#!/usr/bin/env python3
"""Save the four bird's-eye views of one moment as PNG files, side by side.

    ros2 run l5_bev_demo snapshot                      # into ./bev_snapshot
    ros2 run l5_bev_demo snapshot --ros-args -p out:=/tmp/bev -p wait:=3.0

Waits until all four views have arrived, keeps the newest of each for `wait`
more seconds, writes lidar_height.png, occupancy.png, ipm.png, semantic.png and
all_four.png (with a label on each), then exits. Use it for your report: one
picture with the same moment seen four ways.
"""

import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import Image

VIEWS = (("lidar_height", "LiDAR height"), ("occupancy", "LiDAR occupancy"),
         ("ipm", "camera IPM"), ("semantic", "semantic lift-splat"))


class Snapshot(Node):

    def __init__(self) -> None:
        super().__init__("bev_snapshot")
        self.declare_parameter("out", "bev_snapshot")
        self.declare_parameter("wait", 2.0)
        self.out = self.get_parameter("out").value
        self.wait = float(self.get_parameter("wait").value)
        self.images = {}
        self.first_complete = None
        for name in ("lidar_height", "ipm", "semantic"):
            self.create_subscription(Image, f"/l5/bev/{name}",
                                     lambda m, n=name: self._on_image(n, m), 2)
        self.create_subscription(OccupancyGrid, "/l5/bev/occupancy", self._on_occ, 2)

    def _on_image(self, name: str, msg: Image) -> None:
        self.images[name] = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3).copy()

    def _on_occ(self, msg: OccupancyGrid) -> None:
        n = msg.info.width
        grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, n)
        # Back to the display layout of the other views: front up, left on the left.
        disp = grid.T[::-1, ::-1]
        img = np.full(disp.shape + (3,), 128, dtype=np.uint8)   # unknown: gray
        img[disp == 0] = 255                                     # free: white
        img[disp == 100] = 0                                     # occupied: black
        self.images["occupancy"] = img

    def done(self) -> bool:
        if len(self.images) < len(VIEWS):
            return False
        if self.first_complete is None:
            self.first_complete = time.monotonic()
        return time.monotonic() - self.first_complete > self.wait

    def save(self) -> None:
        os.makedirs(self.out, exist_ok=True)
        tiles = []
        for name, label in VIEWS:
            img = self.images[name]
            cv2.imwrite(os.path.join(self.out, f"{name}.png"), img)
            tile = cv2.copyMakeBorder(img, 28, 4, 4, 4, cv2.BORDER_CONSTANT,
                                      value=(255, 255, 255))
            cv2.putText(tile, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(tile)
        cv2.imwrite(os.path.join(self.out, "all_four.png"),
                    np.hstack(tiles))
        self.get_logger().info(f"saved {len(VIEWS) + 1} images to {os.path.abspath(self.out)}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Snapshot()
    try:
        while rclpy.ok() and not node.done():
            rclpy.spin_once(node, timeout_sec=0.1)
        node.save()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass        # Ctrl+C: Jazzy raises the second one from spin()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
