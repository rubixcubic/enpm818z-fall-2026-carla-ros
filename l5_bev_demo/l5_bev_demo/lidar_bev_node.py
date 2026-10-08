#!/usr/bin/env python3
"""Two bird's-eye views from one LiDAR sweep: a height map and an occupancy grid.

Slides: 'PointPillars: from points to 3D boxes' (Steps 1 to 3, without the
learned part) and 'BEV and occupancy in industry' (Autoware's occupancy grid).

Height map. Drop every point into the BEV cell under it and keep the highest
one. That is PointPillars' first step, a grid of pillars, with the learned
feature replaced by one number anyone can read: how tall the stuff in this
cell is.

Occupancy grid. Each cell is free, occupied or unknown, decided ray by ray as
in Autoware's pointcloud-based occupancy grid:
  Step 1. Along each ray, every cell up to the farthest return is free: the
          laser went through it.
  Step 2. Behind an obstacle, cells are unknown: the laser never got there.
  Step 3. Cells that hold an obstacle point are occupied.
Autoware then smooths each cell over time with a Bayes filter; this node shows
one sweep at a time, so you can see a single frame's blind spots.

Subscribes to the L2 bridge's LiDAR. Publishes, in the ego_vehicle frame:
    /l5/bev/lidar_height   sensor_msgs/Image     (bgr8, front up)
    /l5/bev/occupancy      nav_msgs/OccupancyGrid (0 free, 100 occupied, -1 unknown)
"""

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2
from tf2_ros import Buffer, TransformListener

from l5_bev_demo.bev_grid import BevGrid, image_msg, matrix_from_transform

EGO_FRAME = "ego_vehicle"
FREE, OCCUPIED, UNKNOWN = 0, 100, -1


class LidarBev(Node):

    def __init__(self) -> None:
        super().__init__("lidar_bev")
        self.declare_parameter("half_extent", 25.0)
        self.declare_parameter("resolution", 0.2)
        self.declare_parameter("ground_max_z", 0.3)
        self.declare_parameter("obstacle_max_z", 2.5)
        self.declare_parameter("azimuth_bins", 360)
        self.declare_parameter("lidar_topic", "/carla/ego_vehicle/lidar")
        self.declare_parameter("ego_length", 4.8)
        self.declare_parameter("ego_width", 2.2)

        self.grid = BevGrid(self.get_parameter("half_extent").value,
                            self.get_parameter("resolution").value)
        self.ground_max_z = float(self.get_parameter("ground_max_z").value)
        self.obstacle_max_z = float(self.get_parameter("obstacle_max_z").value)
        self.bins = int(self.get_parameter("azimuth_bins").value)
        self.ego_half = (float(self.get_parameter("ego_length").value) / 2.0 + 0.2,
                         float(self.get_parameter("ego_width").value) / 2.0 + 0.2)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.ego_from_lidar = None      # 4 x 4, looked up once: the mount is rigid
        self.polar = None               # per cell: (range, azimuth bin) from the LiDAR

        qos = QoSProfile(depth=2, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(PointCloud2, self.get_parameter("lidar_topic").value,
                                 self._on_cloud, qos)
        self.pub_height = self.create_publisher(Image, "/l5/bev/lidar_height", 2)
        self.pub_occ = self.create_publisher(OccupancyGrid, "/l5/bev/occupancy", 2)
        self.frames = 0

    def _lookup_mount(self, frame: str) -> bool:
        try:
            tf = self.tf_buffer.lookup_transform(EGO_FRAME, frame, Time())
        except Exception:                                   # not published yet
            self.get_logger().info(f"waiting for tf {EGO_FRAME} <- {frame}",
                                   throttle_duration_sec=5.0)
            return False
        self.ego_from_lidar = matrix_from_transform(tf.transform)
        # Range and azimuth of every cell center, seen from the LiDAR's spot on
        # the ground. Computed once: the grid and the mount never move.
        lx, ly = self.ego_from_lidar[0, 3], self.ego_from_lidar[1, 3]
        dx, dy = self.grid.cell_x - lx, self.grid.cell_y - ly
        cell_range = np.hypot(dx, dy)
        cell_bin = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * self.bins).astype(int)
        self.polar = (cell_range, np.clip(cell_bin, 0, self.bins - 1), lx, ly)
        return True

    def _on_cloud(self, msg: PointCloud2) -> None:
        if not self.context.ok():           # mid-shutdown (Ctrl+C)
            return
        if self.ego_from_lidar is None and not self._lookup_mount(msg.header.frame_id):
            return
        pts = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, msg.point_step // 4)
        xyz1 = np.c_[pts[:, :3], np.ones(len(pts))]
        ego = (self.ego_from_lidar @ xyz1.T).T          # points in the ego frame
        # The lowest beams hit the AV's own roof and hood. Left in, they would
        # be an "obstacle" 1 m away in every direction, and the whole grid
        # behind them would turn unknown. Drop everything inside the footprint.
        own = (np.abs(ego[:, 0]) < self.ego_half[0]) & (np.abs(ego[:, 1]) < self.ego_half[1])
        ego = ego[~own]
        x, y, z = ego[:, 0], ego[:, 1], ego[:, 2]

        self._publish_height(msg, x, y, z)
        self._publish_occupancy(msg, x, y, z)

        self.frames += 1
        if self.frames % 50 == 1:
            self.get_logger().info(f"sweep of {len(pts)} points -> grids")

    # ---------------------------------------------------------- height map
    def _publish_height(self, msg, x, y, z) -> None:
        g = self.grid
        i, j, inside = g.index(x, y)
        keep = inside & (z < self.obstacle_max_z)
        r, c = g.to_image_rc(i[keep], j[keep])
        height = np.full((g.n, g.n), -np.inf)
        np.maximum.at(height, (r, c), z[keep])          # tallest point per cell

        hit = np.isfinite(height)
        scaled = np.zeros((g.n, g.n), dtype=np.uint8)
        # 0 m to 2.5 m mapped onto the color scale; empty cells stay black.
        scaled[hit] = np.clip(height[hit] / self.obstacle_max_z * 255, 1, 255)
        bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        bgr[~hit] = 0
        g.draw_ego(bgr)
        header = msg.header
        header.frame_id = EGO_FRAME
        self.pub_height.publish(image_msg(bgr, header))

    # ------------------------------------------------------ occupancy grid
    def _publish_occupancy(self, msg, x, y, z) -> None:
        g = self.grid
        cell_range, cell_bin, lx, ly = self.polar
        dx, dy = x - lx, y - ly
        pt_range = np.hypot(dx, dy)
        pt_bin = np.clip(((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * self.bins)
                         .astype(int), 0, self.bins - 1)
        obstacle = (z > self.ground_max_z) & (z < self.obstacle_max_z)

        # Per ray (azimuth bin): how far the laser reached, and where the
        # nearest obstacle stopped it.
        reach = np.zeros(self.bins)
        np.maximum.at(reach, pt_bin, pt_range)
        nearest = np.full(self.bins, np.inf)
        np.minimum.at(nearest, pt_bin[obstacle], pt_range[obstacle])

        ray_reach = reach[cell_bin]
        ray_obstacle = nearest[cell_bin]
        state = np.full((g.n, g.n), UNKNOWN, dtype=np.int8)
        # Step 1: free up to the farthest return, but not past an obstacle.
        state[cell_range < np.minimum(ray_reach, ray_obstacle)] = FREE
        # Step 2: everything past an obstacle stays unknown.
        # Step 3: cells holding an obstacle point are occupied.
        i, j, inside = g.index(x[obstacle], y[obstacle])
        r, c = g.to_image_rc(i[inside], j[inside])
        state[r, c] = OCCUPIED

        # OccupancyGrid wants rows along y and columns along x, starting at the
        # grid's corner (-half_extent, -half_extent): undo the display flip.
        data = state[::-1, ::-1].T
        occ = OccupancyGrid()
        occ.header = msg.header
        occ.header.frame_id = EGO_FRAME
        occ.info.resolution = g.resolution
        occ.info.width = g.n
        occ.info.height = g.n
        occ.info.origin.position.x = -g.half_extent
        occ.info.origin.position.y = -g.half_extent
        occ.info.origin.orientation.w = 1.0
        occ.data = data.astype(np.int8).ravel().tolist()
        self.pub_occ.publish(occ)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarBev()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass        # Ctrl+C: Jazzy raises the second one from spin()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
