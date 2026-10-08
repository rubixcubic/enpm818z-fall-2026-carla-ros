#!/usr/bin/env python3
"""A surround view from four cameras by inverse perspective mapping (IPM).

Slides: 'Before learning: inverse perspective mapping' and 'IPM breaks for
anything above the road'.

IPM assumes every pixel shows flat ground (z = 0 in the ego frame). Under that
one assumption, each BEV cell has exactly one place in each camera image, so
the whole mapping can be computed once and reused for every frame:

  Step 1. Take the cell's center on the ground, (x, y, 0), in the ego frame.
  Step 2. Move it into each camera's optical frame with the camera's mounting
          transform (from tf; the rig publishes it).
  Step 3. Project it with the pinhole model (L2): u = f x_c / z_c + c_u,
          v = f y_c / z_c + c_v. Keep it if the point is in front of the camera
          and (u, v) falls inside the image.
  Step 4. Where two cameras see the same cell, take the one that looks at it
          most directly (smallest angle from its optical axis).

Each frame is then one cv2.remap per camera. Lane paint lands where it is.
Anything that stands up, a car or a pole, is not at z = 0, so IPM draws it as a
long streak pointing away from the camera: exactly the door that IPM put at
20 m instead of 10 m on the slide.

Publishes /l5/bev/ipm (sensor_msgs/Image, bgr8, front up), same grid as the
other two views.
"""

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener

from l5_bev_demo.bev_grid import BevGrid, image_msg, matrix_from_transform

EGO_FRAME = "ego_vehicle"
CAMERAS = ("front", "left", "right", "rear")


class IpmBev(Node):

    def __init__(self) -> None:
        super().__init__("ipm_bev")
        self.declare_parameter("half_extent", 25.0)
        self.declare_parameter("resolution", 0.2)
        self.grid = BevGrid(self.get_parameter("half_extent").value,
                            self.get_parameter("resolution").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.info = {}       # camera name -> CameraInfo
        self.maps = {}       # camera name -> (map_u, map_v, mask): Steps 1 to 4
        self.latest = {}     # camera name -> newest BGR image
        self.stamp = None

        sensor_qos = QoSProfile(depth=2, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        for name in CAMERAS:
            self.create_subscription(
                CameraInfo, f"/l5/{name}/camera_info",
                lambda m, n=name: self.info.__setitem__(n, m), latched)
            self.create_subscription(
                Image, f"/l5/{name}/rgb", lambda m, n=name: self._on_image(n, m),
                sensor_qos)
        self.pub = self.create_publisher(Image, "/l5/bev/ipm", 2)
        self.create_timer(0.1, self._compose)            # 10 Hz, as the cameras

    def _on_image(self, name: str, msg: Image) -> None:
        bgra = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        self.latest[name] = bgra[:, :, :3]
        self.stamp = msg.header.stamp

    def _build_maps(self) -> bool:
        """Steps 1 to 4, once, as soon as every camera's intrinsics and tf exist."""
        g = self.grid
        ground = np.stack([g.cell_x.ravel(), g.cell_y.ravel(),
                           np.zeros(g.n * g.n), np.ones(g.n * g.n)])   # Step 1
        best = np.full(g.n * g.n, -np.inf)
        owner = np.full(g.n * g.n, -1)
        uv = {}
        for k, name in enumerate(CAMERAS):
            if name not in self.info:
                return False
            try:
                tf = self.tf_buffer.lookup_transform(
                    f"l5/{name}_optical", EGO_FRAME, Time())
            except Exception:
                return False
            cam = matrix_from_transform(tf.transform) @ ground          # Step 2
            xc, yc, zc = cam[0], cam[1], cam[2]
            info = self.info[name]
            fx, cx, fy, cy = info.k[0], info.k[2], info.k[4], info.k[5]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = fx * xc / zc + cx                                   # Step 3
                v = fy * yc / zc + cy
            visible = (zc > 0.5) & (u >= 0) & (u < info.width - 1) & \
                      (v >= 0) & (v < info.height - 1)
            # Step 4: cosine of the angle from the optical axis.
            directness = np.where(visible, zc / np.sqrt(xc**2 + yc**2 + zc**2), -np.inf)
            better = directness > best
            best[better] = directness[better]
            owner[better] = k
            uv[name] = (u, v)
        for k, name in enumerate(CAMERAS):
            u, v = uv[name]
            mask = (owner == k).reshape(g.n, g.n)
            self.maps[name] = (np.where(owner == k, u, -1).reshape(g.n, g.n).astype(np.float32),
                               np.where(owner == k, v, -1).reshape(g.n, g.n).astype(np.float32),
                               mask)
        covered = 100.0 * np.mean(owner >= 0)
        self.get_logger().info(f"IPM maps ready: {covered:.0f}% of the grid is seen "
                               "by at least one camera")
        return True

    def _compose(self) -> None:
        if not self.maps and not self._build_maps():
            self.get_logger().info("waiting for camera_info and tf from surround_rig",
                                   throttle_duration_sec=5.0)
            return
        if len(self.latest) < len(CAMERAS):
            return
        g = self.grid
        out = np.zeros((g.n, g.n, 3), dtype=np.uint8)
        for name, (map_u, map_v, mask) in self.maps.items():
            warped = cv2.remap(self.latest[name], map_u, map_v, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            out[mask] = warped[mask]
        g.draw_ego(out)
        header = self._header()
        if self.context.ok():           # not mid-shutdown (Ctrl+C)
            self.pub.publish(image_msg(out, header))

    def _header(self) -> Header:
        h = Header()
        h.frame_id = EGO_FRAME
        if self.stamp is not None:
            h.stamp = self.stamp
        return h


def main(args=None) -> None:
    rclpy.init(args=args)
    node = IpmBev()
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
