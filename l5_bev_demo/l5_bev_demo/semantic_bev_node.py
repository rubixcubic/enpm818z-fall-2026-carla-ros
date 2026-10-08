#!/usr/bin/env python3
"""A semantic bird's-eye view by lift and splat, with CARLA's true depth.

Slides: 'Lift-Splat-Shoot', 'Lift, by hand' and 'Splat, by hand'.

Lift-Splat-Shoot does not know how far each pixel is, so its network spreads
each pixel over many depths with a probability for each. CARLA's depth camera
knows. So here every pixel goes to exactly one depth: the probability of its
true depth bin is 1 and every other bin gets 0. What is left is the geometry of
lift and splat, with nothing learned:

  Lift.   For each pixel (u, v) with depth d, its point in the camera's optical
          frame is  x_c = (u - c_u) d / f,  y_c = (v - c_v) d / f,  z_c = d.
          CARLA's depth is measured along the optical axis (a z-buffer), which
          is why z_c = d. Move the point into the ego frame with the camera's
          mounting transform.
  Splat.  The point falls in one BEV cell. The "feature" it carries is its
          class from the semantic camera. Where several points land in one
          cell, keep the class that matters most to a planner: people and
          vehicles over road markings, road markings over the road.

Publishes /l5/bev/semantic (sensor_msgs/Image, bgr8, front up), in CARLA's own
class colors, on the same grid as the other two views.

labels:=truth (default) splats CARLA's true classes (/l5/<camera>/semantic).
labels:=network splats a trained network's classes instead
(/l5/<camera>/semantic_net, from l5_seg_demo's seg_node with rig:=true), with
CARLA's true depth still. The network has no lane-marking class, so lane
lines do not appear in that view.
"""

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

# CARLA 0.9.16 semantic tags and their colors (RGB), from the sensor reference.
PALETTE = {
    0: (0, 0, 0), 1: (128, 64, 128), 2: (244, 35, 232), 3: (70, 70, 70),
    4: (102, 102, 156), 5: (190, 153, 153), 6: (153, 153, 153), 7: (250, 170, 30),
    8: (220, 220, 0), 9: (107, 142, 35), 10: (152, 251, 152), 11: (70, 130, 180),
    12: (220, 20, 60), 13: (255, 0, 0), 14: (0, 0, 142), 15: (0, 0, 70),
    16: (0, 60, 100), 17: (0, 80, 100), 18: (0, 0, 230), 19: (119, 11, 32),
    20: (110, 190, 160), 21: (170, 120, 50), 22: (55, 90, 80), 23: (45, 60, 150),
    24: (157, 234, 50), 25: (81, 0, 81), 26: (150, 100, 100), 27: (230, 150, 140),
    28: (180, 165, 180),
}
SKY = 11
# Which class wins a cell. Higher wins. Our choice, made for a planner: what it
# must not hit, then where it may not drive, then the road itself.
PRIORITY = np.ones(29, dtype=np.int16)                  # everything else: 1
PRIORITY[[1, 25, 10]] = 2                               # road, ground, terrain
PRIORITY[2] = 3                                         # sidewalk
PRIORITY[24] = 4                                        # lane markings
PRIORITY[[3, 4, 5, 6, 7, 8, 9, 20, 21, 22, 26, 28]] = 5  # buildings, poles, fences...
PRIORITY[[14, 15, 16, 17, 18, 19]] = 6                  # vehicles
PRIORITY[[12, 13]] = 7                                  # pedestrians and riders
PRIORITY[[0, SKY]] = 0                                  # never splat these


class SemanticBev(Node):

    def __init__(self) -> None:
        super().__init__("semantic_bev")
        self.declare_parameter("half_extent", 25.0)
        self.declare_parameter("resolution", 0.2)
        self.declare_parameter("max_depth", 40.0)
        self.declare_parameter("max_height", 3.0)
        self.declare_parameter("stride", 2)
        self.declare_parameter("labels", "truth")
        self.grid = BevGrid(self.get_parameter("half_extent").value,
                            self.get_parameter("resolution").value)
        self.max_depth = float(self.get_parameter("max_depth").value)
        self.max_height = float(self.get_parameter("max_height").value)
        self.stride = int(self.get_parameter("stride").value)
        labels = self.get_parameter("labels").value
        if labels not in ("truth", "network"):
            raise SystemExit(f"labels must be 'truth' or 'network', not '{labels}'")
        label_topic = "semantic" if labels == "truth" else "semantic_net"

        self.colors = np.zeros((256, 3), dtype=np.uint8)
        for tag, (r, g, b) in PALETTE.items():
            self.colors[tag] = (b, g, r)                 # stored as BGR

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.info = {}
        self.rays = {}        # camera -> rays (x_c/d, y_c/d) of the sampled pixels
        self.ego_from_cam = {}
        self.semantic = {}    # camera -> (stamp, tag image)
        self.depth = {}       # camera -> (stamp, depth image, meters)

        sensor_qos = QoSProfile(depth=2, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        for name in CAMERAS:
            self.create_subscription(
                CameraInfo, f"/l5/{name}/camera_info",
                lambda m, n=name: self.info.__setitem__(n, m), latched)
            self.create_subscription(
                Image, f"/l5/{name}/{label_topic}",
                lambda m, n=name: self._store(self.semantic, n, m, np.uint8), sensor_qos)
            self.create_subscription(
                Image, f"/l5/{name}/depth",
                lambda m, n=name: self._store(self.depth, n, m, np.float32), sensor_qos)
        self.pub = self.create_publisher(Image, "/l5/bev/semantic", 2)
        self.create_timer(0.1, self._splat)

    @staticmethod
    def _store(table: dict, name: str, msg: Image, dtype) -> None:
        """Keep the last few frames per camera, keyed by their time stamp.

        The semantic and the depth image of one tick arrive separately and not
        always together; pairing by stamp from a short history keeps every
        camera in the grid instead of only the ones that happened to line up.
        """
        img = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width)
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        frames = table.setdefault(name, {})
        frames[stamp] = (img, msg.header.stamp)
        for old in sorted(frames)[:-4]:
            del frames[old]

    def _ready(self, name: str) -> bool:
        """Intrinsics and mount, looked up once per camera."""
        if name in self.ego_from_cam:
            return True
        if name not in self.info:
            return False
        try:
            tf = self.tf_buffer.lookup_transform(EGO_FRAME, f"l5/{name}_optical", Time())
        except Exception:
            return False
        self.ego_from_cam[name] = matrix_from_transform(tf.transform)
        info = self.info[name]
        fx, cx, fy, cy = info.k[0], info.k[2], info.k[4], info.k[5]
        s = self.stride
        v, u = np.mgrid[0:info.height:s, 0:info.width:s]
        # The ray of each sampled pixel, per meter of depth: (x_c / d, y_c / d).
        self.rays[name] = ((u - cx) / fx, (v - cy) / fy)
        return True

    def _splat(self) -> None:
        g = self.grid
        best = np.zeros(g.n * g.n, dtype=np.int16)       # priority per cell
        tag_of = np.zeros(g.n * g.n, dtype=np.uint8)     # winning class per cell
        stamp = None
        for name in CAMERAS:
            if not self._ready(name) or name not in self.semantic or name not in self.depth:
                continue
            common = set(self.semantic[name]) & set(self.depth[name])
            if not common:                  # the pair must come from the same tick
                continue
            key = max(common)
            tags, ros_stamp = self.semantic[name][key]
            depth, _ = self.depth[name][key]
            stamp = ros_stamp
            s = self.stride
            tags = tags[::s, ::s]
            d = depth[::s, ::s].astype(np.float64)
            ray_x, ray_y = self.rays[name]

            # Lift: one point per pixel, at its true depth.
            keep = (d < self.max_depth) & (PRIORITY[tags] > 0)
            cam = np.stack([ray_x[keep] * d[keep], ray_y[keep] * d[keep], d[keep],
                            np.ones(keep.sum())])
            ego = self.ego_from_cam[name] @ cam
            x, y, z = ego[0], ego[1], ego[2]
            t = tags[keep]
            low = z < self.max_height
            x, y, t = x[low], y[low], t[low]

            # Splat: into the cell under each point; the most important class wins.
            i, j, inside = g.index(x, y)
            r, c = g.to_image_rc(i[inside], j[inside])
            cell = r * g.n + c
            t = t[inside]
            prio = PRIORITY[t]
            order = np.argsort(prio, kind="stable")       # low first, high last
            cell, prio, t = cell[order], prio[order], t[order]
            win = prio >= best[cell]
            best[cell[win]] = prio[win]
            tag_of[cell[win]] = t[win]

        if stamp is None:
            self.get_logger().info("waiting for class and depth images (surround_rig; "
                                   "with labels:=network, seg_node rig:=true)",
                                   throttle_duration_sec=5.0)
            return
        bgr = self.colors[tag_of].reshape(g.n, g.n, 3)
        g.draw_ego(bgr)
        h = Header()
        h.frame_id = EGO_FRAME
        h.stamp = stamp
        if self.context.ok():           # not mid-shutdown (Ctrl+C)
            self.pub.publish(image_msg(bgr, h))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemanticBev()
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
