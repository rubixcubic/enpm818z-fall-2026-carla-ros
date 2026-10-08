#!/usr/bin/env python3
"""CARLA's true classes for the bridge's front camera: the answer key.

Slide: 'A class for every pixel, in CARLA'. CARLA's semantic segmentation
camera writes the true class of every pixel, with no network. The L2 bridge
has no such camera, so this node attaches one to the AV, at the same place as
the bridge's front camera and with the same lens (size and field of view):
both cameras then see the same picture, pixel for pixel, at every tick.

    out  /l5/seg/truth_tags  mono8  CARLA's tag per pixel (1 road, 2 sidewalk,
                                    24 lane marking, ...; classes.py)
         /l5/seg/truth       bgr8   the same in CARLA's colors (CityScapes palette)

Both carry the same frame and time stamp as the bridge's image of that tick,
so seg_eval can pair them with the network's answer.

How the place is copied: the node finds the bridge's camera among the
vehicle's sensors (the one whose size and field of view match the bridge's
camera_info), reads both world poses from one snapshot, and takes the
camera's pose relative to the vehicle. It logs how far apart the two cameras
are once both are running (expected: under 1 mm).

This node never calls world.tick(). The bridge owns the clock.
"""

from __future__ import annotations

import math

import carla
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from l2_carla_demo import conversions as conv
from l5_seg_demo.classes import COLORS_BGR
from l5_seg_demo.ros_util import spin

CAMERA_OPTICAL_FRAME = "ego_vehicle/rgb_front_optical"   # the bridge's


def relative_transform(parent: carla.Transform, child: carla.Transform) -> carla.Transform:
    """The child's pose in the parent's frame, as a carla.Transform.

    CARLA's rotation matrix (LibCarla Transform::GetMatrix) has
    sin(pitch) at [2, 0], so pitch = asin(R[2, 0]),
    yaw = atan2(R[1, 0], R[0, 0]) and roll = atan2(-R[2, 1], R[2, 2]).
    """
    rel = np.array(parent.get_inverse_matrix()) @ np.array(child.get_matrix())
    R = rel[:3, :3]
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, R[2, 0]))))
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    roll = math.degrees(math.atan2(-R[2, 1], R[2, 2]))
    return carla.Transform(carla.Location(x=float(rel[0, 3]), y=float(rel[1, 3]),
                                          z=float(rel[2, 3])),
                           carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))


class SegTruth(Node):

    def __init__(self) -> None:
        super().__init__("seg_truth")
        self.declare_parameter("host", "localhost")
        self.declare_parameter("port", 2000)
        self.declare_parameter("role_name", "ego")
        self.declare_parameter("camera_info_topic", "/carla/ego_vehicle/rgb_front/camera_info")

        client = carla.Client(self.get_parameter("host").value,
                              int(self.get_parameter("port").value))
        client.set_timeout(20.0)
        self.world = client.get_world()
        self.info = None
        self.camera = None
        self.bridge_camera = None
        self.checked = False

        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        # Reliable: a 1280 x 720 image is about 0.9 MB, and with best effort most
        # of them were lost on the way (measured: 73 of 160 arrived in 8 s), so
        # few pairs reached seg_eval.
        out_qos = QoSProfile(depth=5, reliability=QoSReliabilityPolicy.RELIABLE)
        self.create_subscription(CameraInfo, self.get_parameter("camera_info_topic").value,
                                 self._on_info, latched)
        self.pub_tags = self.create_publisher(Image, "/l5/seg/truth_tags", out_qos)
        self.pub_color = self.create_publisher(Image, "/l5/seg/truth", out_qos)
        self.search_timer = self.create_timer(1.0, self._attach)

    def _on_info(self, msg: CameraInfo) -> None:
        self.info = msg

    # ----------------------------------------------------------- attaching
    def _attach(self) -> None:
        if self.camera is not None:
            if not self.checked:
                self._check_pose()
            return
        if self.info is None:
            self.get_logger().info("waiting for the bridge's camera_info "
                                   "(is l2_carla_demo running?)", throttle_duration_sec=5.0)
            return
        role = self.get_parameter("role_name").value
        actors = self.world.get_actors()
        ego = next((a for a in actors.filter("vehicle.*")
                    if a.attributes.get("role_name") == role), None)
        if ego is None:
            return
        width, height = self.info.width, self.info.height
        fov = math.degrees(2.0 * math.atan(width / (2.0 * self.info.k[0])))
        # The bridge's camera: on the AV, same size and field of view as its
        # camera_info. (l5_bev_demo's roof cameras are smaller.) Lowest id first:
        # the bridge spawned its camera before anything else attached one.
        matches = sorted(
            (a for a in actors.filter("sensor.camera.rgb")
             if a.parent is not None and a.parent.id == ego.id
             and int(a.attributes["image_size_x"]) == width
             and int(a.attributes["image_size_y"]) == height
             and abs(float(a.attributes["fov"]) - fov) < 0.01),
            key=lambda a: a.id)
        if not matches:
            self.get_logger().info("no camera on the AV matches the bridge's camera_info yet",
                                   throttle_duration_sec=5.0)
            return
        self.bridge_camera = matches[0]
        snap = self.world.get_snapshot()
        mount = relative_transform(snap.find(ego.id).get_transform(),
                                   snap.find(self.bridge_camera.id).get_transform())

        bp = self.world.get_blueprint_library().find("sensor.camera.semantic_segmentation")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", self.bridge_camera.attributes["fov"])
        bp.set_attribute("sensor_tick", self.bridge_camera.attributes.get("sensor_tick", "0.0"))
        self.camera = self.world.spawn_actor(bp, mount, attach_to=ego,
                                             attachment_type=carla.AttachmentType.Rigid)
        self.camera.listen(self._on_image)
        self.search_timer.cancel()
        self.search_timer = self.create_timer(2.0, self._attach)
        loc, rot = mount.location, mount.rotation
        self.get_logger().info(
            f"semantic camera on vehicle {ego.id}, copied from camera {self.bridge_camera.id}: "
            f"{width} x {height}, fov {fov:.1f} deg, at x {loc.x:.3f}, y {loc.y:.3f}, "
            f"z {loc.z:.3f} m, pitch {rot.pitch:.2f}, yaw {rot.yaw:.2f}, "
            f"roll {rot.roll:.2f} deg (CARLA's frame)")

    def _check_pose(self) -> None:
        """Both cameras in one snapshot: how far apart are they?"""
        snap = self.world.get_snapshot()
        a = snap.find(self.camera.id)
        b = snap.find(self.bridge_camera.id)
        if a is None or b is None:
            return
        ta, tb = a.get_transform(), b.get_transform()
        dist = ta.location.distance(tb.location)
        dyaw = abs((ta.rotation.yaw - tb.rotation.yaw + 180.0) % 360.0 - 180.0)
        dpitch = abs(ta.rotation.pitch - tb.rotation.pitch)
        self.get_logger().info(f"check: the two cameras are {1000 * dist:.2f} mm apart, "
                               f"yaw differs by {dyaw:.3f} deg, pitch by {dpitch:.3f} deg")
        self.checked = True

    # ------------------------------------------------------------ callback
    def _on_image(self, image) -> None:
        if not self.context.ok():
            return
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
        tags = np.ascontiguousarray(bgra[:, :, 2])     # the tag is the red channel
        header = conv.header(CAMERA_OPTICAL_FRAME, image.timestamp)
        msg = Image()
        msg.header = header
        msg.height, msg.width = image.height, image.width
        msg.encoding, msg.is_bigendian, msg.step = "mono8", 0, image.width
        msg.data = tags.tobytes()
        self.pub_tags.publish(msg)
        color = Image()
        color.header = header
        color.height, color.width = image.height, image.width
        color.encoding, color.is_bigendian, color.step = "bgr8", 0, 3 * image.width
        color.data = COLORS_BGR[tags].tobytes()
        self.pub_color.publish(color)

    def shutdown(self) -> None:
        if self.camera is not None:
            try:
                self.camera.stop()
                self.camera.destroy()
                self.get_logger().info("destroyed the semantic camera")
            except RuntimeError:
                pass
            self.camera = None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SegTruth()
    try:
        spin(node)
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
