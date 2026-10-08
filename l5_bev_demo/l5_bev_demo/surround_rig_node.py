#!/usr/bin/env python3
"""Four roof cameras around the ego vehicle, for the camera bird's-eye views.

Slides: 'Before learning: inverse perspective mapping' and 'Lift, by hand'.

The L2 bridge has one camera, facing forward. A view from above needs the
whole circle around the AV, so this node attaches four more cameras to the
bridge's vehicle: front, left, right and rear, each tilted down. At each place
it mounts three CARLA cameras with the same pose and the same lens:

    rgb        what a real camera sees                    -> ipm_bev
    semantic   the class of each pixel, CARLA's truth     -> semantic_bev
    depth      the distance of each pixel, CARLA's truth  -> semantic_bev

A real AV has only the first one. The other two are what a perfect network
would output, so the semantic view shows what lift-splat can do when its depth
is right, not what a trained network does.

This node never calls world.tick(). The bridge owns the clock; these cameras
simply fire when it ticks. Start the bridge first:

    ros2 launch l2_carla_demo demo.launch.py
    ros2 run l5_bev_demo surround_rig --ros-args --params-file <bev.yaml>
"""

import carla
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster

from l2_carla_demo import conversions as conv

EGO_FRAME = "ego_vehicle"
CAMERAS = ("front", "left", "right", "rear")


class SurroundRig(Node):

    def __init__(self) -> None:
        super().__init__("surround_rig")
        self.declare_parameter("host", "localhost")
        self.declare_parameter("port", 2000)
        self.declare_parameter("role_name", "ego")
        self.declare_parameter("image_width", 480)
        self.declare_parameter("image_height", 360)
        self.declare_parameter("fov", 100.0)
        self.declare_parameter("pitch", -30.0)
        self.declare_parameter("sensor_tick", 0.1)
        self.declare_parameter("half_extent", 25.0)   # shared grid, unused here
        self.declare_parameter("resolution", 0.2)

        self.width = int(self.get_parameter("image_width").value)
        self.height = int(self.get_parameter("image_height").value)
        self.fov = float(self.get_parameter("fov").value)
        self.actors: list = []
        self.ego = None

        client = carla.Client(self.get_parameter("host").value,
                              int(self.get_parameter("port").value))
        client.set_timeout(20.0)
        self.client = client
        self.world = client.get_world()

        sensor_qos = QoSProfile(depth=2,
                                reliability=QoSReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.pubs = {}
        for name in CAMERAS:
            self.pubs[name] = {
                "rgb": self.create_publisher(Image, f"/l5/{name}/rgb", sensor_qos),
                "semantic": self.create_publisher(
                    Image, f"/l5/{name}/semantic", sensor_qos),
                "depth": self.create_publisher(Image, f"/l5/{name}/depth", sensor_qos),
                "info": self.create_publisher(
                    CameraInfo, f"/l5/{name}/camera_info", latched),
            }
        self.static_tf = StaticTransformBroadcaster(self)

        # The bridge spawns its vehicle when it starts; we may be first, so look
        # for it once a second until it is there.
        self.search_timer = self.create_timer(1.0, self._find_ego)

    # ------------------------------------------------------------- the rig
    def _find_ego(self) -> None:
        role = self.get_parameter("role_name").value
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name") == role:
                self.ego = actor
                break
        if self.ego is None:
            self.get_logger().info(
                f"waiting for a vehicle with role_name '{role}' "
                "(is l2_carla_demo running?)", throttle_duration_sec=5.0)
            return
        self.search_timer.cancel()
        self._spawn_rig()

    def _mounts(self) -> dict:
        """One pose per camera, on the roof edges, from the vehicle's own size.

        CARLA's frame: x forward, y RIGHT, z up; yaw is clockwise seen from
        above. So the left camera has yaw -90 and the right one +90.
        """
        e = self.ego.bounding_box.extent          # half-sizes, in meters
        z = 2.0 * e.z + 0.3                       # 0.3 m above the roof
        pitch = float(self.get_parameter("pitch").value)
        return {
            "front": carla.Transform(carla.Location(x=0.5 * e.x, z=z),
                                     carla.Rotation(pitch=pitch, yaw=0.0)),
            "left": carla.Transform(carla.Location(y=-0.5 * e.y, z=z),
                                    carla.Rotation(pitch=pitch, yaw=-90.0)),
            "right": carla.Transform(carla.Location(y=0.5 * e.y, z=z),
                                     carla.Rotation(pitch=pitch, yaw=90.0)),
            "rear": carla.Transform(carla.Location(x=-0.5 * e.x, z=z),
                                    carla.Rotation(pitch=pitch, yaw=180.0)),
        }

    def _spawn_rig(self) -> None:
        bp_lib = self.world.get_blueprint_library()
        mounts = self._mounts()
        tick = str(self.get_parameter("sensor_tick").value)
        rigid = carla.AttachmentType.Rigid
        static = []
        now = self.world.get_snapshot().timestamp.elapsed_seconds

        for name in CAMERAS:
            for kind, blueprint in (("rgb", "sensor.camera.rgb"),
                                    ("semantic", "sensor.camera.semantic_segmentation"),
                                    ("depth", "sensor.camera.depth")):
                bp = bp_lib.find(blueprint)
                bp.set_attribute("image_size_x", str(self.width))
                bp.set_attribute("image_size_y", str(self.height))
                bp.set_attribute("fov", str(self.fov))
                bp.set_attribute("sensor_tick", tick)
                cam = self.world.spawn_actor(bp, mounts[name], attach_to=self.ego,
                                             attachment_type=rigid)
                cam.listen(lambda image, n=name, k=kind: self._on_image(n, k, image))
                self.actors.append(cam)

            body = f"l5/{name}"
            static.append(conv.transform_from_carla(mounts[name], EGO_FRAME, body, now))
            # Camera body (x forward) to optical frame (x right, y down, z
            # forward): the same fixed rotation the L2 bridge publishes.
            optical = TransformStamped()
            optical.header = conv.header(body, now)
            optical.child_frame_id = f"{body}_optical"
            optical.transform.rotation.x = -0.5
            optical.transform.rotation.y = 0.5
            optical.transform.rotation.z = -0.5
            optical.transform.rotation.w = 0.5
            static.append(optical)

        self.static_tf.sendTransform(static)
        self.get_logger().info(
            f"attached {len(self.actors)} cameras ({len(CAMERAS)} places x rgb, "
            f"semantic, depth) to vehicle {self.ego.id}")

    # ------------------------------------------------------------ callbacks
    def _on_image(self, name: str, kind: str, image) -> None:
        frame = f"l5/{name}_optical"
        pubs = self.pubs[name]
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            image.height, image.width, 4)

        if kind == "rgb":
            pubs["rgb"].publish(conv.image_to_msg(image, frame, Image))
            pubs["info"].publish(conv.camera_info_msg(image, frame, self.fov, CameraInfo))
            return

        msg = Image()
        msg.header = conv.header(frame, image.timestamp)
        msg.height, msg.width = image.height, image.width
        msg.is_bigendian = 0
        if kind == "semantic":
            # The class tag is the red channel (CARLA 0.9.16 docs). Bytes come
            # as B, G, R, A, so red is index 2.
            msg.encoding = "mono8"
            msg.step = image.width
            msg.data = np.ascontiguousarray(bgra[:, :, 2]).tobytes()
        else:
            # Depth is spread over three bytes, least significant first
            # (CARLA 0.9.16 docs):
            #   normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
            #   in_meters  = 1000 * normalized
            b = bgra[:, :, 0].astype(np.float64)
            g = bgra[:, :, 1].astype(np.float64)
            r = bgra[:, :, 2].astype(np.float64)
            meters = 1000.0 * (r + g * 256.0 + b * 65536.0) / (256.0 ** 3 - 1.0)
            msg.encoding = "32FC1"
            msg.step = 4 * image.width
            msg.data = meters.astype(np.float32).tobytes()
        pubs[kind].publish(msg)

    # -------------------------------------------------------------- cleanup
    def shutdown(self) -> None:
        for actor in self.actors:
            try:
                if actor.is_listening:
                    actor.stop()
            except RuntimeError:
                pass
        if self.actors:
            self.client.apply_batch([carla.command.DestroyActor(a) for a in self.actors])
            self.get_logger().info(f"destroyed {len(self.actors)} cameras")
            self.actors.clear()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SurroundRig()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass        # Ctrl+C: Jazzy raises the second one from spin()
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
