#!/usr/bin/env python3
"""Publish a CARLA sensor suite onto ROS 2 topics.

Slide: 'Demo 3: The Same Data in ROS 2'.

This is the interface the rest of the course is written against, and it is
deliberately a small readable node rather than CARLA's own --ros2 flag.  Two
reasons:

  1. In 0.9.16 the native interface builds topic names with a doubled slash,
     for example /carla//front_camera/image.  ROS 2 rejects a repeated '/' as
     an invalid topic name, so nothing subscribes and nothing tells you why
     (CARLA issue 9278).
  2. Every conversion in conversions.py is a coordinate-frame decision you
     should be able to see, because you will make the same decisions in GP1.

Run it with:

    ros2 launch l2_carla_demo demo.launch.py

The node owns the simulation clock: it puts the server in synchronous mode and
calls world.tick() from a timer.  Do not run a second client that also ticks.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image, Imu, NavSatFix, PointCloud2
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

import carla

from l2_carla_demo import conversions as conv

# Frame names follow the carla-ros-bridge convention, so anything students
# find online lines up with what this node publishes.
MAP_FRAME = "map"
EGO_FRAME = "ego_vehicle"
CAMERA_FRAME = "ego_vehicle/rgb_front"
CAMERA_OPTICAL_FRAME = "ego_vehicle/rgb_front_optical"
LIDAR_FRAME = "ego_vehicle/lidar"
RADAR_FRAME = "ego_vehicle/radar_front"
IMU_FRAME = "ego_vehicle/imu"
GNSS_FRAME = "ego_vehicle/gnss"


class CarlaBridge(Node):

    def __init__(self) -> None:
        super().__init__("carla_bridge")

        self.declare_parameter("host", "localhost")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 20.0)
        self.declare_parameter("town", "")          # empty: use the loaded map
        self.declare_parameter("delta", 0.05)       # 20 Hz simulation step
        self.declare_parameter("vehicle", "vehicle.tesla.model3")
        self.declare_parameter("autopilot", True)
        self.declare_parameter("image_width", 1280)
        self.declare_parameter("image_height", 720)
        self.declare_parameter("camera_fov", 90.0)
        self.declare_parameter("lidar_channels", 32)
        self.declare_parameter("lidar_range", 50.0)
        self.declare_parameter("lidar_points_per_second", 300000)
        self.declare_parameter("gnss_noise_m", 0.0)  # std dev per axis; 0: CARLA's default

        self.delta = self.get_parameter("delta").value
        self.camera_fov = self.get_parameter("camera_fov").value
        self.actors: list = []
        self.original_settings = None

        self._connect()
        self._make_publishers()
        self._spawn()
        self._publish_static_transforms()

        # One timer drives the whole thing. The server is in synchronous mode,
        # so nothing advances until this fires.
        self.timer = self.create_timer(self.delta, self._tick)
        self.get_logger().info(
            f"ticking at {1.0 / self.delta:.0f} Hz on {self.world.get_map().name}")

    # ---------------------------------------------------------------- setup
    def _connect(self) -> None:
        host = self.get_parameter("host").value
        port = self.get_parameter("port").value
        client = carla.Client(host, port)
        client.set_timeout(self.get_parameter("timeout").value)
        try:
            server = client.get_server_version()
        except RuntimeError as exc:
            raise SystemExit(
                f"Could not reach the CARLA server at {host}:{port} ({exc}). "
                "Is it running, and did you give it time to load the level?")
        if server != client.get_client_version():
            self.get_logger().warning(
                f"client {client.get_client_version()} against server {server}")

        self.client = client
        town = self.get_parameter("town").value
        self.world = client.load_world(town) if town else client.get_world()

        # Synchronous mode, from the first line. Asynchronous, the server hands
        # you whatever happened to be ready and no two runs agree.
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.delta
        self.world.apply_settings(settings)

    def _make_publishers(self) -> None:
        # Sensor data is best-effort: a dropped frame is better than a stalled
        # pipeline, and this matches what every real driver does.
        sensor_qos = QoSProfile(depth=5,
                                reliability=QoSReliabilityPolicy.BEST_EFFORT)
        # Intrinsics are latched instead: a subscriber that starts late still
        # needs them, and they never change.
        latched = QoSProfile(depth=1,
                             durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        self.pub_image = self.create_publisher(
            Image, "/carla/ego_vehicle/rgb_front/image", sensor_qos)
        self.pub_info = self.create_publisher(
            CameraInfo, "/carla/ego_vehicle/rgb_front/camera_info", latched)
        self.pub_lidar = self.create_publisher(
            PointCloud2, "/carla/ego_vehicle/lidar", sensor_qos)
        self.pub_radar = self.create_publisher(
            PointCloud2, "/carla/ego_vehicle/radar_front", sensor_qos)
        self.pub_imu = self.create_publisher(
            Imu, "/carla/ego_vehicle/imu", sensor_qos)
        self.pub_gnss = self.create_publisher(
            NavSatFix, "/carla/ego_vehicle/gnss", sensor_qos)

        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

    # ---------------------------------------------------------------- actors
    def _mounts(self, vehicle) -> dict:
        """Derive the mounting transforms from this vehicle's own body.

        bounding_box.extent is a HALF-size in metres, so the roof is near
        2 * extent.z and the front bumper near extent.x. Treat it as a full
        size and the camera ends up inside the bodywork, returning a perfectly
        valid image of upholstery.
        """
        e = vehicle.bounding_box.extent
        roof = 2.0 * e.z
        return {
            "camera": carla.Transform(carla.Location(x=0.5 * e.x, z=roof * 0.95)),
            "lidar": carla.Transform(carla.Location(z=roof + 0.10)),
            "radar": carla.Transform(carla.Location(x=e.x, z=0.5 * e.z),
                                     carla.Rotation(pitch=5.0)),
            "imu": carla.Transform(),
            "gnss": carla.Transform(),
        }

    def _spawn(self) -> None:
        bp_lib = self.world.get_blueprint_library()

        veh_bp = bp_lib.find(self.get_parameter("vehicle").value)
        veh_bp.set_attribute("role_name", "ego")
        self.vehicle = None
        for point in self.world.get_map().get_spawn_points():
            # try_spawn_actor returns None rather than raising when the point
            # is occupied, which matters the moment there is traffic.
            self.vehicle = self.world.try_spawn_actor(veh_bp, point)
            if self.vehicle is not None:
                break
        if self.vehicle is None:
            raise SystemExit("every spawn point on this map is occupied")
        self.actors.append(self.vehicle)
        self.traffic_manager = None
        if self.get_parameter("autopilot").value:
            # The autopilot is CARLA's Traffic Manager. With the world in
            # synchronous mode, the Traffic Manager must be synchronous too, or
            # it may never issue a command: the vehicle then sits at its spawn
            # point with autopilot "on". Seen here in 2 of 3 runs before this.
            self.traffic_manager = self.client.get_trafficmanager()
            self.traffic_manager.set_synchronous_mode(True)
            # Tick once first: before it, the TM may never move the new vehicle.
            self.world.tick()
            self.vehicle.set_autopilot(True, self.traffic_manager.get_port())

        self.mounts = self._mounts(self.vehicle)
        # Rigid, always. SpringArm smooths the sensor's motion for video, which
        # makes the sensor-to-vehicle transform stop being constant, and every
        # extrinsic published below assumed it was.
        rigid = carla.AttachmentType.Rigid

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x",
                             str(self.get_parameter("image_width").value))
        cam_bp.set_attribute("image_size_y",
                             str(self.get_parameter("image_height").value))
        cam_bp.set_attribute("fov", str(self.camera_fov))
        camera = self.world.spawn_actor(cam_bp, self.mounts["camera"],
                                        attach_to=self.vehicle,
                                        attachment_type=rigid)
        camera.listen(self._on_image)
        self.actors.append(camera)

        li_bp = bp_lib.find("sensor.lidar.ray_cast")
        li_bp.set_attribute("channels",
                            str(self.get_parameter("lidar_channels").value))
        li_bp.set_attribute("range",
                            str(self.get_parameter("lidar_range").value))
        li_bp.set_attribute(
            "points_per_second",
            str(self.get_parameter("lidar_points_per_second").value))
        # One full revolution per simulation step, so a sweep is one message.
        li_bp.set_attribute("rotation_frequency", str(1.0 / self.delta))
        lidar = self.world.spawn_actor(li_bp, self.mounts["lidar"],
                                       attach_to=self.vehicle,
                                       attachment_type=rigid)
        lidar.listen(self._on_lidar)
        self.actors.append(lidar)

        ra_bp = bp_lib.find("sensor.other.radar")
        ra_bp.set_attribute("horizontal_fov", "30")
        ra_bp.set_attribute("vertical_fov", "10")
        ra_bp.set_attribute("range", "100")
        radar = self.world.spawn_actor(ra_bp, self.mounts["radar"],
                                       attach_to=self.vehicle,
                                       attachment_type=rigid)
        radar.listen(self._on_radar)
        self.actors.append(radar)

        imu = self.world.spawn_actor(bp_lib.find("sensor.other.imu"),
                                     self.mounts["imu"],
                                     attach_to=self.vehicle,
                                     attachment_type=rigid)
        imu.listen(self._on_imu)
        self.actors.append(imu)

        gnss_bp = bp_lib.find("sensor.other.gnss")
        noise_m = float(self.get_parameter("gnss_noise_m").value)
        if noise_m > 0.0:
            # CARLA adds the noise to latitude and longitude in DEGREES and to
            # altitude in meters (GnssSensor.cpp, 0.9.16). Meters to degrees
            # comes from the map's own geolocation, the one the GNSS uses: the
            # change in latitude over 100 m of CARLA y, in longitude over 100 m
            # of CARLA x, measured at the spawn point.
            m = self.world.get_map()
            p = self.vehicle.get_transform().location
            g0 = m.transform_to_geolocation(p)
            gy = m.transform_to_geolocation(carla.Location(p.x, p.y + 100.0, p.z))
            gx = m.transform_to_geolocation(carla.Location(p.x + 100.0, p.y, p.z))
            lat_per_m = abs(gy.latitude - g0.latitude) / 100.0
            lon_per_m = abs(gx.longitude - g0.longitude) / 100.0
            gnss_bp.set_attribute("noise_lat_stddev", str(noise_m * lat_per_m))
            gnss_bp.set_attribute("noise_lon_stddev", str(noise_m * lon_per_m))
            gnss_bp.set_attribute("noise_alt_stddev", str(noise_m))
            self.get_logger().info(
                f"GNSS noise {noise_m:.2f} m per axis: {noise_m * lat_per_m:.3e} deg "
                f"of latitude, {noise_m * lon_per_m:.3e} deg of longitude")
        gnss = self.world.spawn_actor(gnss_bp,
                                      self.mounts["gnss"],
                                      attach_to=self.vehicle,
                                      attachment_type=rigid)
        gnss.listen(self._on_gnss)
        self.actors.append(gnss)

        self.get_logger().info(f"spawned {len(self.actors)} actors")

    # ------------------------------------------------------------ transforms
    def _publish_static_transforms(self) -> None:
        """The extrinsics, published once.

        These are the same six numbers per sensor that the calibration section
        is about. Here they are exact, because the mounting transform and the
        extrinsic are literally the same object. On a vehicle you would have to
        measure them, and they would drift.
        """
        now = self.world.get_snapshot().timestamp.elapsed_seconds
        static = [
            conv.transform_from_carla(self.mounts["camera"], EGO_FRAME,
                                      CAMERA_FRAME, now),
            conv.transform_from_carla(self.mounts["lidar"], EGO_FRAME,
                                      LIDAR_FRAME, now),
            conv.transform_from_carla(self.mounts["radar"], EGO_FRAME,
                                      RADAR_FRAME, now),
            conv.transform_from_carla(self.mounts["imu"], EGO_FRAME,
                                      IMU_FRAME, now),
            conv.transform_from_carla(self.mounts["gnss"], EGO_FRAME,
                                      GNSS_FRAME, now),
        ]

        # The optical frame IS the axis permutation from the projection slide,
        # written as a transform instead of a matrix. x right, y down, z along
        # the optical axis, which is what K expects and what the body frame is
        # not. Publishing it means RViz and image_geometry do the relabelling
        # for you, and you never multiply by P by hand again.
        optical = TransformStamped()
        optical.header = conv.header(CAMERA_FRAME, now)
        optical.child_frame_id = CAMERA_OPTICAL_FRAME
        optical.transform.rotation.x = -0.5
        optical.transform.rotation.y = 0.5
        optical.transform.rotation.z = -0.5
        optical.transform.rotation.w = 0.5
        static.append(optical)

        self.static_tf_broadcaster.sendTransform(static)

    # -------------------------------------------------------------- callbacks
    def _on_image(self, image) -> None:
        self.pub_image.publish(
            conv.image_to_msg(image, CAMERA_OPTICAL_FRAME, Image))
        self.pub_info.publish(
            conv.camera_info_msg(image, CAMERA_OPTICAL_FRAME,
                                 self.camera_fov, CameraInfo))

    def _on_lidar(self, cloud) -> None:
        self.pub_lidar.publish(conv.lidar_to_msg(cloud, LIDAR_FRAME))

    def _on_radar(self, measurement) -> None:
        self.pub_radar.publish(conv.radar_to_msg(measurement, RADAR_FRAME))

    def _on_imu(self, measurement) -> None:
        self.pub_imu.publish(conv.imu_to_msg(measurement, IMU_FRAME))

    def _on_gnss(self, measurement) -> None:
        self.pub_gnss.publish(conv.gnss_to_msg(measurement, GNSS_FRAME))

    # ------------------------------------------------------------------ loop
    def _tick(self) -> None:
        self.world.tick()
        snapshot = self.world.get_snapshot()
        now = snapshot.timestamp.elapsed_seconds
        self.tf_broadcaster.sendTransform(
            conv.transform_from_carla(self.vehicle.get_transform(),
                                      MAP_FRAME, EGO_FRAME, now))

    # --------------------------------------------------------------- cleanup
    def shutdown(self) -> None:
        """Stop every sensor, destroy every actor, hand the clock back.

        The last part matters as much as the first: leaving the server in
        synchronous mode with nothing ticking it looks exactly like a frozen
        simulator, and the next person to connect will think they broke it.
        """
        for actor in reversed(self.actors):
            try:
                if isinstance(actor, carla.Sensor) and actor.is_listening:
                    actor.stop()
            except RuntimeError:
                pass
        if self.actors:
            self.client.apply_batch(
                [carla.command.DestroyActor(a) for a in reversed(self.actors)])
            self.get_logger().info(f"destroyed {len(self.actors)} actors")
            self.actors.clear()
        if self.traffic_manager is not None:
            self.traffic_manager.set_synchronous_mode(False)
        if self.original_settings is not None:
            self.world.apply_settings(self.original_settings)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CarlaBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
