#!/usr/bin/env python3
"""L10: the controller's input topics, live from CARLA.

A second CARLA client next to the L2 bridge. The bridge owns the clock and drives
the AV on CARLA's autopilot; this node never ticks. It waits for the AV
(role_name "ego"), then, at every new simulation frame, publishes:

    /localization/odom      nav_msgs/Odometry           the AV's true pose and velocity
    /carla/speedometer      geometry_msgs/TwistStamped  the AV's speed, m/s, in twist.linear.x
    /behavior/target_speed  geometry_msgs/TwistStamped  the road's speed limit, m/s
    /planning/path          nav_msgs/Path               the lane center, 50 m ahead (every 0.5 s)

The launch file renames the bridge's IMU topic to /carla/imu/data, the fifth input.

Stand-ins, said plainly: the pose is CARLA's truth, not an EKF's estimate; the
target speed is the road's speed limit, not a behavior layer's choice; the path is
the lane center ahead, not a local planner's path. At a junction the path takes the
branch that turns least, which may not be the one the autopilot takes.

    ros2 launch l10_topics_demo l10_topics.launch.py
    ros2 topic echo /planning/path --once
"""

from __future__ import annotations

import math

import carla
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node

from l2_carla_demo.conversions import header as make_header
from l2_carla_demo.conversions import quaternion_from_euler

MAP_FRAME = "map"
EGO_FRAME = "ego_vehicle"


def ros_yaw(carla_yaw_deg: float) -> float:
    """CARLA's yaw (degrees, y to the right) as a ROS yaw (radians, y to the left)."""
    return math.radians(-carla_yaw_deg)


def heading_change(a_deg: float, b_deg: float) -> float:
    """The size of the turn from heading a to heading b, in degrees, 0 to 180."""
    return abs((b_deg - a_deg + 180.0) % 360.0 - 180.0)


class L10Topics(Node):

    def __init__(self) -> None:
        super().__init__("l10_topics")
        p = self.declare_parameter
        p("host", "localhost")
        p("port", 2000)
        p("rate_hz", 20.0)          # how often to look for a new frame: the bridge's 20 Hz
        p("path_length", 50.0)      # m of lane center ahead (chosen)
        p("path_step", 1.0)         # m between path points (chosen)
        p("path_every", 0.5)        # s of sim time between paths: a local planner's 0.5 s
        g = lambda name: self.get_parameter(name).value
        self.g = g

        self.client = carla.Client(g("host"), int(g("port")))
        self.client.set_timeout(5.0)
        self.world = None
        self.map = None
        self.ego = None
        self.last_frame = -1
        self.last_path_time = -math.inf

        self.pub_odom = self.create_publisher(Odometry, "/localization/odom", 10)
        self.pub_speed = self.create_publisher(TwistStamped, "/carla/speedometer", 10)
        self.pub_target = self.create_publisher(TwistStamped, "/behavior/target_speed", 10)
        self.pub_path = self.create_publisher(Path, "/planning/path", 10)

        self.timer = self.create_timer(1.0 / float(g("rate_hz")), self._tick)
        self.get_logger().info("waiting for the AV (role_name 'ego') from the L2 bridge")

    # ------------------------------------------------------------------ loop
    def _tick(self) -> None:
        try:
            self._step()
        except RuntimeError as exc:
            # The server did not answer in time: the bridge may be loading a map.
            self.world, self.map, self.ego = None, None, None
            self.get_logger().warn(f"CARLA is busy, trying again ({exc})",
                                   throttle_duration_sec=5.0)

    def _find_ego(self) -> None:
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.attributes.get("role_name") == "ego":
                self.ego = actor
                self.get_logger().info(f"found the AV: {actor.type_id}, id {actor.id}")
                return

    def _step(self) -> None:
        if self.world is None:
            self.world = self.client.get_world()
            self.map = self.world.get_map()
        snapshot = self.world.get_snapshot()
        if snapshot.frame == self.last_frame:
            return                                   # the bridge has not ticked yet
        self.last_frame = snapshot.frame
        now = snapshot.timestamp.elapsed_seconds
        if self.ego is None:
            self._find_ego()
            if self.ego is None:
                return
        # A client that never ticks reads the AV's state from the frame's snapshot:
        # its cached actor handle is not refreshed.
        snap = snapshot.find(self.ego.id)
        if snap is None:
            self.ego = None                          # the bridge stopped: wait again
            return

        tf = snap.get_transform()
        vel = snap.get_velocity()
        speed = math.hypot(vel.x, vel.y)
        self._publish_odom(tf, vel, snap.get_angular_velocity(), now)
        self._publish_twist(self.pub_speed, speed, now)
        # get_speed_limit() is in km/h, as CARLA's signs are.
        self._publish_twist(self.pub_target, self.ego.get_speed_limit() / 3.6, now)
        if now - self.last_path_time >= float(self.g("path_every")):
            self.last_path_time = now
            self._publish_path(tf, now)

    # ------------------------------------------------------------- messages
    def _publish_odom(self, tf: carla.Transform, vel: carla.Vector3D,
                      ang: carla.Vector3D, now: float) -> None:
        msg = Odometry()
        msg.header = make_header(MAP_FRAME, now)
        msg.child_frame_id = EGO_FRAME
        msg.pose.pose.position.x = float(tf.location.x)
        msg.pose.pose.position.y = float(-tf.location.y)
        msg.pose.pose.position.z = float(tf.location.z)
        msg.pose.pose.orientation = quaternion_from_euler(0.0, 0.0, ros_yaw(tf.rotation.yaw))
        # Twist in the AV's own frame, as ROS expects: x forward, y to the left.
        fwd, right = tf.get_forward_vector(), tf.get_right_vector()
        msg.twist.twist.linear.x = float(vel.x * fwd.x + vel.y * fwd.y)
        msg.twist.twist.linear.y = float(-(vel.x * right.x + vel.y * right.y))
        msg.twist.twist.angular.z = float(-math.radians(ang.z))     # deg/s, left-handed
        self.pub_odom.publish(msg)

    def _publish_twist(self, pub, value: float, now: float) -> None:
        msg = TwistStamped()
        msg.header = make_header(EGO_FRAME, now)
        msg.twist.linear.x = float(value)
        pub.publish(msg)

    def _publish_path(self, tf: carla.Transform, now: float) -> None:
        step = float(self.g("path_step"))
        wp = self.map.get_waypoint(tf.location, project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
        msg = Path()
        msg.header = make_header(MAP_FRAME, now)
        travelled = 0.0
        while wp is not None and travelled <= float(self.g("path_length")):
            loc, yaw = wp.transform.location, wp.transform.rotation.yaw
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(loc.x)
            pose.pose.position.y = float(-loc.y)
            pose.pose.position.z = float(loc.z)
            pose.pose.orientation = quaternion_from_euler(0.0, 0.0, ros_yaw(yaw))
            msg.poses.append(pose)
            nxt = wp.next(step)
            wp = min(nxt, key=lambda w: heading_change(yaw, w.transform.rotation.yaw)) if nxt else None
            travelled += step
        self.pub_path.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = L10Topics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
