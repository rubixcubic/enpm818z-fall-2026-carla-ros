"""GP4 Task 3 node: steering and speed from the path, the pose and the
behavior node's target speed. Given.

The speed is the behavior node's target, capped so the car can slow in
time for the sharpest curve ahead (a_lat 2.0 m/s^2 in the curve, a_brake
2.0 m/s^2 before it) and stop at the goal once it is within 40 m.

The car's speed comes from /carla/speedometer (a wheel-speed reading), not
from the EKF: the EKF's speed jumps at each GNSS fix and lags when the car
brakes, and in a turn the controller then braked the car to a stop.

The lookahead is lookahead_distance (5 m) or lookahead_time (1 s) of travel
at the current speed, whichever is longer: the EKF's pose jumps a little at
each GNSS fix, and a short lookahead turns each jump into a swerve.

On each /behavior/target_speed it takes the latest pose at or before that
stamp, moves it from the car's origin to the middle of the rear axle
(rear_offset meters behind, from your team's values), and publishes one
/control/vehicle_cmd stamped with the target's stamp: the answer
sensor_manager waits for before the next tick. Before the first pose
(GP3's EKF waits until the car has moved 3 m) it drives straight at
creep_speed.

Yours (controller_core.py): pure_pursuit_steer and PIDController.compute.
"""
import bisect
import math

import numpy as np
import rclpy
from ads_msgs.msg import VehicleCmd
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from ads_pipeline.controller_core import (PIDController, creep_or_track, curve_speed,
                                          pure_pursuit_steer, to_vehicle_cmd)


def stamp_s(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')

        def p(name, default):
            return self.declare_parameter(name, default).value

        self.lookahead = p('lookahead_distance', 5.0)
        self.lookahead_time = p('lookahead_time', 1.0)
        self.wheelbase = p('wheelbase', 2.875)
        self.rear_offset = p('rear_offset', 1.4)
        # max_steer_angle: the wheel angle CARLA gives at steer = 1 (measured
        # per vehicle, 70 deg = 1.222 rad for every rig); steer_limit: the most
        # this controller commands.
        self.max_steer = p('max_steer_angle', 1.222)
        self.steer_limit = p('steer_limit', 0.7)
        self.max_throttle, self.max_brake = p('max_throttle', 0.75), p('max_brake', 1.0)
        self.creep_speed = p('creep_speed', 2.0)
        self.a_lat, self.a_brake = p('a_lat', 2.0), p('a_brake', 2.0)
        self.pid = PIDController(p('pid_kp', 0.5), p('pid_ki', 0.05), p('pid_kd', 0.1),
                                 p('pid_windup_limit', 2.0))
        self.path = np.zeros((0, 2))
        self.pose_t, self.poses = [], []         # stamps, (x, y, theta, v)
        self.speed_t, self.speeds = [], []       # the speedometer: stamps, m/s
        self.t_last = None
        self.pub = self.create_publisher(VehicleCmd, '/control/vehicle_cmd', 10)
        self.create_subscription(Path, '/planning/path', self.on_path, 10)
        pose_topic = p('pose_topic', '/localization/pose')
        if pose_topic.startswith('/ground_truth'):
            self.create_subscription(Odometry, pose_topic, self.on_odom, 50)
        else:
            self.create_subscription(Odometry, '/localization/odom', self.on_odom, 50)
        self.create_subscription(TwistStamped, '/carla/speedometer', self.on_speedometer, 50)
        self.create_subscription(TwistStamped, '/behavior/target_speed', self.on_target, 10)

    def on_path(self, msg):
        self.path = np.array([[q.pose.position.x, q.pose.position.y] for q in msg.poses])

    def on_odom(self, msg):
        t = stamp_s(msg.header)
        if self.pose_t and t <= self.pose_t[-1]:
            return
        tw = msg.twist.twist.linear
        self.pose_t.append(t)
        self.poses.append((msg.pose.pose.position.x, msg.pose.pose.position.y,
                           yaw_of(msg.pose.pose.orientation), math.hypot(tw.x, tw.y)))
        if len(self.pose_t) > 2000:
            del self.pose_t[:1000], self.poses[:1000]

    def on_speedometer(self, msg):
        self.speed_t.append(stamp_s(msg.header))
        self.speeds.append(msg.twist.linear.x)
        if len(self.speed_t) > 2000:
            del self.speed_t[:1000], self.speeds[:1000]

    def on_target(self, msg):
        t = stamp_s(msg.header)
        i = bisect.bisect_right(self.pose_t, t + 1e-3) - 1
        have_pose = i >= 0 and len(self.path) >= 2
        override, speed = creep_or_track(have_pose, self.creep_speed, msg.twist.linear.x)
        dt = 0.05 if self.t_last is None else max(1e-3, t - self.t_last)
        self.t_last = t
        j = bisect.bisect_right(self.speed_t, t + 1e-3) - 1
        # The speedometer, as a real car's wheel speed: the EKF's speed jumps at
        # each GNSS fix and lags when the car brakes. The EKF's if none came yet.
        v = self.speeds[j] if j >= 0 else (self.poses[i][3] if i >= 0 else 0.0)
        if override is not None:
            delta = override
        else:
            x, y, th, _ = self.poses[i]
            rear = (x - self.rear_offset * math.cos(th), y - self.rear_offset * math.sin(th))
            ld = max(self.lookahead, self.lookahead_time * v)   # longer at speed: less swerve
            delta = pure_pursuit_steer(self.path, rear, th, ld, self.wheelbase)
            delta = max(-self.steer_limit, min(self.steer_limit, delta))
            k = int(np.argmin(np.hypot(self.path[:, 0] - x, self.path[:, 1] - y)))
            speed = min(speed, curve_speed(self.path[k:], self.a_lat, self.a_brake))
        u = self.pid.compute(speed, v, dt)
        thr, steer, brk = to_vehicle_cmd(delta, u, self.max_steer,
                                         self.max_throttle, self.max_brake)
        if speed <= 0.0 and v < 0.5:
            thr, brk = 0.0, self.max_brake           # hold the car at a stop
        cmd = VehicleCmd()
        cmd.header = msg.header
        cmd.throttle, cmd.steer, cmd.brake = float(thr), float(steer), float(brk)
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(ControllerNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
