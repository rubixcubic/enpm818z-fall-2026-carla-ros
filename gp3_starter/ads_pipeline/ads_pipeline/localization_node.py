"""GP3 Task 2 node: IMU and GNSS in, pose out. The wrapper is given.

Yours: on_imu and on_gnss (the order of predict and update). Use message
stamps for dt, never the wall clock: a bag played at --rate 0.25 must give
the same answer.

    ros2 run ads_pipeline localization_node --ros-args --params-file ekf_config.yaml
"""
import csv
import math
import os

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix

from ads_pipeline.ekf_core import VehicleEKF, initial_state, nis
from ads_pipeline.geo import MapProjection
from l2_carla_demo.conversions import quaternion_from_euler


# The antenna must move this far (m) before the heading is set: about 5
# times the rigs' GNSS noise (0.56 m), so the direction is not noise.
INIT_DISTANCE = 3.0


def stamp_s(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class Clock:
    """Time of the last predict. advance_to returns how far to predict."""

    def __init__(self):
        self.t = None

    def advance_to(self, t):
        if self.t is None:
            self.t = t
            return 0.0
        dt = max(0.0, t - self.t)
        self.t = max(self.t, t)
        return dt


class LocalizationNode(Node):
    def __init__(self):
        super().__init__('localization_node')

        def p(name, default):
            return self.declare_parameter(name, default).value

        q = [p('q_x', 0.01), p('q_y', 0.01), p('q_theta', 0.001), p('q_v', 0.1)]
        r = [p('r_x', 0.36), p('r_y', 0.36)]
        self.ekf = VehicleEKF(q, r, (p('antenna_x', 0.0), p('antenna_y', 0.0)))
        self.proj = MapProjection(p('gnss_origin_lat', 0.0),
                                  p('gnss_origin_lon', 0.0))
        self.nis_csv = p('nis_csv', 'results/gp3/nis.csv')
        self.clock = Clock()
        self.omega = 0.0
        self.initialized = False
        self.first_fix = None
        self.nis_log = []
        self.pub_pose = self.create_publisher(PoseStamped, '/localization/pose', 10)
        self.pub_odom = self.create_publisher(Odometry, '/localization/odom', 10)
        self.create_subscription(Imu, '/carla/imu/data', self.on_imu, 50)
        self.create_subscription(NavSatFix, '/gnss/fix_used', self.on_gnss, 10)

    # ---- yours -------------------------------------------------------------
    def on_imu(self, msg):
        """YOURS. One IMU message. Advance self.clock to its stamp; if the filter is
        initialized and time moved forward, predict over that dt with the yaw rate
        of the PREVIOUS IMU message (it held over the interval). Then keep this
        message's yaw rate (angular_velocity.z) and publish (self.publish).
        """
        raise NotImplementedError('GP3 Task 2: on_imu')

    def on_gnss(self, msg):
        """YOURS. One GNSS fix. Convert it to the map frame (self.proj), and
        advance self.clock to the fix's stamp. Until the filter is initialized:
        keep the first fix in self.first_fix; once a fix is INIT_DISTANCE or
        more from it, set self.ekf.x with initial_state (the heading comes
        from that move) and mark the filter initialized. After that: predict
        over the dt if it is positive, update with the fix, and append
        (stamp, NIS) to self.nis_log.
        """
        raise NotImplementedError('GP3 Task 2: on_gnss')

    # ---- given -------------------------------------------------------------
    def publish(self, header):
        if not self.initialized:
            return
        x, y, th, v = self.ekf.x
        pose = PoseStamped()
        pose.header.stamp, pose.header.frame_id = header.stamp, 'map'
        pose.pose.position.x, pose.pose.position.y = float(x), float(y)
        pose.pose.orientation = quaternion_from_euler(0.0, 0.0, float(th))
        self.pub_pose.publish(pose)
        odom = Odometry()
        odom.header, odom.child_frame_id = pose.header, 'ego_vehicle'
        odom.pose.pose = pose.pose
        odom.twist.twist.linear.x = float(v)
        P = self.ekf.P
        cov = list(odom.pose.covariance)
        cov[0], cov[1], cov[6], cov[7] = P[0, 0], P[0, 1], P[1, 0], P[1, 1]
        cov[35] = P[2, 2]
        odom.pose.covariance = cov
        self.pub_odom.publish(odom)

    def save_nis(self):
        if not self.nis_log:
            return
        os.makedirs(os.path.dirname(self.nis_csv) or '.', exist_ok=True)
        with open(self.nis_csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['t', 'nis'])
            w.writerows(self.nis_log)


def main(args=None):
    rclpy.init(args=args)
    node = LocalizationNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.save_nis()
