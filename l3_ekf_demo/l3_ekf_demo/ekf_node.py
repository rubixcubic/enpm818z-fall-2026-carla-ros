#!/usr/bin/env python3
"""Tasks 2 and 3: run the filter, and log whether you may believe it.

Slides: 'CARLA Hands-On' Tasks 2 and 3; 'The Innovation, and the NIS';
'Divergence and Gating'.

Subscribes to the topics l2_carla_demo publishes, so the filter never talks to
the simulator. It sees what a vehicle would see.

    ros2 run l3_ekf_demo ekf --ros-args --params-file config/ekf.yaml

Publishes:
    /l3/ekf/odometry   nav_msgs/Odometry   estimate, with its covariance
    /l3/ekf/nis        std_msgs/Float64    one number per GNSS fix
"""

import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float64

from l3_ekf_demo.ekf import EKF2D
from l3_ekf_demo.geodetic import LocalENU

SENSOR_QOS = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)


class EkfNode(Node):
    def __init__(self):
        super().__init__("l3_ekf")
        p = self.declare_parameters("", [
            ("origin_lat", 0.0), ("origin_lon", 0.0), ("latch_first_fix", True),
            ("gnss_sigma_m", 1.5), ("accel_process_sigma", 1.5),
            ("nis_dof", 2), ("nis_gate_chi2", 9.21),
            ("nis_lo_chi2", 0.051), ("nis_hi_chi2", 7.378),
            ("gate_enabled", False), ("max_consecutive_rejects", 10),
        ])
        g = {k.name: k.value for k in p}
        self.cfg = g

        self.frame = LocalENU(g["origin_lat"], g["origin_lon"])
        if not g["latch_first_fix"]:
            self.frame.latch(g["origin_lat"], g["origin_lon"])
        self.filt = EKF2D(g["accel_process_sigma"])
        self.R = (g["gnss_sigma_m"] ** 2) * np.eye(2)

        self._t_prev = None
        self._a = np.zeros(2)
        self._rejects = 0
        self._nis_hist = []

        self.create_subscription(Imu, "/carla/ego_vehicle/imu",
                                 self.on_imu, SENSOR_QOS)
        self.create_subscription(NavSatFix, "/carla/ego_vehicle/gnss",
                                 self.on_gnss, SENSOR_QOS)
        self.pub_odom = self.create_publisher(Odometry, "/l3/ekf/odometry", 10)
        self.pub_nis = self.create_publisher(Float64, "/l3/ekf/nis", 10)
        self.create_timer(10.0, self.report)
        self.get_logger().info("EKF up. gate=%s  sigma_gnss=%.2f m  q_accel=%.2f m/s^2"
                               % (g["gate_enabled"], g["gnss_sigma_m"],
                                  g["accel_process_sigma"]))

    # IMU drives prediction. Nothing measures position between fixes.
    def on_imu(self, msg: Imu) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._t_prev is None:
            self._t_prev = t
            return
        dt, self._t_prev = t - self._t_prev, t
        if not (0.0 < dt < 1.0):          # a stale or duplicated stamp
            return
        # The IMU measures acceleration along the AV's own axes (x forward,
        # y left). The state is in east-north. Rotate by the heading first,
        # or every acceleration is applied as if the AV were facing east.
        # The bridge publishes the heading in the orientation: yaw from east,
        # counterclockwise (l2_carla_demo conversions.imu_to_msg).
        q = msg.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        c, s = math.cos(yaw), math.sin(yaw)
        ax, ay = msg.linear_acceleration.x, msg.linear_acceleration.y
        self._a = np.array([c * ax - s * ay, s * ax + c * ay])
        self.filt.predict(dt, self._a[0], self._a[1])
        self.publish_odom(msg.header.stamp)

    # GNSS corrects it, once a second, and is the only thing that is bounded.
    def on_gnss(self, msg: NavSatFix) -> None:
        if not self.frame.latched:
            self.frame.latch(msg.latitude, msg.longitude)
            self.get_logger().info("local origin latched at %.7f, %.7f"
                                   % (msg.latitude, msg.longitude))
        e, n = self.frame.to_enu(msg.latitude, msg.longitude)
        # CARLA's GNSS is mirrored. In 0.9.16 latitude INCREASES with CARLA's
        # +y (map.transform_to_geolocation: +100 m in y gives +0.0008987 deg of
        # latitude), but CARLA's +y is the world's right of +x, so south. The
        # bridge's map frame, its IMU heading and the truth all use y = -CARLA y.
        # Measured on Town10HD: over one drive the truth moved (31.0, +34.2) m,
        # the unflipped GNSS (31.0, -34.2) m. Flip north so every input of the
        # filter is in the same frame.
        n = -n

        # peek at the innovation BEFORE committing, so the gate can refuse it
        nu = np.array([e, n]) - self.filt.H @ self.filt.x
        S = self.filt.H @ self.filt.P @ self.filt.H.T + self.R
        eps = EKF2D.nis(nu, S)

        if self.cfg["gate_enabled"] and eps > self.cfg["nis_gate_chi2"]:
            self._rejects += 1
            self.get_logger().warn("fix rejected, NIS %.2f > %.2f (%d in a row)"
                                   % (eps, self.cfg["nis_gate_chi2"], self._rejects))
            if self._rejects >= self.cfg["max_consecutive_rejects"]:
                # A gate rejects whatever disagrees with the estimate. If the
                # estimate is already wrong, it rejects the fixes that would
                # have saved it. Counting is the standard mitigation.
                self.get_logger().error(
                    "%d consecutive rejections: filter is UNHEALTHY, not merely "
                    "uncertain. Report it; do not retune Q to silence it."
                    % self._rejects)
            return

        self._rejects = 0
        self.filt.update(np.array([e, n]), self.R)
        self._nis_hist.append(eps)
        self.pub_nis.publish(Float64(data=eps))
        self.publish_odom(msg.header.stamp)

    def publish_odom(self, stamp) -> None:
        m = Odometry()
        m.header.stamp = stamp
        m.header.frame_id = "map"
        m.child_frame_id = "ego_vehicle"
        m.pose.pose.position.x = float(self.filt.x[0])
        m.pose.pose.position.y = float(self.filt.x[1])
        m.pose.pose.orientation.w = 1.0
        m.twist.twist.linear.x = float(self.filt.x[2])
        m.twist.twist.linear.y = float(self.filt.x[3])
        cov = [0.0] * 36
        cov[0], cov[1] = self.filt.P[0, 0], self.filt.P[0, 1]
        cov[6], cov[7] = self.filt.P[1, 0], self.filt.P[1, 1]
        m.pose.covariance = cov
        self.pub_odom.publish(m)

    def report(self) -> None:
        """Task 3 in one line: is the covariance honest?"""
        if not self._nis_hist:
            # Returning silently here meant an empty NIS history looked exactly
            # like a healthy filter: no output either way. Say which it is.
            self.get_logger().warn(
                "no GNSS fixes processed yet, so there is nothing to report. "
                "Check the bridge: ros2 topic hz /carla/ego_vehicle/gnss")
            return
        a = np.array(self._nis_hist)
        lo, hi = self.cfg["nis_lo_chi2"], self.cfg["nis_hi_chi2"]
        inside = float(np.mean((a >= lo) & (a <= hi))) * 100.0
        verdict = ("consistent" if 85.0 <= inside <= 100.0 else
                   "OVERCONFIDENT, P or R too small (usually Q)" if a.mean() > hi else
                   "underconfident, wasteful but safe")
        self.get_logger().info(
            "NIS over %d fixes: mean %.2f (want about %d), %.0f%% inside "
            "[%.3f, %.3f] -> %s" % (len(a), a.mean(), self.cfg["nis_dof"],
                                    inside, lo, hi, verdict))


def main() -> None:
    rclpy.init()
    node = EkfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
