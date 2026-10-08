#!/usr/bin/env python3
"""Task 1: log GNSS, IMU and ground truth to one CSV.

Slide: 'CARLA Hands-On', Task 1, 'What the noise actually looks like'.

Park the car and run this for a minute to measure R the way the Variance
section did it: many readings of a known point, then compute sigma. That is
the one number in the filter you can measure rather than tune.

    ros2 run l3_ekf_demo logger --ros-args -p out:=drive.csv
"""

import csv

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu, NavSatFix

SENSOR_QOS = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)


class LoggerNode(Node):
    def __init__(self):
        super().__init__("l3_logger")
        self.declare_parameter("out", "l3_log.csv")
        path = self.get_parameter("out").value
        self._fh = open(path, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(["t", "source", "a", "b"])
        self.create_subscription(NavSatFix, "/carla/ego_vehicle/gnss",
                                 lambda m: self.row(m, "gnss", m.latitude, m.longitude),
                                 SENSOR_QOS)
        self.create_subscription(Imu, "/carla/ego_vehicle/imu",
                                 lambda m: self.row(m, "imu",
                                                    m.linear_acceleration.x,
                                                    m.linear_acceleration.y),
                                 SENSOR_QOS)
        self.create_subscription(Odometry, "/carla/ego_vehicle/odometry_truth",
                                 lambda m: self.row(m, "truth",
                                                    m.pose.pose.position.x,
                                                    m.pose.pose.position.y), 10)
        self.get_logger().info("logging to %s  (gnss rows are DEGREES, not meters)" % path)

    def row(self, msg, source: str, a: float, b: float) -> None:
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._w.writerow(["%.6f" % t, source, "%.9f" % a, "%.9f" % b])

    def destroy_node(self):
        self._fh.close()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = LoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
