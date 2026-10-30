"""Drops GNSS fixes inside the team's outage window. Given.

/carla/gnss/fix in, /gnss/fix_used out. The window is in seconds from the
first fix's stamp, so it is the same at any replay speed.
"""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix


def in_outage(t, t_first, start_s, end_s):
    return start_s <= t - t_first < end_s


class GnssOutage(Node):
    def __init__(self):
        super().__init__('gnss_outage')
        self.start_s = self.declare_parameter('outage_start_s', 0.0).value
        self.end_s = self.declare_parameter('outage_end_s', 0.0).value
        self.t_first = None
        self.pub = self.create_publisher(NavSatFix, '/gnss/fix_used', 10)
        self.create_subscription(NavSatFix, '/carla/gnss/fix', self.on_fix, 10)

    def on_fix(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.t_first is None:
            self.t_first = t
        if not in_outage(t, self.t_first, self.start_s, self.end_s):
            self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(GnssOutage())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
