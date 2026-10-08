#!/usr/bin/env python3
"""Publish the simulator's ground-truth pose.

Slide: 'Two things to get right', the line 'CARLA gives you ground truth, which
a real vehicle does not have.'

This node exists so you can do the one thing that is impossible on a road:
compare the error the filter is ACTUALLY making against the uncertainty it
REPORTS. Nothing else in this package uses it, and the filter must never
subscribe to it. That would be cheating, and it would hide exactly the failure
the lecture is about.

    ros2 run l3_ekf_demo truth
"""

import carla
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node


class TruthNode(Node):
    def __init__(self):
        super().__init__("l3_truth")
        self.declare_parameter("host", "localhost")
        self.declare_parameter("port", 2000)
        # The L2 bridge names its vehicle "ego" (carla_bridge_node.py).
        self.declare_parameter("role_name", "ego")
        host = self.get_parameter("host").value
        port = int(self.get_parameter("port").value)
        self.role = self.get_parameter("role_name").value

        client = carla.Client(host, port)
        client.set_timeout(20.0)
        self.world = client.get_world()
        self.ego = None
        self.origin = None

        self.pub = self.create_publisher(Odometry, "/carla/ego_vehicle/odometry_truth", 10)
        self.create_timer(0.05, self.tick)      # 20 Hz is plenty for a reference
        # Searching for the ego actor and not finding it used to be completely
        # silent: tick() just returned, twenty times a second, forever. That is
        # the quiet failure this lecture spends an hour warning about, so the
        # node now says so.
        self._ticks_without_ego = 0

    def find_ego(self):
        for a in self.world.get_actors().filter("vehicle.*"):
            if a.attributes.get("role_name") == self.role:
                return a
        return None

    def tick(self) -> None:
        if self.ego is None:
            self.ego = self.find_ego()
            if self.ego is None:
                self._ticks_without_ego += 1
                if self._ticks_without_ego % 100 == 1:      # every 5 s at 20 Hz
                    others = [a.attributes.get("role_name", "")
                              for a in self.world.get_actors().filter("vehicle.*")]
                    self.get_logger().warn(
                        "no vehicle with role_name='%s' in the world, so there is "
                        "no ground truth to publish. %s  Spawn the ego vehicle, or "
                        "pass role_name:=<name>."
                        % (self.role,
                           ("Vehicles present have role_name: %s." % ", ".join(
                               repr(r) for r in others)) if others
                           else "There are no vehicles in the world at all."))
                return
            self._ticks_without_ego = 0
            self.get_logger().info("found ego actor %d" % self.ego.id)
        tf = self.ego.get_transform()
        v = self.ego.get_velocity()
        # CARLA is left-handed with y flipped relative to ROS REP-103.
        # Getting this wrong is the L2 axis trap, and it does not raise either.
        x, y = tf.location.x, -tf.location.y
        if self.origin is None:
            self.origin = (x, y)
        m = Odometry()
        # Simulation time, the same clock the bridge stamps the sensors with.
        # The wall clock would put truth and GNSS on two different time axes,
        # and every comparison between them would be off by however long the
        # last tick took.
        sim = self.world.get_snapshot().timestamp.elapsed_seconds
        m.header.stamp.sec = int(sim)
        m.header.stamp.nanosec = int((sim - int(sim)) * 1e9)
        m.header.frame_id = "map"
        m.child_frame_id = "ego_vehicle_truth"
        m.pose.pose.position.x = x - self.origin[0]
        m.pose.pose.position.y = y - self.origin[1]
        m.pose.pose.orientation.w = 1.0
        m.twist.twist.linear.x = v.x
        m.twist.twist.linear.y = -v.y
        self.pub.publish(m)


def main() -> None:
    rclpy.init()
    node = TruthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
