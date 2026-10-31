"""GP4 Task 4 node: the target speed, from the tracks, the light and the path.
Given.

On every /perception/tracks message (one per camera frame) it publishes
/behavior/state and /behavior/target_speed, stamped with the tracks' stamp,
so the chain detector, fusion, tracker, behavior, controller answers every
frame and sensor_manager can tick. The path is cut at the waypoint nearest
the car, and an object counts as ahead when its constant-velocity position
over the next lookahead_s seconds comes within clearance of what is left.

At the goal (less than 1.5 m of path left) it holds the car: speed 0, STOP.

Yours (behavior_core.py): BehaviorFSM.update.
"""
import math

import numpy as np
import rclpy
from ads_msgs.msg import TrackedObjectArray
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from ads_pipeline.behavior_core import BehaviorFSM, nearest_ahead
from ads_pipeline.ground_truth import parse_light


def goal_reached(path_left_xy, within=1.5):
    """True when less than `within` meters of path are left."""
    p = np.asarray(path_left_xy)
    if len(p) == 1:                # only the goal is left
        return True
    if len(p) == 0:                # no path: the no-route rule decides
        return False
    return float(np.hypot(*np.diff(p, axis=0).T).sum()) < within


class BehaviorNode(Node):
    def __init__(self):
        super().__init__('behavior_node')

        def p(name, default):
            return self.declare_parameter(name, default).value

        self.fsm = BehaviorFSM(p('target_speed', 8.0), p('slow_speed', 3.0),
                               p('emergency_distance', 8.0), p('obstacle_distance', 15.0),
                               p('stop_distance', 20.0), p('creep_speed', 2.0))
        self.lookahead_s, self.clearance = p('lookahead_s', 2.0), p('clearance', 3.0)
        self.path, self.path_seen, self.car = np.zeros((0, 2)), False, None
        self.light = ('Unknown', False, math.nan)
        self.pub_state = self.create_publisher(String, '/behavior/state', 10)
        self.pub_speed = self.create_publisher(TwistStamped, '/behavior/target_speed', 10)
        self.create_subscription(Path, '/planning/path', self.on_path, 10)
        self.create_subscription(String, '/ground_truth/traffic_light', self.on_light, 10)
        pose_topic = p('pose_topic', '/localization/pose')
        if pose_topic.startswith('/ground_truth'):
            self.create_subscription(Odometry, pose_topic, lambda m: self.on_pose(m.pose.pose), 50)
        else:
            self.create_subscription(PoseStamped, pose_topic, lambda m: self.on_pose(m.pose), 50)
        self.create_subscription(TrackedObjectArray, '/perception/tracks', self.on_tracks, 10)

    def on_path(self, msg):
        self.path = np.array([[q.pose.position.x, q.pose.position.y] for q in msg.poses]
                             ).reshape(-1, 2)
        self.path_seen = True

    def on_light(self, msg):
        self.light = parse_light(msg.data)

    def on_pose(self, pose):
        self.car = (pose.position.x, pose.position.y)

    def on_tracks(self, msg):
        path = self.path
        if len(path) and self.car is not None:
            k = int(np.argmin(np.hypot(path[:, 0] - self.car[0], path[:, 1] - self.car[1])))
            path = path[k:]
        tracks = [{'xy': (o.position.x, o.position.y), 'v': (o.velocity.x, o.velocity.y)}
                  for o in msg.objects]
        ahead = nearest_ahead(path, tracks, self.lookahead_s, self.clearance)
        state, speed = self.fsm.update(ahead, self.light[0], self.light[2],
                                       len(self.path) == 0, self.path_seen)
        name = state.name
        if self.path_seen and goal_reached(path):         # at the goal: hold the car
            speed, name = 0.0, 'STOP'
        s = String()
        s.data = name
        self.pub_state.publish(s)
        t = TwistStamped()
        t.header = msg.header
        t.twist.linear.x = float(speed)
        self.pub_speed.publish(t)


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(BehaviorNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
