"""GP4 Task 2 node: plans your route and re-plans on the two triggers. Given.

At the first pose, it searches from the node nearest the car to your goal
(routes/<rig>.yaml) and publishes /planning/path. Then, every
1 / replan_frequency seconds, it computes both triggers afresh from the
latest tracks and pose:

1. a STANDING VEHICLE (a 'vehicle' track slower than static_speed, from
   GP3's track speeds) within obstacle_clearance of a waypoint AHEAD of the
   car: close the graph's edges within obstacle_clearance of it and search
   again. If no route avoids it, the edges are reopened and the current
   path stays: the behavior node stops for the object. Moving cars and
   pedestrians are the behavior node's job (slow down or stop). A vehicle
   standing within queue_m (30 m) of a traffic light's stop line is taken
   as queued at the light, not parked;
2. the car more than max_deviation from the nearest waypoint: search again
   from the node nearest the car.

Each trigger has its own Hysteresis, so a trigger that stays on starts one
search, not one per check. No route: an empty Path and an error in the log.

Yours (planner_core.py): astar_waypoints, obstacle_on_path, off_path.
"""
import math

import numpy as np
import rclpy
import yaml
from ads_msgs.msg import TrackedObjectArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from ads_pipeline.planner_core import (Hysteresis, astar_waypoints,
                                       obstacle_on_path, off_path)
from ads_pipeline.route_graph import RouteGraph
from l2_carla_demo.conversions import quaternion_from_euler


def path_ahead(path_xy, car_xy):
    """The path from the waypoint nearest the car on."""
    p = np.asarray(path_xy)
    if len(p) == 0:
        return p
    return p[int(np.argmin(np.hypot(p[:, 0] - car_xy[0], p[:, 1] - car_xy[1]))):]


def away_from_lights(objects_xy, stop_xy, queue_m=30.0):
    """The objects farther than queue_m from every traffic light's stop
    line: a vehicle standing closer is most likely queued at the light."""
    if len(stop_xy) == 0:
        return list(objects_xy)
    stops = np.asarray(stop_xy)
    return [o for o in objects_xy
            if np.min(np.hypot(stops[:, 0] - o[0], stops[:, 1] - o[1])) > queue_m]


def to_path_msg(xy, header):
    """A nav_msgs/Path through the points xy, each pose facing the next."""
    msg = Path()
    msg.header = header
    for i, (x, y) in enumerate(xy):
        j = min(i + 1, len(xy) - 1)
        k = j - 1 if j == i else i
        yaw = math.atan2(xy[j][1] - xy[k][1], xy[j][0] - xy[k][0])
        p = PoseStamped()
        p.header = header
        p.pose.position.x, p.pose.position.y = float(x), float(y)
        p.pose.orientation = quaternion_from_euler(0.0, 0.0, yaw)
        msg.poses.append(p)
    return msg


class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')

        def p(name, default):
            return self.declare_parameter(name, default).value

        route = yaml.safe_load(open(p('route_file', 'routes/route.yaml')))
        self.goal = tuple(route['goal_xy'])
        self.graph = RouteGraph.from_xodr(p('map_file', 'ads_pipeline/maps/Town01.xodr'))
        self.clearance = p('obstacle_clearance', 3.0)
        self.max_dev = p('max_deviation', 2.0)
        self.max_nodes = p('max_astar_nodes', 10000)
        self.static_speed = p('static_speed', 0.5)
        self.queue_m = p('queue_m', 30.0)
        points_file = p('points_file', '')
        self.stops = np.zeros((0, 2))
        if points_file:
            import json
            pts = json.load(open(points_file))
            self.stops = np.array([s for l in pts['lights'] for s in l['stop_xy']])
        clear_s = p('clear_s', 1.0)
        self.h_obstacle, self.h_deviation = Hysteresis(clear_s), Hysteresis(clear_s)
        self.pose, self.objects, self.path_xy, self.replans = None, [], np.zeros((0, 2)), 0
        self.pub = self.create_publisher(Path, '/planning/path', 10)
        pose_topic = p('pose_topic', '/localization/pose')
        if pose_topic.startswith('/ground_truth'):
            self.create_subscription(Odometry, pose_topic,
                                     lambda m: self.on_pose(m.header, m.pose.pose), 10)
        else:
            self.create_subscription(PoseStamped, pose_topic,
                                     lambda m: self.on_pose(m.header, m.pose), 10)
        self.create_subscription(TrackedObjectArray, '/perception/tracks', self.on_tracks, 10)
        self.create_timer(1.0 / p('replan_frequency', 2.0), self.check)

    def on_pose(self, header, pose):
        first = self.pose is None
        self.pose, self.header = (pose.position.x, pose.position.y), header
        if first:
            self.plan('first pose')

    def on_tracks(self, msg):
        standing = [(o.position.x, o.position.y) for o in msg.objects
                    if o.class_id == 'vehicle'
                    and math.hypot(o.velocity.x, o.velocity.y) < self.static_speed]
        self.objects = away_from_lights(standing, self.stops, self.queue_m)

    def plan(self, why, closed=None):
        start = self.graph.nearest(*self.pose)
        path, expanded = astar_waypoints(self.graph, start,
                                         self.graph.nearest(*self.goal), self.max_nodes)
        if path is None and closed:
            self.graph.reopen(closed)
            self.get_logger().warn(f'no route around the object ({why}); keeping the path')
            return
        if path is None:
            self.path_xy = np.zeros((0, 2))
            self.get_logger().error(f'no route ({why}; {expanded} nodes expanded)')
        else:
            self.path_xy = self.graph.xy[path]
            self.get_logger().info(f'route: {len(path)} nodes, {expanded} expanded ({why})')
        self.header.frame_id = 'map'
        self.pub.publish(to_path_msg(self.path_xy, self.header))

    def check(self):
        if self.pose is None:
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        ahead = path_ahead(self.path_xy, self.pose)
        near = [o for o in self.objects if obstacle_on_path(ahead, [o], self.clearance)]
        if self.h_obstacle.should_replan(bool(near), t):
            closed = []
            for o in near:
                self.graph.close_edges_near(o[0], o[1], self.clearance, record=closed)
            self.replans += 1
            self.plan(f'obstacle on the path, re-plan {self.replans}', closed)
        elif self.h_deviation.should_replan(off_path(self.path_xy, self.pose, self.max_dev), t):
            self.replans += 1
            self.plan(f'car off the path, re-plan {self.replans}')


def main(args=None):
    rclpy.init(args=args)
    try:
        rclpy.spin(PlannerNode())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
