"""GP4 Task 2: the route planner's search and its two re-planning triggers.

The graph is route_graph.RouteGraph: neighbors(i) gives the open edges out
of node i with their costs, xy[i] each node's map position.
"""
import heapq
import itertools
import math

import numpy as np


def astar_waypoints(graph, start, goal, max_nodes=10000):
    """YOURS. A* from node start to node goal on graph (route_graph.RouteGraph:
    neighbors(i) gives [(j, cost)], xy[i] the node's map position). Use the
    straight-line distance to the goal as the heuristic. Return (path,
    expanded): the node list from start to goal, or None when no route exists
    or after max_nodes expansions; and how many nodes you expanded.
    """
    raise NotImplementedError('GP4 Task 2: astar_waypoints')


def obstacle_on_path(path_xy, objects_xy, clearance):
    """YOURS. Trigger 1: True when any object's center is within clearance meters of
    any waypoint of the path.
    """
    raise NotImplementedError('GP4 Task 2: obstacle_on_path')


def off_path(path_xy, car_xy, max_deviation):
    """YOURS. Trigger 2: True when the car is more than max_deviation meters from the
    nearest waypoint of the path.
    """
    raise NotImplementedError('GP4 Task 2: off_path')


class Hysteresis:
    """Given. Turns a trigger into re-plans: a re-plan when the trigger fires,
    then none until the trigger has been false for clear_s seconds. Without
    it, an object that stays near the new route too would start a new
    search at every check."""

    def __init__(self, clear_s=1.0):
        self.clear_s = clear_s
        self.armed = True
        self.clear_since = None

    def should_replan(self, triggered, t):
        if triggered:
            self.clear_since = None
            if self.armed:
                self.armed = False
                return True
            return False
        if self.clear_since is None:
            self.clear_since = t
        if t - self.clear_since >= self.clear_s:
            self.armed = True
        return False
