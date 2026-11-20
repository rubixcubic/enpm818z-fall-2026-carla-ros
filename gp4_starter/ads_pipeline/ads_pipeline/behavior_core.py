"""GP4 Task 4: the behavior tree.

A tree of four branches decides the state and the target speed. When more
than one condition holds, the safer branch wins, because the root tries the
branches in order, safest first: EMERGENCY_STOP, then STOP, then
OBSTACLE_AVOIDANCE, then LANE_FOLLOW.
"""
import math
from enum import Enum, auto

import numpy as np


class BehaviorState(Enum):
    LANE_FOLLOW = auto()
    STOP = auto()
    OBSTACLE_AVOIDANCE = auto()
    EMERGENCY_STOP = auto()


# Given: the four node types of L10's behavior_tree.py. A node's tick(bb)
# returns a status. bb is the blackboard, a dict that holds what the car
# senses now; the leaves read it and an Action writes the answer into it.
SUCCESS, FAILURE, RUNNING = 'Success', 'Failure', 'Running'


class Condition:
    """Checks a fact on the blackboard: Success if check(bb) is true, else
    Failure. name says what it checks."""

    def __init__(self, name, check):
        self.name, self.check = name, check

    def tick(self, bb):
        return SUCCESS if self.check(bb) else FAILURE


class Action:
    """Chooses a behavior: writes its state and speed into the blackboard and
    returns Running (driving is never done)."""

    def __init__(self, state, speed):
        self.state, self.speed = state, speed

    def tick(self, bb):
        bb['state'], bb['speed'] = self.state, self.speed
        return RUNNING


class Sequence:
    """Ticks its children in order and stops at the first one that does not
    succeed."""

    def __init__(self, *children):
        self.children = children

    def tick(self, bb):
        for child in self.children:
            status = child.tick(bb)
            if status != SUCCESS:
                return status
        return SUCCESS


class Fallback:
    """Ticks its children in order and stops at the first one that does not
    fail."""

    def __init__(self, *children):
        self.children = children

    def tick(self, bb):
        for child in self.children:
            status = child.tick(bb)
            if status != FAILURE:
                return status
        return FAILURE


def nearest_ahead(path_xy, tracks, lookahead_s, clearance, dt=0.1):
    """Given. How far along the path (m, from its first point, the car) the
    first object is whose constant-velocity position over the next
    lookahead_s seconds comes within clearance of the path; inf if none.
    tracks: [{'xy', 'v'}], map frame."""
    path = np.asarray(path_xy, dtype=float)
    if len(path) < 2 or not tracks:
        return math.inf
    along = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
    n = max(1, int(round(lookahead_s / dt)))
    times = np.linspace(0.0, lookahead_s, n + 1)
    best = math.inf
    for o in tracks:
        pos = np.asarray(o['xy'])[None, :] + times[:, None] * np.asarray(o['v'])[None, :]
        d = np.linalg.norm(pos[:, None, :] - path[None, :, :], axis=2)   # times x waypoints
        if d.min() <= clearance:
            k = int(np.argmin(d.min(axis=0)))
            best = min(best, float(along[k]))
    return best


def ahead_by_class(path_xy, tracks, lookahead_s, ped_lookahead_s, clearance):
    """Given. nearest_ahead with a shorter prediction for pedestrians: their
    tracks' speeds are noisy (the camera places them), and 2 s of a wrong
    speed carried people standing on the sidewalk into the band. tracks:
    [{'xy', 'v', 'cls'}]; a track with no 'cls' counts as a vehicle."""
    peds = [o for o in tracks if o.get('cls') == 'pedestrian']
    others = [o for o in tracks if o.get('cls') != 'pedestrian']
    return min(nearest_ahead(path_xy, others, lookahead_s, clearance),
               nearest_ahead(path_xy, peds, ped_lookahead_s, clearance))


def stop_line_speed(dist, target_speed, a_brake=2.0, margin=1.0):
    """Given. The speed (m/s) from which the car stops, braking at a_brake,
    margin meters before a stop line dist meters ahead; at most target_speed,
    0 when the line is that close or the distance is unknown. With a target
    of 0 as soon as the line was 20 m away, our car stopped 16 m short."""
    if math.isnan(dist) or dist <= margin:
        return 0.0
    return min(target_speed, math.sqrt(2.0 * a_brake * (dist - margin)))


def lidar_blocked(points_xy, path_xy, half_width=1.1, min_points=5):
    """Given. How far along the path (m, from its first point, the car) the
    nearest LiDAR point is that lies within half_width of the path; inf when
    fewer than min_points lie there. points_xy: above-ground LiDAR points,
    map frame. It sees what the camera misses, such as the back of a truck
    that fills the whole image."""
    pts = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    path = np.asarray(path_xy, dtype=float)
    if len(path) < 2 or len(pts) < min_points:
        return math.inf
    a, b = path[:-1], path[1:]
    ab = b - a
    seg = np.hypot(ab[:, 0], ab[:, 1])
    s0 = np.r_[0.0, np.cumsum(seg)][:-1]
    rel = pts[:, None, :] - a[None, :, :]                       # points x segments
    t = np.clip((rel * ab[None]).sum(axis=2) / np.maximum(seg ** 2, 1e-9)[None], 0.0, 1.0)
    d = np.linalg.norm(rel - t[..., None] * ab[None], axis=2)
    k = np.argmin(d, axis=1)
    i = np.arange(len(pts))
    inside = d[i, k] <= half_width
    if inside.sum() < min_points:
        return math.inf
    return float((s0[k] + t[i, k] * seg[k])[inside].min())


class BehaviorTree:
    def __init__(self, target_speed=8.0, slow_speed=3.0, emergency_dist=8.0,
                 obstacle_dist=15.0, stop_dist=20.0, creep_speed=2.0):
        self.target_speed, self.slow_speed = target_speed, slow_speed
        self.emergency_dist, self.obstacle_dist = emergency_dist, obstacle_dist
        self.stop_dist, self.creep_speed = stop_dist, creep_speed
        self.state = BehaviorState.LANE_FOLLOW
        self.root = None                                    # built on the first tick

    def tick(self, nearest_obstacle, light_state, dist_to_light, path_empty, path_seen):
        """Given. Next (state, target speed in m/s).

        nearest_obstacle: m along the path to the first object ahead (inf if
        none). light_state: 'Red', 'Yellow', 'Green', ... dist_to_light: m to
        the light's stop line, positive while it is ahead, nan when no light
        affects the car. path_empty: the planner found no route.
        path_seen: a path has been published at least once.

        The blackboard the leaves read has the keys 'nearest_obstacle',
        'light_state', 'dist_to_light' and 'path_empty', the arguments above."""
        if not path_seen:                                   # given: start-up
            self.state = BehaviorState.LANE_FOLLOW
            return self.state, self.creep_speed
        if self.root is None:
            self.root = self.build_tree()
        bb = {'nearest_obstacle': nearest_obstacle, 'light_state': light_state,
              'dist_to_light': dist_to_light, 'path_empty': path_empty}
        self.root.tick(bb)                                  # an Action sets bb['state'], bb['speed']
        self.state = bb['state']
        return self.state, bb['speed']

    def build_tree(self):
        """Return the root of the tree."""
        # YOURS: the root is a Fallback of four branches, safest first. Each of
        # the first three is a Sequence(Condition, Action); the last is a
        # bare Action. A condition reads the blackboard: lambda bb: ...
        # 1. EMERGENCY_STOP (speed 0): no route (path_empty) or an object
        #    within emergency_dist ahead.
        # 2. STOP (0): a red or yellow light whose stop line is ahead and
        #    within stop_dist (dist_to_light between 0 and stop_dist; nan
        #    means no light).
        # 3. OBSTACLE_AVOIDANCE (slow_speed): an object within obstacle_dist.
        # 4. LANE_FOLLOW (target_speed): nothing else holds.
        raise NotImplementedError('GP4 Task 4: BehaviorTree.build_tree')
