"""GP4 Task 4: the behavior state machine.

Four states. When more than one condition holds, the safer state wins:
EMERGENCY_STOP, then STOP, then OBSTACLE_AVOIDANCE, then LANE_FOLLOW.
"""
import math
from enum import Enum, auto

import numpy as np


class BehaviorState(Enum):
    LANE_FOLLOW = auto()
    STOP = auto()
    OBSTACLE_AVOIDANCE = auto()
    EMERGENCY_STOP = auto()


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


class BehaviorFSM:
    def __init__(self, target_speed=8.0, slow_speed=3.0, emergency_dist=8.0,
                 obstacle_dist=15.0, stop_dist=20.0, creep_speed=2.0):
        self.target_speed, self.slow_speed = target_speed, slow_speed
        self.emergency_dist, self.obstacle_dist = emergency_dist, obstacle_dist
        self.stop_dist, self.creep_speed = stop_dist, creep_speed
        self.state = BehaviorState.LANE_FOLLOW

    def update(self, nearest_obstacle, light_state, dist_to_light, path_empty, path_seen):
        """Next (state, target speed in m/s).

        nearest_obstacle: m along the path to the first object ahead (inf if
        none). light_state: 'Red', 'Yellow', 'Green', ... dist_to_light: m to
        the light's stop line, positive while it is ahead, nan when no light
        affects the car. path_empty: the planner found no route.
        path_seen: a path has been published at least once."""
        if not path_seen:                                   # given: start-up
            self.state = BehaviorState.LANE_FOLLOW
            return self.state, self.creep_speed
        # YOURS from here: the four states, the safer one wins.
        # EMERGENCY_STOP (speed 0): no route (path_empty) or an object within
        #   emergency_dist ahead. STOP (0): a red or yellow light whose stop
        #   line is ahead and within stop_dist (dist_to_light between 0 and
        #   stop_dist; nan means no light). OBSTACLE_AVOIDANCE (slow_speed): an
        #   object within obstacle_dist. Otherwise LANE_FOLLOW (target_speed).
        #   Set self.state and return (state, speed).
        raise NotImplementedError('GP4 Task 4: BehaviorFSM.update')
