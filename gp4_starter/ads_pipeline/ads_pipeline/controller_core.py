"""GP4 Task 3: Pure Pursuit for steering and a PID with anti-windup for speed.

Frames: the map frame, heading counterclockwise from +x, steering positive
to the LEFT. Pure Pursuit works at the middle of the rear axle.
"""
import math

import numpy as np


def pure_pursuit_steer(path_xy, rear_xy, heading, lookahead, wheelbase):
    """YOURS. The front wheel angle (rad, + left). Find the waypoint nearest the rear
    axle; ahead of it, the goal point is where the path leaves the circle of
    radius lookahead around the rear axle (interpolate on that segment). Put
    the goal in the car's frame, take the angle alpha to it, and return
    atan(2 * wheelbase * sin(alpha) / distance to the goal).
    """
    raise NotImplementedError('GP4 Task 3: pure_pursuit_steer')


class PIDController:
    """Speed PID; the integral of the error is held within +-windup_limit."""

    def __init__(self, kp, ki, kd, windup_limit=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.windup_limit = windup_limit
        self.integral = 0.0
        self._prev_error = None

    def compute(self, setpoint, measurement, dt):
        """YOURS. One PID step on the speed error setpoint - measurement. Keep the
        integral of the error in self.integral, held within +-self.windup_limit
        (anti-windup). Return kp e + ki integral + kd de/dt, clamped to [-1, 1].
        """
        raise NotImplementedError('GP4 Task 3: PIDController.compute')


def to_vehicle_cmd(delta, u, max_steer, max_throttle, max_brake):
    """Given. (throttle, steer, brake) from the wheel angle (rad) and the PID
    output u: steer = delta / max_steer, clamped to [-1, 1]; u > 0 is
    throttle (capped at max_throttle), u < 0 is brake (capped at max_brake)."""
    steer = float(np.clip(delta / max_steer, -1.0, 1.0))
    return (min(max(u, 0.0), max_throttle), steer, min(max(-u, 0.0), max_brake))


def creep_or_track(have_pose, creep_speed, target_speed):
    """Given. Before the first pose (the EKF waits for the car to move), drive
    straight at creep_speed: (steering override 0.0, speed). After it:
    (None, target_speed)."""
    return (None, target_speed) if have_pose else (0.0, creep_speed)


def curve_speed(path_ahead_xy, a_lat=2.0, a_brake=2.0, horizon=40.0):
    """Given. The highest speed (m/s) from which the car can still slow, at
    a_brake, to sqrt(a_lat / curvature) at every point of the next horizon
    meters of path, and stop at the path's end (the goal) when the end is
    within the horizon. Curvature from each three consecutive points. inf on
    a straight path that ends beyond the horizon."""
    p = np.asarray(path_ahead_xy, dtype=float)
    if len(p) < 2:
        return math.inf
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.r_[0.0, np.cumsum(seg)]
    best = math.sqrt(2.0 * a_brake * s[-1]) if s[-1] <= horizon else math.inf
    for i in range(1, len(p) - 1):
        if s[i] > horizon:
            break
        a, b, c = p[i - 1], p[i], p[i + 1]
        area2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        sides = seg[i - 1] * seg[i] * math.dist(a, c)
        kappa = 2.0 * area2 / sides if sides > 1e-9 else 0.0
        if kappa < 1e-4:
            continue
        v_here = math.sqrt(a_lat / kappa)
        best = min(best, math.sqrt(v_here ** 2 + 2.0 * a_brake * s[i]))
    return best
