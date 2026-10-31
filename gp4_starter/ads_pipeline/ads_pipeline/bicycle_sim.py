"""Offline simulator for tuning the GP4 controller without CARLA. Given.

The kinematic bicycle model at the rear axle (the hands-on's model), with a
first-order lag on the steering and on the acceleration, as the course's
reference simulation chose them: steering lag 0.2 s, acceleration lag
0.3 s, 4 m/s^2 at full throttle, 8 m/s^2 at full brake.
"""
import math

import numpy as np

from ads_pipeline.controller_core import (PIDController, pure_pursuit_steer,
                                          to_vehicle_cmd)

STEER_TAU, ACC_TAU = 0.2, 0.3
ACC_PER_THROTTLE, DEC_PER_BRAKE = 4.0, 8.0


def bicycle_step(x, y, theta, v, delta, a, dt, L):
    return (x + v * math.cos(theta) * dt, y + v * math.sin(theta) * dt,
            theta + v / L * math.tan(delta) * dt, max(0.0, v + a * dt))


def cross_track(path, x, y):
    """Distance from (x, y) to the path's polyline."""
    p = np.asarray(path)
    a, b = p[:-1], p[1:]
    ab = b - a
    t = np.clip(((x - a[:, 0]) * ab[:, 0] + (y - a[:, 1]) * ab[:, 1])
                / np.maximum((ab ** 2).sum(1), 1e-12), 0.0, 1.0)
    q = a + ab * t[:, None]
    return float(np.min(np.hypot(q[:, 0] - x, q[:, 1] - y)))


def track(path_xy, v_target=8.0, lookahead=5.0, L=2.875, max_steer=0.7,
          gains=(0.5, 0.05, 0.1, 2.0), dt=0.05, t_max=120.0):
    """Drive the path from its start with Pure Pursuit and the PID. Returns
    completed (reached the last point within 2 m), max and mean cross-track
    error, and the time taken."""
    path = np.asarray(path_xy, dtype=float)
    x, y = path[0]
    theta = math.atan2(path[1, 1] - path[0, 1], path[1, 0] - path[0, 0])
    v, delta, acc = 0.0, 0.0, 0.0
    pid = PIDController(*gains)
    ctes, t = [], 0.0
    while t < t_max:
        d_cmd = pure_pursuit_steer(path, (x, y), theta, lookahead, L)
        u = pid.compute(v_target, v, dt)
        thr, steer, brk = to_vehicle_cmd(d_cmd, u, max_steer, 0.75, 1.0)
        delta += (steer * max_steer - delta) * dt / STEER_TAU
        acc += ((thr * ACC_PER_THROTTLE - brk * DEC_PER_BRAKE) - acc) * dt / ACC_TAU
        x, y, theta, v = bicycle_step(x, y, theta, v, delta, acc, dt, L)
        ctes.append(cross_track(path, x, y))
        t += dt
        if math.hypot(x - path[-1, 0], y - path[-1, 1]) < 2.0:
            return {'completed': True, 'max_cross_track': max(ctes),
                    'mean_cross_track': float(np.mean(ctes)), 'time_s': t}
    return {'completed': False, 'max_cross_track': max(ctes),
            'mean_cross_track': float(np.mean(ctes)), 'time_s': t}
