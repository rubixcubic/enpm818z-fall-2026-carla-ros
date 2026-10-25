"""Constant-velocity (CV) prediction and the closest approach of two predictions.

Plain functions with no ROS and no CARLA import, so the tests run anywhere.
Positions are (x, y) in meters and velocities (vx, vy) in m/s, in one frame.

This is L8's CV model: the road user keeps the velocity it has now,
    x(h) = x0 + vx * h,   y(h) = y0 + vy * h,
where h is the horizon, how far ahead in seconds.
"""

from __future__ import annotations

import math


def horizons(horizon: float, step: float) -> list[float]:
    """The predicted times step, 2 step, ..., horizon (L8: every 0.1 s for 5 s)."""
    n = int(round(horizon / step))
    return [round(step * (i + 1), 9) for i in range(n)]


def cv_path(x0: float, y0: float, vx: float, vy: float,
            times: list[float]) -> list[tuple[float, float]]:
    """CV positions at each time in `times`."""
    return [(x0 + vx * h, y0 + vy * h) for h in times]


def closest_approach(path_a: list[tuple[float, float]],
                     path_b: list[tuple[float, float]],
                     times: list[float]) -> tuple[float, float]:
    """The smallest distance between two predictions at the SAME time.

    Two paths that cross at different times never meet, so the distance is
    taken time by time, not between the two lines. Returns (distance in m,
    the time in s at which it happens).
    """
    best_d, best_t = math.inf, math.nan
    for (xa, ya), (xb, yb), t in zip(path_a, path_b, times):
        d = math.hypot(xa - xb, ya - yb)
        if d < best_d:
            best_d, best_t = d, t
    return best_d, best_t


def speed(vx: float, vy: float) -> float:
    return math.hypot(vx, vy)
