"""L-shape fitting: the rotated rectangle that best fits a cluster seen from above.

Method: search-based rectangle fitting with the closeness criterion, from
    X. Zhang, W. Xu, C. Dong and J. M. Dolan, "Efficient L-Shape Fitting for
    Vehicle Detection Using Laser Scanners", IEEE Intelligent Vehicles
    Symposium (IV), 2017, pp. 54-59 (Algorithm 2 and Algorithm 4).
Autoware runs the same method in autoware_shape_estimation
(lib/model/bounding_box.cpp, commit 0d5e45d of autoware_universe); two of its
choices are used here because the paper gives no numbers for them:
    - the closest-edge distance is taken point by point, and
    - d0 = 0.1 m (the smallest distance a point votes with) and points farther
      than 0.4 m from both edges do not vote.
The angle step, 1 degree, is also Autoware's; the paper calls it delta.

Algorithm 2: for every direction theta in [0, 90) degrees, project the points
on e1 = (cos theta, sin theta) and e2 = (-sin theta, cos theta). Each point's
distance to the nearest of the four edges of the bounding rectangle in that
direction is d; the closeness score is the sum of 1 / max(d, d0). The best
theta gives the rectangle: its edges are the min and max of the projections.

One sweep cannot tell the front from the back, so the heading is returned
modulo 180 degrees, along the longer side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Box2D:
    cx: float          # center, in the frame of the input points
    cy: float
    length: float      # the longer side, m
    width: float       # the shorter side, m
    yaw: float         # direction of the longer side, rad, in [-pi/2, pi/2): mod 180 deg


def closeness(c1: np.ndarray, c2: np.ndarray, d0: float = 0.1, d_max: float = 0.4) -> float:
    """Algorithm 4: sum over points of 1 / max(d, d0), d = distance to the nearest edge."""
    d1 = np.minimum(c1.max() - c1, c1 - c1.min())
    d2 = np.minimum(c2.max() - c2, c2 - c2.min())
    d = np.minimum(d1, d2)
    d = d[d <= d_max]
    return float(np.sum(1.0 / np.maximum(d, d0)))


def fit_lshape(xy: np.ndarray, step_deg: float = 1.0, d0: float = 0.1,
               d_max: float = 0.4) -> Box2D:
    """Algorithm 2 with the closeness criterion. xy: (n, 2) points, n >= 3."""
    best_q, best_t = -1.0, 0.0
    for t in np.arange(0.0, 90.0, step_deg):
        th = math.radians(t)
        e1 = np.array([math.cos(th), math.sin(th)])
        e2 = np.array([-math.sin(th), math.cos(th)])
        q = closeness(xy @ e1, xy @ e2, d0, d_max)
        if q > best_q:
            best_q, best_t = q, th
    e1 = np.array([math.cos(best_t), math.sin(best_t)])
    e2 = np.array([-math.sin(best_t), math.cos(best_t)])
    c1, c2 = xy @ e1, xy @ e2
    m1, m2 = (c1.min() + c1.max()) / 2.0, (c2.min() + c2.max()) / 2.0
    s1, s2 = c1.max() - c1.min(), c2.max() - c2.min()
    center = m1 * e1 + m2 * e2
    if s1 >= s2:
        length, width, yaw = s1, s2, best_t
    else:
        length, width, yaw = s2, s1, best_t + math.pi / 2.0
    return Box2D(float(center[0]), float(center[1]), float(length), float(width),
                 wrap_half_turn(yaw))


def wrap_half_turn(a: float) -> float:
    """An angle modulo 180 degrees, in [-pi/2, pi/2)."""
    return (a + math.pi / 2.0) % math.pi - math.pi / 2.0


def heading_error_mod_half_turn(a: float, b: float) -> float:
    """|a - b| modulo 180 degrees, in [0, pi/2]: a box's heading has no front or back."""
    return abs(wrap_half_turn(a - b))


def corners(box: Box2D) -> np.ndarray:
    """The 4 corners, counterclockwise, (4, 2)."""
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    hl, hw = box.length / 2.0, box.width / 2.0
    local = np.array([[hl, hw], [-hl, hw], [-hl, -hw], [hl, -hw]])
    return local @ np.array([[c, s], [-s, c]]) + np.array([box.cx, box.cy])
