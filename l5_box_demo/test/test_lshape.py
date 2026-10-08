"""L-shape fit on synthetic rectangles with a known center, size and heading."""

import math

import numpy as np
import pytest

from l5_box_demo.lshape import (fit_lshape, heading_error_mod_half_turn, wrap_half_turn)


def face_points(a, b, n):
    """n points spread evenly along the segment a to b (as in frustum.tex)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.array([a + (b - a) * (k + 0.5) / n for k in range(n)])


def car_corners(cx, cy, length, width, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    def at(u, v):
        return (cx + c * u - s * v, cy + s * u + c * v)
    hl, hw = length / 2, width / 2
    return {"rear_left": at(-hl, hw), "rear_right": at(-hl, -hw),
            "front_left": at(hl, hw), "front_right": at(hl, -hw)}


def test_car_b_from_the_frustum_slides():
    """Car B: center (12.0, -2.0), 4.5 x 1.9 m, heading 30 deg; the LiDAR sees only
    its rear face (15 points) and its left side (26 points), as on the slides."""
    k = car_corners(12.0, -2.0, 4.5, 1.9, math.radians(30))
    pts = np.vstack([face_points(k["rear_left"], k["rear_right"], 15),
                     face_points(k["rear_left"], k["front_left"], 26)])
    centroid = pts.mean(axis=0)
    assert centroid == pytest.approx([10.99, -1.89], abs=0.01)      # the slide's centroid
    box = fit_lshape(pts)
    assert math.hypot(box.cx - 12.0, box.cy + 2.0) < 0.15
    assert box.length == pytest.approx(4.5, abs=0.15)
    assert box.width == pytest.approx(1.9, abs=0.15)
    assert math.degrees(heading_error_mod_half_turn(box.yaw, math.radians(30))) < 1.5


@pytest.mark.parametrize("yaw_deg", [0, 15, 30, 45, 60, 89, 120, -40])
def test_rectangles_at_many_headings(yaw_deg):
    yaw = math.radians(yaw_deg)
    k = car_corners(5.0, 3.0, 4.6, 1.8, yaw)
    pts = np.vstack([face_points(k["rear_left"], k["rear_right"], 12),
                     face_points(k["rear_left"], k["front_left"], 30)])
    box = fit_lshape(pts)
    assert math.hypot(box.cx - 5.0, box.cy - 3.0) < 0.1
    assert box.length == pytest.approx(4.6, abs=0.1)
    assert box.width == pytest.approx(1.8, abs=0.1)
    assert math.degrees(heading_error_mod_half_turn(box.yaw, yaw)) < 1.5


def test_heading_is_modulo_180_degrees():
    assert wrap_half_turn(math.radians(200)) == pytest.approx(math.radians(20))
    assert math.degrees(heading_error_mod_half_turn(math.radians(10), math.radians(190))) \
        == pytest.approx(0.0, abs=1e-9)


def test_noise_tolerance():
    """Points scattered by 3 cm, as a LiDAR adds: still within 0.2 m and 3 deg."""
    rng = np.random.default_rng(1)
    k = car_corners(12.0, -2.0, 4.5, 1.9, math.radians(30))
    pts = np.vstack([face_points(k["rear_left"], k["rear_right"], 15),
                     face_points(k["rear_left"], k["front_left"], 26)])
    pts = pts + rng.normal(0, 0.03, pts.shape)
    box = fit_lshape(pts)
    assert math.hypot(box.cx - 12.0, box.cy + 2.0) < 0.2
    assert math.degrees(heading_error_mod_half_turn(box.yaw, math.radians(30))) < 3.0
