"""CV prediction and closest approach, with L8's own numbers where it has them."""

import math

import pytest

from l8_foxglove_demo import cv_predict as cvp


def test_horizons_are_l8s_50_times():
    h = cvp.horizons(5.0, 0.1)
    assert len(h) == 50
    assert h[0] == pytest.approx(0.1) and h[-1] == pytest.approx(5.0)


def test_cv_path_matches_l8_car_a_by_hand():
    # L8, "Step 2: constant velocity, by hand": x0 = 174.59, y0 = 1.98 m,
    # vx = -5.985, vy = -0.194 m/s; at h = 1, 3, 5 s the slide gets
    # (168.60, 1.79), (156.64, 1.40), (144.67, 1.01) m, rounded.
    path = cvp.cv_path(174.59, 1.98, -5.985, -0.194, [1.0, 3.0, 5.0])
    for (x, y), (xs, ys) in zip(path, [(168.60, 1.79), (156.64, 1.40), (144.67, 1.01)]):
        assert x == pytest.approx(xs, abs=0.006)
        assert y == pytest.approx(ys, abs=0.006)


def test_closest_approach_head_on():
    # Two cars 40 m apart, driving at each other at 8 m/s each: they meet at 2.5 s.
    times = cvp.horizons(5.0, 0.1)
    a = cvp.cv_path(0.0, 0.0, 8.0, 0.0, times)
    b = cvp.cv_path(40.0, 0.0, -8.0, 0.0, times)
    d, t = cvp.closest_approach(a, b, times)
    assert d == pytest.approx(0.0, abs=1e-9)
    assert t == pytest.approx(2.5)


def test_paths_that_cross_at_different_times_stay_apart():
    # Same crossing point (0, 0), reached at 1 s by one car and at 4 s by the other.
    times = cvp.horizons(5.0, 0.1)
    a = cvp.cv_path(-8.0, 0.0, 8.0, 0.0, times)       # at (0, 0) at t = 1 s
    b = cvp.cv_path(0.0, -32.0, 0.0, 8.0, times)      # at (0, 0) at t = 4 s
    d, _ = cvp.closest_approach(a, b, times)
    assert d > 2.0


def test_speed():
    assert cvp.speed(3.0, -4.0) == pytest.approx(5.0)
    assert math.isclose(cvp.speed(0.0, 0.0), 0.0)
