"""GP4 Task 3 tests: Pure Pursuit, the PID with anti-windup, the command.

The Pure Pursuit case is the lecture's worked example: rear axle at
(82.50, -0.07) m, heading -0.0796 rad, lookahead 5 m, goal point
(87.22, -1.72) m, wheelbase 2.875 m: delta = -0.285 rad (to the right).

    cd ads_pipeline && python3 -m pytest test/test_gp4_controller.py -v
"""
import math
from pathlib import Path

import numpy as np
import pytest

from ads_pipeline.controller_core import (PIDController, creep_or_track,
                                          pure_pursuit_steer, to_vehicle_cmd)

ROOT = Path(__file__).resolve().parents[1]


def test_worked_example():
    path = np.array([[82.50, -0.07], [87.22, -1.72], [92.0, -4.0]])
    delta = pure_pursuit_steer(path, (82.50, -0.07), -0.0796, 5.0, 2.875)
    assert delta == pytest.approx(-0.285, abs=0.002)


def test_goal_straight_ahead_gives_no_steering():
    path = np.c_[np.arange(0.0, 20.0, 2.0), np.zeros(10)]
    assert pure_pursuit_steer(path, (0.0, 0.0), 0.0, 5.0, 2.875) == pytest.approx(0.0)


def test_pid_proportional_part():
    pid = PIDController(kp=0.1, ki=0.0, kd=0.0, windup_limit=2.0)
    assert pid.compute(8.0, 6.0, 0.05) == pytest.approx(0.2)


def test_pid_output_is_clamped():
    pid = PIDController(kp=1.0, ki=0.0, kd=0.0, windup_limit=2.0)
    assert pid.compute(8.0, 0.0, 0.05) == 1.0


def test_anti_windup_holds_the_integral():
    pid = PIDController(kp=0.0, ki=1.0, kd=0.0, windup_limit=2.0)
    for _ in range(1000):                  # 50 s of a 5 m/s error
        u = pid.compute(5.0, 0.0, 0.05)
    assert u == pytest.approx(1.0)
    assert abs(pid.integral) <= 2.0 + 1e-9
    for _ in range(10):                    # the error flips: the output follows quickly
        u = pid.compute(0.0, 5.0, 0.05)
    assert u < 1.0


def test_command_from_steering_and_speed_output():
    assert to_vehicle_cmd(0.35, 0.9, 0.7, 0.75, 1.0) == pytest.approx((0.75, 0.5, 0.0))
    assert to_vehicle_cmd(-1.0, -0.4, 0.7, 0.75, 1.0) == pytest.approx((0.0, -1.0, 0.4))


def test_creep_until_the_first_pose():
    assert creep_or_track(False, 2.0, 8.0) == (0.0, 2.0)
    assert creep_or_track(True, 2.0, 8.0) == (None, 8.0)


def test_closed_loop_on_the_lecture_curve():
    from ads_pipeline.bicycle_sim import track
    c = np.load(ROOT / 'test_data' / 'curve.npz')
    r = track(np.c_[c['x'], c['y']], v_target=8.0, lookahead=5.0)
    assert r['completed'] and r['max_cross_track'] <= MAX_CTE


# The reference controller's largest cross-track error on this curve, 5 m
# lookahead, 8 m/s: 0.458 m (bicycle_sim.track). The bound is 1.5 times it.
MAX_CTE = 0.69


def test_curve_speed_slows_for_a_turn_ahead():
    from ads_pipeline.controller_core import curve_speed
    straight = np.c_[np.arange(0.0, 60.0, 2.0), np.zeros(30)]
    assert curve_speed(straight, a_lat=2.0, a_brake=2.0) == np.inf
    # a quarter circle of radius 6 m starting 20 m ahead: v = sqrt(2.0 * 6) = 3.46 m/s
    # there; 20 m before it, braking at 2.0 m/s^2 allows sqrt(3.46^2 + 2 * 2.0 * 20) = 9.6
    th = np.linspace(0, np.pi / 2, 10)
    turn = np.c_[20.0 + 6 * np.sin(th), 6 - 6 * np.cos(th)]
    path = np.vstack([np.c_[np.arange(0.0, 20.0, 2.0), np.zeros(10)], turn])
    assert curve_speed(path, 2.0, 2.0) == pytest.approx(9.6, abs=0.4)
    # From the turn's start: the first point with a curvature is the arc's
    # second point, 1.05 m along (6 m x (pi / 2) / 9): sqrt(3.46^2 + 2 x 2.0 x 1.05) = 4.02.
    assert curve_speed(path[10:], 2.0, 2.0) == pytest.approx(4.02, abs=0.05)


def test_curve_speed_stops_at_the_end_of_the_path():
    from ads_pipeline.controller_core import curve_speed
    # A straight path that ends 8 m ahead, at the goal: braking at 2.0 m/s^2
    # stops the car in 8 m from sqrt(2 x 2.0 x 8) = 5.66 m/s.
    end = np.c_[np.arange(0.0, 10.0, 2.0), np.zeros(5)]
    assert curve_speed(end, 2.0, 2.0) == pytest.approx(5.66, abs=0.01)
    # Two points 1 m apart: sqrt(2 x 2.0 x 1) = 2.0 m/s.
    assert curve_speed([[0.0, 0.0], [1.0, 0.0]], 2.0, 2.0) == pytest.approx(2.0)
    # The end 58 m ahead is past the 40 m horizon: no limit yet.
    assert curve_speed(np.c_[np.arange(0.0, 60.0, 2.0), np.zeros(30)], 2.0, 2.0) == np.inf
