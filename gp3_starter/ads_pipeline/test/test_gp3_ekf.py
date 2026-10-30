"""GP3 Task 2 tests: the EKF and the order of predict and update.

    cd ads_pipeline && python3 -m pytest test/test_gp3_ekf.py -v
"""
import math

import numpy as np
import pytest

from ads_pipeline.ekf_core import VehicleEKF, nis, wrap
from ads_pipeline.localization_node import Clock

Q = [0.01, 0.01, 0.001, 0.1]
R = [0.36, 0.36]


def numerical(fun, x, eps=1e-6):
    fx = fun(x)
    J = np.zeros((len(fx), len(x)))
    for i in range(len(x)):
        d = np.zeros(len(x))
        d[i] = eps
        J[:, i] = (fun(x + d) - fx) / eps
    return J


@pytest.mark.parametrize('theta', [0.0, 0.7, 2.5, -3.0])
def test_F_matches_finite_differences(theta):
    ekf = VehicleEKF(Q, R, antenna_xy=(1.2, -0.4))
    x = np.array([3.0, -2.0, theta, 7.5])
    J = numerical(lambda s: ekf.f(s, 0.2, 0.05), x)
    assert ekf.F_jacobian(x, 0.2, 0.05) == pytest.approx(J, abs=1e-4)


@pytest.mark.parametrize('theta', [0.0, 0.7, 2.5, -3.0])
def test_H_matches_finite_differences(theta):
    ekf = VehicleEKF(Q, R, antenna_xy=(1.2, -0.4))
    x = np.array([3.0, -2.0, theta, 7.5])
    assert ekf.H_jacobian(x) == pytest.approx(numerical(ekf.h, x), abs=1e-4)


def test_h_reports_the_antenna_not_the_car():
    ekf = VehicleEKF(Q, R, antenna_xy=(1.0, 0.0))
    assert ekf.h(np.array([0.0, 0.0, math.pi / 2, 0.0])) == pytest.approx(
        [0.0, 1.0])


def test_straight_drive_at_constant_speed():
    ekf = VehicleEKF(Q, R, antenna_xy=(0.0, 0.0))
    ekf.x = np.array([0.0, 0.0, 0.0, 10.0])
    for _ in range(20):
        ekf.predict(0.0, 0.05)
    assert ekf.x[:2] == pytest.approx([10.0, 0.0])


def test_heading_stays_wrapped():
    ekf = VehicleEKF(Q, R, antenna_xy=(0.0, 0.0))
    ekf.x = np.array([0.0, 0.0, math.pi - 0.01, 5.0])
    ekf.predict(1.0, 0.05)
    assert -math.pi <= ekf.x[2] <= math.pi


def test_update_pulls_toward_the_fix_and_shrinks_P():
    ekf = VehicleEKF(Q, R, antenna_xy=(0.0, 0.0))
    p0 = np.trace(ekf.P)
    ekf.update(np.array([1.0, 0.0]))
    assert 0.0 < ekf.x[0] < 1.0 and np.trace(ekf.P) < p0


def test_nis_of_a_one_sigma_miss_in_one_axis_is_one():
    assert nis(np.array([1.0, 0.0]), np.eye(2)) == pytest.approx(1.0)


def test_clock_uses_stamps_and_predicts_once_per_stamp():
    c = Clock()
    assert c.advance_to(10.00) == 0.0          # first message: no predict
    assert c.advance_to(10.05) == pytest.approx(0.05)
    assert c.advance_to(10.05) == 0.0          # GNSS at the IMU's stamp
    assert c.advance_to(10.04) == 0.0          # late message: never backwards


def test_wrap():
    assert wrap(3 * math.pi / 2) == pytest.approx(-math.pi / 2)


def test_initial_state_takes_heading_from_the_first_move():
    """A car driving toward -x: the heading is pi, not the filter's default 0,
    and the car's origin sits 1 m behind an antenna 1 m forward."""
    from ads_pipeline.ekf_core import initial_state
    x = initial_state((0.0, 0.0), (-3.0, 0.0), antenna_xy=(1.0, 0.0))
    assert abs(abs(x[2]) - math.pi) < 1e-9
    assert x[:2] == pytest.approx([-2.0, 0.0]) and x[3] == 0.0


def test_node_waits_until_the_car_has_moved():
    from ads_pipeline.localization_node import INIT_DISTANCE
    assert INIT_DISTANCE >= 3 * 0.56     # well above the rigs' GNSS noise
