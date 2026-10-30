"""GP3 Task 2: the EKF. State [x, y, theta, v] in the map frame.

Predict with the IMU yaw rate omega and constant speed. Update with GNSS,
which measures the antenna, mounted at antenna_xy = (dx, dy) in the vehicle
frame (x forward, y left).
"""
import math

import numpy as np


def wrap(a):
    """An angle in [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def nis(nu, S):
    """Normalized innovation squared: nu^T S^-1 nu."""
    return float(nu @ np.linalg.solve(S, nu))


def initial_state(z0, z1, antenna_xy):
    """Given. The first state, from two GNSS fixes far enough apart.

    While the car stands still, its heading cannot be seen, and the motion
    model cannot tell heading theta at speed v from heading theta + pi at
    speed -v. So the heading comes from the direction the antenna moved,
    z0 to z1, and the car's origin is placed behind the antenna at z1.
    """
    th = math.atan2(z1[1] - z0[1], z1[0] - z0[0])
    c, s = math.cos(th), math.sin(th)
    dx, dy = antenna_xy
    return np.array([z1[0] - (dx * c - dy * s), z1[1] - (dx * s + dy * c), th, 0.0])


class VehicleEKF:
    def __init__(self, q, r, antenna_xy):
        self.q = np.asarray(q, dtype=float)       # per second, scaled by dt
        self.R = np.diag(r)
        self.dx, self.dy = antenna_xy
        self.x = np.zeros(4)
        self.P = np.diag([100.0, 100.0, 1.0, 25.0])

    def f(self, x, omega, dt):
        """YOURS. The motion model: advance the state x = [x, y, theta, v] by dt seconds,
        turning at the IMU yaw rate omega (rad/s) with the speed held constant.
        Keep theta in [-pi, pi) (use wrap). Returns a (4,) array.
        """
        raise NotImplementedError('GP3 Task 2: f')

    def F_jacobian(self, x, omega, dt):
        """YOURS. The 4 x 4 Jacobian of f with respect to the state, at x. Derive it on
        paper first: four entries off the diagonal carry heading and speed error
        into position error. test_F_matches_finite_differences checks it.
        """
        raise NotImplementedError('GP3 Task 2: F_jacobian')

    def h(self, x):
        """YOURS. What the GNSS measures for state x: the map position of the ANTENNA,
        which sits at (self.dx, self.dy) in the vehicle frame (x forward, y left),
        not of the car's origin. Returns a (2,) array.
        """
        raise NotImplementedError('GP3 Task 2: h')

    def H_jacobian(self, x):
        """YOURS. The 2 x 4 Jacobian of h with respect to the state, at x. Your antenna
        offset puts nonzero entries in the heading column.
        """
        raise NotImplementedError('GP3 Task 2: H_jacobian')

    # Given: the Kalman filter equations.
    def predict(self, omega, dt):
        F = self.F_jacobian(self.x, omega, dt)
        self.x = self.f(self.x, omega, dt)
        self.P = F @ self.P @ F.T + np.diag(self.q) * dt

    def update(self, z):
        H = self.H_jacobian(self.x)
        nu = z - self.h(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        self.x[2] = wrap(self.x[2])
        self.P = (np.eye(4) - K @ H) @ self.P
        return nu, S
