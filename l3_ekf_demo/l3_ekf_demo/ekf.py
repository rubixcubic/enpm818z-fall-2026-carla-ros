#!/usr/bin/env python3
"""The filter itself. No ROS, no CARLA, so you can unit-test it.

Slide: 'CARLA Hands-On', Task 2, the line 'the filter itself is about twenty
lines of code, and the difficulty is in choosing Q.'

Count the lines in predict() and update(). The claim holds. Everything else in
this package is plumbing, logging and the simulator.

State is [px, py, vx, vy] in the local ENU frame, meters and meters/second.
Constant velocity, with IMU acceleration as a known control input u.
"""

import numpy as np


class EKF2D:
    def __init__(self, accel_process_sigma: float):
        self.x = np.zeros(4)                       # px, py, vx, vy
        self.P = np.diag([100.0, 100.0, 10.0, 10.0])
        self.q = float(accel_process_sigma)
        # GNSS observes position only: H picks the first two rows of the state.
        self.H = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]])

    # --- predict: age the estimate. P always grows. ------------------------
    def predict(self, dt: float, ax: float = 0.0, ay: float = 0.0) -> None:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        B = np.array([[0.5 * dt * dt, 0.0],
                      [0.0, 0.5 * dt * dt],
                      [dt, 0.0],
                      [0.0, dt]])
        self.x = F @ self.x + B @ np.array([ax, ay])
        # Q: the acceleration the constant-velocity model did not predict,
        # pushed through the same B. This is the parameter you will get wrong.
        Q = B @ (self.q ** 2 * np.eye(2)) @ B.T
        self.P = F @ self.P @ F.T + Q

    # --- update: fold in a measurement. P always shrinks, good fix or not. --
    def update(self, z: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nu = z - self.H @ self.x                   # innovation: the surprise
        S = self.H @ self.P @ self.H.T + R         # how surprised it expected to be
        K = self.P @ self.H.T @ np.linalg.inv(S)   # gain: how much to act on it
        self.x = self.x + K @ nu
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return nu, S

    # --- the whole point of Task 3 -----------------------------------------
    @staticmethod
    def nis(nu: np.ndarray, S: np.ndarray) -> float:
        """Surprise, in units of expected surprise. Dimensionless."""
        return float(nu.T @ np.linalg.inv(S) @ nu)


# --- Task 3's verdict, shared by the node and plot_nis --------------------
# The words follow the slide 'Reading epsilon over time: above, below or
# inside the band'.
VERDICT_TEXT = {
    "consistent": "consistent: the covariance is honest",
    "overconfident": "OVERCONFIDENT: Q or R too small (usually Q). Dangerous",
    "underconfident": "underconfident: wasteful, but safe",
}


def nis_verdict(values, dof: float, lo: float, hi: float,
                min_inside: float = 85.0) -> tuple[str, float, float]:
    """Return (verdict, mean NIS, percent inside [lo, hi]).

    The band is the consistency check: at least min_inside percent of the
    values inside it means consistent. Otherwise the mean says which way the
    filter is wrong. An honest filter's NIS averages dof, the number of values
    the sensor reports. Above dof, the surprises are bigger than the filter
    predicted: overconfident. Below, smaller: underconfident. (Comparing the
    mean with hi instead called a filter whose mean is between dof and hi
    underconfident, which is the wrong direction.)
    """
    a = np.asarray(values, dtype=float)
    inside = float(np.mean((a >= lo) & (a <= hi))) * 100.0
    mean = float(a.mean())
    if inside >= min_inside:
        verdict = "consistent"
    elif mean > dof:
        verdict = "overconfident"
    else:
        verdict = "underconfident"
    return verdict, mean, inside
