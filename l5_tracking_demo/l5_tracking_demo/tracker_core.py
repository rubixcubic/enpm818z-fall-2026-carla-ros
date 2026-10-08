"""The multi-object tracker of L5's Tracking section, with no ROS in it.

Slides, in order:
  'One Kalman filter per object'           -> KalmanCV
  'Every frame, the same loop'             -> MultiTracker.step
  'Distance in units of the track's
   uncertainty'                            -> KalmanCV.epsilon, GATE_CHI2
  'Four ways to associate'                 -> associate_nn, associate_gnn
  'The track lifecycle'                    -> Track.status, M of N, coasting
  'Tempe, in tracking terms'               -> reset_on_class_change

Everything here is plain NumPy (and SciPy for the Hungarian algorithm), so you
can run it, test it and change it without CARLA or ROS:

    python3 -m pytest test/test_tracker_core.py

The ROS node (tracker_node.py) only converts messages to and from these
classes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

# 99 % of a chi-square with 2 degrees of freedom: the gate of L3 and L5.
GATE_CHI2 = 9.21

TENTATIVE, CONFIRMED, COASTING, DELETED = "tentative", "confirmed", "coasting", "deleted"


# --------------------------------------------------------------------------
# One Kalman filter per object
# --------------------------------------------------------------------------
class KalmanCV:
    """Constant-velocity Kalman filter, state x = [px, py, vx, vy] (L3).

    The state lives in a fixed frame (the bridge's 'map'), so a parked car
    stays put while the AV drives past it.
    """

    H = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0]])      # picks the position out of x

    def __init__(self, z: np.ndarray, R: np.ndarray, init_vel_sigma: float,
                 accel_sigma: float) -> None:
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([R[0, 0], R[1, 1], init_vel_sigma ** 2, init_vel_sigma ** 2])
        self.accel_sigma = accel_sigma

    def predict(self, dt: float) -> None:
        """x <- F x, P <- F P F^T + Q, with Q from a random acceleration."""
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[0.5 * dt * dt, 0.0],
                      [0.0, 0.5 * dt * dt],
                      [dt, 0.0],
                      [0.0, dt]])
        Q = (self.accel_sigma ** 2) * G @ G.T
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def innovation(self, z: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """nu = z - H x (detection minus predicted position), S = H P H^T + R."""
        nu = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        return nu, S

    def epsilon(self, z: np.ndarray, R: np.ndarray) -> float:
        """The NIS of L3, nu^T S^-1 nu: distance in units of the uncertainty."""
        nu, S = self.innovation(z, R)
        return float(nu @ np.linalg.solve(S, nu))

    def epsilon_and_logdet(self, z: np.ndarray, R: np.ndarray) -> tuple[float, float]:
        """epsilon, and ln|S|: how spread out the track expects its detection to be."""
        nu, S = self.innovation(z, R)
        return float(nu @ np.linalg.solve(S, nu)), float(np.log(np.linalg.det(S)))

    def update(self, z: np.ndarray, R: np.ndarray) -> None:
        nu, S = self.innovation(z, R)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        # Joseph form: keeps P symmetric and positive with rounding errors.
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T


# --------------------------------------------------------------------------
# Association: which detection belongs to which track
# --------------------------------------------------------------------------
def associate_nn(eps: np.ndarray, order: list[int], gate: float = GATE_CHI2,
                 cost: np.ndarray | None = None) -> dict[int, int]:
    """Nearest neighbor: each track IN TURN takes its nearest free detection.

    eps[t, d] is the distance of detection d from track t. `order` is the
    order in which the tracks choose. Change it and the answer can change:
    that is the exercise 'two tracks, three detections'. Pairs must pass the
    gate, eps < gate; among them each track takes the lowest `cost` (eps
    itself unless another cost is given).
    Returns {track row: detection column}.
    """
    cost = eps if cost is None else cost
    taken: set[int] = set()
    pairs: dict[int, int] = {}
    for t in order:
        best, best_cost = None, np.inf
        for d in range(eps.shape[1]):
            if d not in taken and eps[t, d] < gate and cost[t, d] < best_cost:
                best, best_cost = d, cost[t, d]
        if best is not None:
            pairs[t] = best
            taken.add(best)
    return pairs


def associate_gnn(eps: np.ndarray, gate: float = GATE_CHI2,
                  cost: np.ndarray | None = None) -> dict[int, int]:
    """Global nearest neighbor: the pairing with the lowest TOTAL cost.

    Solved with the Hungarian algorithm (scipy's linear_sum_assignment). The
    cost is eps unless another one is given. A pair outside the gate gets a
    cost so large that the solver only uses it when it has no other choice,
    and such pairs are dropped afterwards.
    """
    if eps.size == 0:
        return {}
    big = 1e6
    cost = np.where(eps < gate, eps if cost is None else cost, big)
    rows, cols = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] < big}


# --------------------------------------------------------------------------
# Tracks and their lifecycle
# --------------------------------------------------------------------------
@dataclass
class Detection:
    z: np.ndarray                  # position (x, y) in the fixed frame, m
    R: np.ndarray                  # its 2 x 2 covariance, m^2
    label: str = "object"
    size: tuple = (0.0, 0.0, 0.0)  # extent along x, y, z, m (for display)


@dataclass
class Track:
    id: int
    kf: KalmanCV
    label: str
    status: str = TENTATIVE
    hits: deque = field(default_factory=deque)   # last N frames: 1 detected, 0 not
    age: int = 0                                 # frames since the track started
    detections: int = 0                          # frames it was detected in
    last_seen: float = 0.0                       # time of the last detection, s


@dataclass
class StepReport:
    """What happened in one frame, for the log and the JSON topic."""
    detections: int = 0
    matched: int = 0
    started: int = 0
    deleted: int = 0
    resets: int = 0                # tracks restarted because the class changed


class MultiTracker:
    """Predict every track, gate, associate, update, manage the rest."""

    def __init__(self, association: str = "gnn", nn_order: str = "ascending",
                 gate: float = GATE_CHI2, confirm_m: int = 3, confirm_n: int = 5,
                 max_coast_time: float = 1.0, init_vel_sigma: float = 5.0,
                 accel_sigma: float = 3.0, reset_on_class_change: bool = False,
                 cost: str = "epsilon", confirmed_first: bool = True) -> None:
        if association not in ("gnn", "nn"):
            raise ValueError("association must be 'gnn' or 'nn'")
        if cost not in ("epsilon", "likelihood"):
            raise ValueError("cost must be 'epsilon' or 'likelihood'")
        self.cost = cost
        self.confirmed_first = confirmed_first
        self.association = association
        self.nn_order = nn_order
        self.gate = gate
        self.m, self.n = confirm_m, confirm_n
        self.max_coast_time = max_coast_time
        self.init_vel_sigma = init_vel_sigma
        self.accel_sigma = accel_sigma
        self.reset_on_class_change = reset_on_class_change
        self.tracks: list[Track] = []
        self.next_id = 1
        self.t = None

    # ---------------------------------------------------------------- steps
    def _start(self, det: Detection, t: float) -> Track:
        trk = Track(id=self.next_id,
                    kf=KalmanCV(det.z, det.R, self.init_vel_sigma, self.accel_sigma),
                    label=det.label, hits=deque([1], maxlen=self.n), detections=1,
                    last_seen=t)
        self.next_id += 1
        self.tracks.append(trk)
        return trk

    def _associate(self, eps, cost, rows: list[int], cols: list[int]) -> dict[int, int]:
        """NN or GNN on the sub-matrix rows x cols; returns full-matrix indices."""
        if not rows or not cols:
            return {}
        e, c = eps[np.ix_(rows, cols)], cost[np.ix_(rows, cols)]
        if self.association == "gnn":
            sub = associate_gnn(e, self.gate, c)
        else:
            by_id = sorted(range(len(rows)), key=lambda k: self.tracks[rows[k]].id)
            order = by_id if self.nn_order == "ascending" else by_id[::-1]
            sub = associate_nn(e, order, self.gate, c)
        return {rows[r]: cols[d] for r, d in sub.items()}

    def step(self, detections: list[Detection], t: float) -> StepReport:
        """One frame: everything on the slide 'Every frame, the same loop'."""
        rep = StepReport(detections=len(detections))
        dt = 0.0 if self.t is None else max(0.0, t - self.t)
        self.t = t

        # 1. Predict every track to this instant.
        for trk in self.tracks:
            trk.kf.predict(dt)
            trk.age += 1

        # 2. Associate: epsilon for every pair, then NN or GNN inside the gate.
        # The gate always uses epsilon. The cost that picks among the pairs
        # inside the gate is epsilon alone, or epsilon + ln|S| ('likelihood',
        # see the README): with epsilon alone, a new track, whose S is large,
        # looks close to everything and takes detections from older tracks.
        eps = np.full((len(self.tracks), len(detections)), np.inf)
        cost = np.full_like(eps, np.inf)
        for i, trk in enumerate(self.tracks):
            for j, det in enumerate(detections):
                e, logdet = trk.kf.epsilon_and_logdet(det.z, det.R)
                eps[i, j] = e
                cost[i, j] = e + logdet if self.cost == "likelihood" else e
        if self.confirmed_first:
            # Two rounds: confirmed and coasting tracks first, then tentative
            # tracks on the detections left over. Without it, a new track
            # (large S) can take the detection of an older one.
            old = [i for i, trk in enumerate(self.tracks) if trk.status != TENTATIVE]
            new = [i for i, trk in enumerate(self.tracks) if trk.status == TENTATIVE]
            pairs = self._associate(eps, cost, old, list(range(len(detections))))
            left = [j for j in range(len(detections)) if j not in pairs.values()]
            pairs.update(self._associate(eps, cost, new, left))
        else:
            pairs = self._associate(eps, cost, list(range(len(self.tracks))),
                                    list(range(len(detections))))

        # 3. Update the matched tracks.
        used = set(pairs.values())
        restart: list[Detection] = []
        for i, trk in enumerate(self.tracks):
            j = pairs.get(i)
            if j is None:
                trk.hits.append(0)
                continue
            det = detections[j]
            if self.reset_on_class_change and det.label != trk.label:
                # Tempe: a new class is a new object. The old track, and its
                # history, are thrown away.
                trk.status = DELETED
                restart.append(det)
                rep.resets += 1
                continue
            trk.kf.update(det.z, det.R)
            trk.hits.append(1)
            trk.detections += 1
            trk.last_seen = t
            rep.matched += 1

        # 4. Manage the rest: the lifecycle.
        for trk in self.tracks:
            if trk.status == DELETED:
                continue
            detected = trk.hits[-1] == 1
            if trk.status == TENTATIVE:
                if sum(trk.hits) >= self.m:
                    trk.status = CONFIRMED
                elif len(trk.hits) == self.n or sum(trk.hits) + (self.n - len(trk.hits)) < self.m:
                    trk.status = DELETED        # can no longer pass M of N
            elif trk.status == CONFIRMED and not detected:
                trk.status = COASTING
            elif trk.status == COASTING:
                if detected:
                    trk.status = CONFIRMED
                elif t - trk.last_seen > self.max_coast_time:
                    trk.status = DELETED
        rep.deleted = sum(1 for trk in self.tracks if trk.status == DELETED)
        self.tracks = [trk for trk in self.tracks if trk.status != DELETED]

        # Detections that matched no track start a tentative one.
        for j, det in enumerate(detections):
            if j not in used:
                self._start(det, t)
                rep.started += 1
        for det in restart:
            self._start(det, t)
            rep.started += 1
        return rep

    def counts(self) -> dict[str, int]:
        out = {TENTATIVE: 0, CONFIRMED: 0, COASTING: 0}
        for trk in self.tracks:
            out[trk.status] += 1
        return out
