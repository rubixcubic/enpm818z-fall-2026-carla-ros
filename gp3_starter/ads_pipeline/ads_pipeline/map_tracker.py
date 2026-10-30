"""GP3 Task 4: GP2's tracker, moved from the image onto the map.

The same steps as GP2's BoxTracker.step and the same lifecycle constants,
imported from your box_tracker.py. What changes: the state is the object's
map position and velocity, in meters and meters per second, and a detection
is matched to a track by the Mahalanobis distance of the innovation, not
by box overlap. The gate is 9.21, the chi-square 99% bound for 2 values.
"""
import numpy as np

from ads_pipeline.box_tracker import CONFIRM_HITS, MAX_MISSES, Track, associate

R_STD = 1.29    # m, fused position noise: 1.62 m mean error measured / sqrt(pi / 2)
Q_ACC = 1.0     # m/s^2, random acceleration, chosen for GP3
V_STD0 = 5.0    # m/s, initial velocity uncertainty


class MapKalman:
    """State x = [x, y, vx, vy] in the map frame; covariance P."""

    def __init__(self, xy):
        """YOURS. Start at the detection xy, at rest: x = [x, y, 0, 0]. P: R_STD^2 on the
        position, V_STD0^2 on the velocity. H picks the position; R = R_STD^2 I.
        """
        raise NotImplementedError('GP3 Task 4: MapKalman.__init__')

    def predict(self, dt):
        """YOURS. Constant velocity over dt seconds. Q comes from a
        random acceleration of Q_ACC through G = [0.5 dt^2 I; dt I].
        """
        raise NotImplementedError('GP3 Task 4: MapKalman.predict')

    def S(self):
        return self.H @ self.P @ self.H.T + self.R

    def update(self, xy):
        """YOURS. The Kalman update with a matched detection's xy: K = P H^T S^-1,
        x = x + K (z - H x), P = (I - K H) P.
        """
        raise NotImplementedError('GP3 Task 4: MapKalman.update')


def mahalanobis_cost(tracks, dets):
    """YOURS. Cost matrix, tracks x detections: nu^T S^-1 nu, with nu = z - H x and
    S the track's innovation covariance (MapKalman.S).
    """
    raise NotImplementedError('GP3 Task 4: mahalanobis_cost')


class MapTracker:
    """GP2's BoxTracker on the map. One class never matches another."""

    def __init__(self, gate=9.21):
        self.gate = gate
        self.tracks = []
        self.next_id = 1

    def step(self, dets, dt):
        """YOURS. GP2's BoxTracker.step, steps 1 to 5, on the map: predict every track
        by dt; per class, mahalanobis_cost and associate(cost, max_cost=self.gate);
        update matches, count misses, start tentative tracks; drop tracks at
        MAX_MISSES. Return the confirmed tracks matched this frame as
        [{'track_id', 'cls', 'xy', 'v', 'P'}].
        """
        raise NotImplementedError('GP3 Task 4: MapTracker.step')
