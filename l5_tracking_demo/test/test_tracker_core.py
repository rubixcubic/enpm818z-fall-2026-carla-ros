"""Checks for tracker_core, including the L5 exercise 'two tracks, three detections'.

    cd l5_tracking_demo && python3 -m pytest test/test_tracker_core.py -q
"""

import numpy as np

from l5_tracking_demo.tracker_core import (
    CONFIRMED, COASTING, TENTATIVE, Detection, KalmanCV, MultiTracker,
    associate_gnn, associate_nn)

# The exercise's table: rows track B, track D; columns detections 1, 2, 3.
EPS = np.array([[2.0, 1.0, 12.0],
                [14.0, 0.5, 6.0]])
B, D = 0, 1


def total(pairs):
    return sum(EPS[t, d] for t, d in pairs.items())


def test_nn_order_changes_the_answer():
    b_first = associate_nn(EPS, [B, D])
    d_first = associate_nn(EPS, [D, B])
    assert b_first == {B: 1, D: 2} and total(b_first) == 7.0
    assert d_first == {D: 1, B: 0} and total(d_first) == 2.5


def test_gnn_picks_the_lowest_total():
    pairs = associate_gnn(EPS)
    assert pairs == {B: 0, D: 1} and total(pairs) == 2.5


def test_epsilon_matches_the_slide():
    # 'Distance in units of the track's uncertainty': S = sigma^2 I, d = 2 m.
    for sigma, expected in ((0.1, 400.0), (3.0, 4.0 / 9.0)):
        kf = KalmanCV(np.zeros(2), np.zeros((2, 2)), 0.0, 0.0)
        kf.P = np.diag([sigma ** 2, sigma ** 2, 0.0, 0.0])
        assert np.isclose(kf.epsilon(np.array([2.0, 0.0]), np.zeros((2, 2))), expected)


def test_constant_velocity_predict_matches_the_slide():
    # Car B: (12.0, -2.0), 4.0 m/s along 30 deg, dt = 0.1 s -> (12.35, -1.80).
    kf = KalmanCV(np.array([12.0, -2.0]), np.eye(2), 1.0, 0.0)
    kf.x[2:] = 4.0 * np.cos(np.radians(30)), 4.0 * np.sin(np.radians(30))
    kf.predict(0.1)
    assert np.allclose(kf.x[:2], [12.35, -1.80], atol=0.005)


def test_lifecycle_confirm_coast_delete():
    trk = MultiTracker(confirm_m=3, confirm_n=5, max_coast_time=1.0)
    R = 0.25 * np.eye(2)
    det = lambda x: [Detection(np.array([x, 0.0]), R)]
    t = 0.0
    trk.step(det(10.0), t)
    assert trk.tracks[0].status == TENTATIVE
    for k in range(1, 3):
        t += 0.1
        trk.step(det(10.0 + 0.5 * k), t)
    assert trk.tracks[0].status == CONFIRMED          # 3 detections in 3 frames
    t += 0.1
    trk.step([], t)
    assert trk.tracks[0].status == COASTING
    for _ in range(11):                                 # 1.1 s with nothing
        t += 0.1
        trk.step([], t)
    assert trk.tracks == []


def test_reset_on_class_change_restarts_the_track():
    trk = MultiTracker(reset_on_class_change=True)
    R = 0.25 * np.eye(2)
    trk.step([Detection(np.array([10.0, 0.0]), R, "vehicle")], 0.0)
    first = trk.tracks[0].id
    trk.step([Detection(np.array([10.5, 0.0]), R, "bicycle")], 0.1)
    assert [t.id for t in trk.tracks] == [first + 1]
    assert np.allclose(trk.tracks[0].kf.x[2:], 0.0)   # no velocity: no history


def test_confirmed_tracks_choose_first():
    # An old, sure track at x = 0 and a new, unsure one at x = 1.0 m. One
    # detection at x = 0.6 m. On epsilon alone, the unsure track looks closer
    # (its S is large); with confirmed_first, the old track keeps its object.
    R = 0.09 * np.eye(2)
    for first, winner in ((False, "new"), (True, "old")):
        trk = MultiTracker(association="gnn", confirmed_first=first)
        trk.t = 0.0
        old = trk._start(Detection(np.array([0.0, 0.0]), R), 0.0)
        old.status = "confirmed"
        old.kf.P = np.diag([0.04, 0.04, 0.01, 0.01])
        new = trk._start(Detection(np.array([1.0, 0.0]), R), 0.0)
        new.kf.P = np.diag([4.0, 4.0, 25.0, 25.0])
        trk.step([Detection(np.array([0.6, 0.0]), R)], 0.0)
        got = old if old.detections == 2 else new
        assert (got is old) == (winner == "old")
