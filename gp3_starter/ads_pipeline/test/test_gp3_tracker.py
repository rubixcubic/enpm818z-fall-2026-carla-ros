"""GP3 Task 4 tests: GP2's tracker, moved onto the map.

    cd ads_pipeline && python3 -m pytest test/test_gp3_tracker.py -v
"""
import numpy as np
import pytest

from ads_pipeline import map_tracker as M
from ads_pipeline.box_tracker import CONFIRM_HITS, MAX_MISSES

GATE = 9.21   # chi-square 99% bound for 2 values


def walker(k, v=1.5, dt=0.05):
    return {'xy': np.array([v * dt * k, 2.0]), 'cls': 'pedestrian'}


def test_filter_learns_the_speed():
    kf = M.MapKalman(walker(0)['xy'])
    for k in range(1, 40):
        kf.predict(0.05)
        kf.update(walker(k)['xy'])
    assert kf.x[2] == pytest.approx(1.5, abs=0.1)


def test_gate_rejects_a_far_detection():
    kf = M.MapKalman(np.zeros(2))
    C = M.mahalanobis_cost([kf], [np.array([30.0, 0.0])])
    assert C[0, 0] > GATE


def test_one_id_for_one_walker():
    tr = M.MapTracker(gate=GATE)
    ids = set()
    for k in range(30):
        out = tr.step([walker(k)], 0.05)
        ids |= {t['track_id'] for t in out}
    assert len(ids) == 1


def test_object_that_leaves_gets_a_new_id_on_return():
    tr = M.MapTracker(gate=GATE)
    for k in range(CONFIRM_HITS + 1):
        first = tr.step([walker(k)], 0.05)
    for _ in range(MAX_MISSES + 1):
        tr.step([], 0.05)
    for k in range(CONFIRM_HITS + 1):
        back = tr.step([walker(100 + k)], 0.05)
    assert back[0]['track_id'] != first[0]['track_id']


def test_output_carries_speed_and_covariance():
    tr = M.MapTracker(gate=GATE)
    for k in range(20):
        out = tr.step([walker(k)], 0.05)
    t = out[0]
    assert t['v'][0] == pytest.approx(1.5, abs=0.3) and t['P'].shape == (4, 4)


def test_tracks_become_an_ads_msgs_array():
    from std_msgs.msg import Header
    from ads_pipeline.tracker_node import to_tracked_array
    t = {'track_id': 4, 'cls': 'pedestrian', 'xy': np.array([1.0, 2.0]),
         'v': np.array([0.5, 0.0]), 'P': np.eye(4)}
    msg = to_tracked_array([t], Header(frame_id='map'))
    o = msg.objects[0]
    assert (o.id, o.class_id) == ('4', 'pedestrian')
    assert (o.position.x, o.velocity.x) == (1.0, 0.5)
    assert list(o.covariance) == list(np.eye(4).flatten())
