"""GP3 Task 4 evaluation: ID switches, MOTA and speed error on the map."""
import numpy as np
import pytest

from tools.track_eval import count_on_map


def tr(tid, x, cls='pedestrian', v=(1.0, 0.0)):
    return {'track_id': tid, 'cls': cls, 'xy': np.array([x, 0.0]), 'v': np.array(v)}


def gt(aid, x, cls='pedestrian', v=(1.0, 0.0)):
    return {'id': aid, 'cls': cls, 'xy': np.array([x, 0.0]), 'v': np.array(v)}


def test_one_switch_when_the_track_id_changes():
    frames = [([tr(1, 0.0)], [gt(7, 0.0)]),
              ([tr(1, 1.0)], [gt(7, 1.0)]),
              ([tr(2, 2.0)], [gt(7, 2.0)])]
    out = count_on_map(frames, match_radius=2.0)
    assert out['id_switches'] == 1 and out['matched'] == 3
    assert out['mota'] == pytest.approx(1 - 1 / 3)


def test_speed_error_is_the_velocity_difference():
    frames = [([tr(1, 0.0, v=(1.5, 0.0))], [gt(7, 0.0, v=(1.0, 0.0))])]
    assert count_on_map(frames, 2.0)['mean_speed_error'] == pytest.approx(0.5)


def test_far_track_is_a_false_track_and_a_miss():
    frames = [([tr(1, 10.0)], [gt(7, 0.0)])]
    out = count_on_map(frames, 2.0)
    assert out['missed'] == 1 and out['false_tracks'] == 1
