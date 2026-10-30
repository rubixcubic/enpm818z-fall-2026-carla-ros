"""GP3 Task 3 evaluation: fused objects against the true boxes."""
import numpy as np
import pytest

from tools.fusion_eval import score_frames


def obj(x, y, cls='vehicle'):
    return {'xy': np.array([x, y]), 'cls': cls}


def test_one_match_and_one_miss():
    ego = np.array([0.0, 0.0])
    frames = [(ego, [obj(10.5, 0.0)], [obj(10.0, 0.0), obj(50.0, 0.0)])]
    s = score_frames(frames, match_radius=2.0)
    assert s['vehicle']['matched'] == 1 and s['vehicle']['gt'] == 2
    assert s['vehicle']['mean_error'] == pytest.approx(0.5)
    assert s['vehicle']['by_band']['0-20']['mean_error'] == pytest.approx(0.5)


def test_classes_never_match_each_other():
    ego = np.array([0.0, 0.0])
    frames = [(ego, [obj(10.0, 0.0, 'pedestrian')], [obj(10.0, 0.0)])]
    s = score_frames(frames, match_radius=2.0)
    assert s['vehicle']['matched'] == 0 and s['pedestrian']['false'] == 1


def test_only_objects_the_camera_can_see_count():
    from tools.fusion_eval import in_view
    ego, yaw = np.array([0.0, 0.0]), 0.0          # car at the origin, facing +x
    cam = (1.5, 0.0)                              # camera 1.5 m ahead of the origin
    assert in_view(np.array([20.0, 1.0]), ego, yaw, cam, 90.0, 60.0)
    assert not in_view(np.array([-10.0, 0.0]), ego, yaw, cam, 90.0, 60.0)   # behind
    assert not in_view(np.array([5.0, 10.0]), ego, yaw, cam, 90.0, 60.0)    # 63 deg left
    assert not in_view(np.array([70.0, 0.0]), ego, yaw, cam, 90.0, 60.0)    # too far


def test_points_in_a_turned_box():
    from tools.fusion_eval import points_in_box
    pts = np.array([[10.0, 0.0, 0.5], [10.0, 2.0, 0.5], [12.1, 0.0, 0.5], [10.0, 0.0, 3.0]])
    # a 4.6 m by 1.8 m car centered at (10, 0), turned 90 deg: long side along y
    n = points_in_box(pts, center=(10.0, 0.0, 0.75), size=(4.6, 1.8, 1.5), yaw=np.pi / 2)
    assert n == 2          # the first two; the third is 2.1 m to the side, the fourth above


def test_margin_counts_points_just_outside_a_thin_box():
    from tools.fusion_eval import points_in_box
    pts = np.array([[0.39, 0.0, 0.9]])            # 0.2 m outside a 0.38 m wide person
    box = dict(center=(0.0, 0.0, 0.93), size=(0.38, 0.38, 1.86), yaw=0.0)
    assert points_in_box(pts, **box) == 0
    assert points_in_box(pts, **box, margin=0.5) == 1
