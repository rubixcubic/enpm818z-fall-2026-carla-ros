"""GP3 Task 3 tests: frustum association and the map transform.

    cd ads_pipeline && python3 -m pytest test/test_gp3_fusion.py -v
"""
import math

import numpy as np
import pytest

from ads_pipeline import fusion_core as F
from ads_pipeline.lidar_projection import build_extrinsic, build_intrinsic_matrix


def scene():
    """A car's rear at 10 m (40 points) and a wall at 25 m (30 points),
    both inside one box."""
    rng = np.random.default_rng(0)
    car = np.c_[10 + rng.uniform(0, 0.3, 40), rng.uniform(-0.8, 0.8, 40),
                rng.uniform(0, 1.2, 40)]
    wall = np.c_[25 + rng.uniform(0, 0.2, 30), rng.uniform(-1, 1, 30),
                 rng.uniform(0, 1, 30)]
    pts = np.vstack([car, wall])
    uv = np.c_[np.full(len(pts), 500.0), np.full(len(pts), 300.0)]
    return pts, uv, pts[:, 0]


def test_keeps_the_nearest_cluster():
    pts, uv, depth = scene()
    c = F.associate_lidar_to_detection([450, 250, 550, 350], pts, uv, depth,
                                       1.0, 60.0, eps=0.7, min_pts=5)
    assert 10.0 < c[0] < 10.4


def test_no_points_in_the_box_gives_none():
    pts, uv, depth = scene()
    assert F.associate_lidar_to_detection([0, 0, 10, 10], pts, uv, depth,
                                          1.0, 60.0, eps=0.7, min_pts=5) is None


def test_range_window_drops_points_too_close():
    pts, uv, depth = scene()
    c = F.associate_lidar_to_detection([450, 250, 550, 350], pts, uv, depth,
                                       15.0, 60.0, eps=0.7, min_pts=5)
    assert c[0] > 24.0


def test_to_map_turns_with_the_car():
    T = np.eye(4)                       # LiDAR at the car's origin
    p = F.to_map(np.array([10.0, 0.0, 0.0]), T, (5.0, 5.0, math.pi / 2))
    assert p[:2] == pytest.approx((5.0, 15.0))


def test_projection_keeps_each_point_with_its_pixel():
    """A point 10 m ahead and 1 m to the LEFT (ROS y = +1) must land left of
    the image center: the cloud is in ROS axes, GP1's extrinsic in CARLA's."""
    K = build_intrinsic_matrix(1280, 720, 90.0)
    T = build_extrinsic((0.0, 0.0, 2.0), (0.0, 0.0, 2.0), 0.0)
    pts = np.array([[10.0, 1.0, 0.0], [-5.0, 0.0, 0.0]])   # second is behind
    kept, uv, depth = F.project_keep_points(pts, K, T, 1280, 720)
    assert kept.tolist() == [[10.0, 1.0, 0.0]]
    assert uv[0, 0] < 640 and depth[0] == pytest.approx(10.0)


def test_load_rig_reads_the_team_yaml(tmp_path):
    from ads_pipeline.fusion_node import load_rig
    cfg = tmp_path / 'rig.yaml'
    cfg.write_text('''sensor_manager:
  ros__parameters:
    camera: {width: 1280, height: 720, fov: 90.0}
    mounts:
      lidar_top: {x: 0.2, y: 0.3, z: 2.8, roll: 0.0, pitch: 0.0, yaw: 0.0}
      camera_rgb_front: {x: 1.5, y: 0.0, z: 1.6, roll: 0.0, pitch: -5.0, yaw: 0.0}
''')
    K, T_cam_lidar, T_ego_lidar, w, h = load_rig(str(cfg))
    assert (w, h) == (1280, 720) and K[0, 0] == pytest.approx(640.0)
    assert T_ego_lidar[:3, 3] == pytest.approx([0.2, -0.3, 2.8])   # ROS: y left
    assert T_cam_lidar == pytest.approx(
        build_extrinsic((0.2, 0.3, 2.8), (1.5, 0.0, 1.6), -5.0))


def test_ground_points_are_dropped_before_clustering():
    # LiDAR 2.8 m above the road: the road is at z = -2.8 in the LiDAR frame
    pts = np.array([[8.0, 0.0, -2.78], [8.0, 0.0, -2.0], [8.0, 0.0, -1.0]])
    kept = F.remove_ground(pts, lidar_height=2.8, clearance=0.3)
    assert kept[:, 2].tolist() == [-2.0, -1.0]


def test_a_detection_without_a_hypothesis_is_skipped():
    pytest.importorskip('vision_msgs')
    from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose
    from ads_pipeline.fusion_node import box_and_hypothesis
    d = Detection2D()
    d.bbox.center.position.x, d.bbox.center.position.y = 100.0, 50.0
    d.bbox.size_x, d.bbox.size_y = 20.0, 10.0
    assert box_and_hypothesis(d) is None
    h = ObjectHypothesisWithPose()
    h.hypothesis.class_id = 'vehicle'
    d.results.append(h)
    box, hyp = box_and_hypothesis(d)
    assert box == [90.0, 45.0, 110.0, 55.0] and hyp.class_id == 'vehicle'
