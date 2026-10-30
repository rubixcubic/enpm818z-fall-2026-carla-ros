"""GP3 ground truth helper: conversions that need no CARLA.

    cd ads_pipeline && python3 -m pytest test/test_gp3_ground_truth.py -v
"""
import math

import pytest

from ads_pipeline import ground_truth as G


def test_carla_to_map_flips_y():
    assert G.carla_to_map_xy(3.0, 4.0) == (3.0, -4.0)


def test_yaw_flips_sign_and_turns_to_radians():
    assert G.carla_to_map_yaw(90.0) == pytest.approx(-math.pi / 2)


def test_box_detection_carries_id_class_and_center():
    d = G.box_detection(42, 'person', (1.0, 2.0, 0.9), (0.5, 0.5, 1.8), 0.3)
    assert d.id == '42'
    assert d.results[0].hypothesis.class_id == 'person'
    assert d.bbox.center.position.x == 1.0 and d.bbox.center.position.y == 2.0
    assert d.bbox.size.z == 1.8


def test_time_is_up_after_max_sim_seconds():
    from ads_pipeline.sensor_manager import time_is_up
    assert not time_is_up(105.0, 10.0, 0.0)      # 0 means run until Ctrl+C
    assert not time_is_up(99.9, 10.0, 90.0)
    assert time_is_up(100.0, 10.0, 90.0)


def test_ground_truth_classes_are_gp2s_classes():
    from ads_pipeline.detector_node import COLORS
    assert set(G.CLASS_OF.values()) == set(COLORS)
