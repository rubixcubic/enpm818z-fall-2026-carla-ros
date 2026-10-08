import math

import numpy as np
import pytest

from l5_box_demo.box_eval import BoxScore, greedy_pairs, scale_iou


def test_scale_iou_matches_the_devkit_definition():
    assert scale_iou((4.5, 1.9, 1.5), (4.5, 1.9, 1.5)) == pytest.approx(1.0)
    # half the length: intersection 2.25*1.9*1.5, union 4.5*1.9*1.5
    assert scale_iou((4.5, 1.9, 1.5), (2.25, 1.9, 1.5)) == pytest.approx(0.5)


def test_greedy_pairs_closest_first():
    t = np.array([[0.0, 0.0], [10.0, 0.0]])
    d = np.array([[0.3, 0.0], [10.0, 3.0], [50.0, 0.0]])
    assert greedy_pairs(t, d, 2.0) == [(0, 0, pytest.approx(0.3))]
    assert len(greedy_pairs(t, d, 4.0)) == 2


def test_score_counts():
    s = BoxScore()
    truth = [{"x": 12.0, "y": -2.0, "yaw": math.radians(30), "size": (4.5, 1.9, 1.5),
              "visible": True}]
    boxes = [{"x": 11.93, "y": -2.0, "yaw": math.radians(29 + 180), "size": (4.41, 1.91, 1.5),
              "centroid": (10.99, -1.89)}]
    s.add(truth, boxes)
    out = s.summary()
    assert out["recall_visible"]["0.5"] == 1.0
    assert out["heading_error_deg_mod180"]["mean"] == pytest.approx(1.0, abs=1e-6)
    assert out["centroid_error_m"]["mean"] == pytest.approx(math.hypot(1.01, 0.11), abs=1e-3)
