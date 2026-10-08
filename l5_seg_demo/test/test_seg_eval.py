"""IoU arithmetic, checked by hand on small grids."""

import math

import numpy as np
import pytest

from l5_seg_demo.classes import IGNORE
from l5_seg_demo.seg_eval import ConfusionMatrix


def test_one_frame_by_hand():
    # 2 x 4 pixels, classes 0 (road) and 13 (car).
    truth = np.array([[0, 0, 0, 0],
                      [13, 13, 0, 0]], dtype=np.uint8)
    pred = np.array([[0, 0, 0, 13],
                     [13, 0, 0, 0]], dtype=np.uint8)
    cm = ConfusionMatrix()
    cm.add(truth, pred)
    tp, fp, fn = cm.counts()
    # road: truth 6 pixels, predicted 6; 5 in both -> TP 5, FP 1, FN 1 -> 5 / 7
    assert (tp[0], fp[0], fn[0]) == (5, 1, 1)
    # car: truth 2, predicted 2, 1 in both -> TP 1, FP 1, FN 1 -> 1 / 3
    assert (tp[13], fp[13], fn[13]) == (1, 1, 1)
    iou = cm.iou()
    assert iou[0] == pytest.approx(5 / 7)
    assert iou[13] == pytest.approx(1 / 3)
    assert cm.miou() == pytest.approx((5 / 7 + 1 / 3) / 2)   # only classes in the truth
    assert cm.pixel_accuracy() == pytest.approx(6 / 8)
    assert math.isnan(iou[2])                                 # building: never in the truth


def test_ignored_pixels_do_not_count():
    truth = np.array([[0, IGNORE]], dtype=np.uint8)
    pred = np.array([[0, 13]], dtype=np.uint8)
    cm = ConfusionMatrix()
    cm.add(truth, pred)
    assert cm.iou()[0] == 1.0
    assert cm.m.sum() == 1


def test_counts_are_summed_over_frames_before_dividing():
    cm = ConfusionMatrix()
    # frame 1: 1 car pixel, missed.  frame 2: 99 car pixels, all found.
    cm.add(np.array([[13]], np.uint8), np.array([[0]], np.uint8))
    cm.add(np.full((1, 99), 13, np.uint8), np.full((1, 99), 13, np.uint8))
    # summed: TP 99, FN 1 -> 0.99 (a mean of per-frame IoUs would give 0.5)
    assert cm.iou()[13] == pytest.approx(0.99)
    assert cm.frames == 2


def test_report_lists_only_classes_in_the_truth():
    cm = ConfusionMatrix()
    cm.add(np.array([[0, 13]], np.uint8), np.array([[0, 2]], np.uint8))
    rep = cm.report()
    assert set(rep["classes"]) == {"road", "car"}
    assert rep["classes"]["car"]["iou"] == 0.0
    assert rep["miou"] == 0.5


def test_shapes_must_match():
    with pytest.raises(ValueError):
        ConfusionMatrix().add(np.zeros((2, 2), np.uint8), np.zeros((2, 3), np.uint8))


def test_report_says_what_the_network_called_each_class():
    cm = ConfusionMatrix()
    # 4 truck pixels (14): the network calls 3 of them car (13), 1 truck.
    cm.add(np.array([[14, 14, 14, 14]], np.uint8), np.array([[13, 13, 13, 14]], np.uint8))
    called = cm.report()["classes"]["truck"]["network_called_it"]
    assert called == {"car": 0.75, "truck": 0.25}
