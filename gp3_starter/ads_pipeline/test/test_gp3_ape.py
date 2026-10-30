"""GP3 Task 2 evaluation: APE.

    cd ads_pipeline && python3 -m pytest test/test_gp3_ape.py -v
"""
import numpy as np
import pytest

from tools.ape import ape_rmse, match_by_stamp


def test_rmse_of_constant_offset():
    pairs = [(t, np.array([t, 1.0]), np.array([t, 0.0])) for t in range(5)]
    assert ape_rmse(pairs) == pytest.approx(1.0)


def test_window_selects_the_outage_only():
    pairs = [(t, np.array([0.0, 3.0 if 2 <= t < 4 else 0.0]), np.zeros(2))
             for t in range(6)]
    assert ape_rmse(pairs, window=(2, 4)) == pytest.approx(3.0)


def test_match_skips_estimates_without_a_reference_close_in_time():
    est = [(0.00, np.zeros(2)), (0.05, np.zeros(2)), (0.30, np.zeros(2))]
    ref = [(0.00, np.zeros(2)), (0.05, np.zeros(2))]
    assert len(match_by_stamp(est, ref)) == 2


def test_nis_verdict_reads_the_direction_from_the_mean():
    from tools.ape import nis_verdict
    assert nis_verdict([2.0] * 100)[0] == 'consistent'
    assert nis_verdict([20.0] * 100)[0] == 'overconfident'
    assert nis_verdict([0.01] * 100)[0] == 'underconfident'


def test_outside_the_window_is_everything_else():
    pairs = [(t, np.array([0.0, 3.0 if 2 <= t < 4 else 1.0]), np.zeros(2))
             for t in range(6)]
    assert ape_rmse(pairs, window=(2, 4), outside=True) == pytest.approx(1.0)
