"""GP4 evaluation: route completion, red-light violations, collisions."""
import numpy as np
import pytest

from tools.evaluate_run import collision_count, red_light_violations

ROUTE = np.c_[np.arange(0.0, 101.0, 2.0), np.zeros(51)]       # 100 m straight


def test_red_light_violation_is_leaving_the_light_on_red_while_moving():
    s = [(0.0, 'Red', True, 6.0), (0.1, 'Red', False, 6.0),      # drove through: 1
         (5.0, 'Red', True, 0.0), (9.0, 'Green', True, 0.0),     # waited: 0
         (9.5, 'Green', False, 2.0)]
    assert red_light_violations(s) == 1


def test_one_collision_counted_once():
    e = [(10.0, 'vehicle.audi.tt'), (10.2, 'vehicle.audi.tt'), (30.0, 'static.prop')]
    assert collision_count(e) == 2


def test_run_is_scored_until_the_car_reaches_the_goal():
    from tools.evaluate_run import arrival_index
    driven = np.c_[np.arange(0.0, 105.0, 1.0), np.zeros(105)]
    assert arrival_index(ROUTE, driven) == 99          # first point within 2 m of (100, 0)
    assert arrival_index(ROUTE, driven[:50]) == 49     # never arrived: the whole run


def test_completion_counts_a_detour_by_the_distance_left():
    from tools.evaluate_run import completion_by_distance_left
    assert completion_by_distance_left(arrived=True, left_m=50.0, route_m=500.0) == 1.0
    assert completion_by_distance_left(False, 100.0, 500.0) == pytest.approx(0.8)
    assert completion_by_distance_left(False, 700.0, 500.0) == 0.0     # farther than at the start
