"""GP4 Task 4 tests: the behavior state machine and what counts as ahead.

    cd ads_pipeline && python3 -m pytest test/test_gp4_behavior.py -v
"""
import math

import numpy as np

from ads_pipeline.behavior_core import BehaviorFSM, BehaviorState as S, nearest_ahead

PATH = np.c_[np.arange(0.0, 61.0, 2.0), np.zeros(31)]        # x 0 to 60 m


def fsm():
    return BehaviorFSM(target_speed=8.0, slow_speed=3.0, emergency_dist=8.0,
                       obstacle_dist=15.0, stop_dist=20.0, creep_speed=2.0)


def walker(x, y, vx, vy):
    return {'xy': np.array([x, y]), 'v': np.array([vx, vy])}


def test_pedestrian_walking_toward_the_path_counts_with_2_s_lookahead():
    # 6 m beside the path at x = 20, walking toward it at 1.5 m/s: 3.0 m in 2 s
    p = [walker(20.0, 6.0, 0.0, -1.5)]
    assert nearest_ahead(PATH, p, lookahead_s=2.0, clearance=3.0) == 20.0
    assert nearest_ahead(PATH, p, lookahead_s=1.0, clearance=3.0) == math.inf


def test_object_behind_the_car_does_not_count():
    p = [walker(-5.0, 0.0, 0.0, 0.0)]
    assert nearest_ahead(PATH, p, lookahead_s=2.0, clearance=3.0) == math.inf


def test_before_the_first_path_the_car_creeps():
    state, speed = fsm().update(math.inf, 'Red', 5.0, path_empty=True, path_seen=False)
    assert (state, speed) == (S.LANE_FOLLOW, 2.0)


def test_no_route_after_one_was_seen_stops_the_car():
    assert fsm().update(math.inf, 'Green', math.nan, True, True)[0] == S.EMERGENCY_STOP


def test_safer_state_wins():
    f = fsm()
    assert f.update(5.0, 'Red', 10.0, False, True)[0] == S.EMERGENCY_STOP
    assert f.update(12.0, 'Red', 10.0, False, True)[0] == S.STOP
    assert f.update(12.0, 'Green', math.nan, False, True)[0] == S.OBSTACLE_AVOIDANCE
    assert f.update(math.inf, 'Green', math.nan, False, True) == (S.LANE_FOLLOW, 8.0)


def test_red_light_ahead_stops_and_behind_does_not():
    assert fsm().update(math.inf, 'Red', 15.0, False, True)[0] == S.STOP
    assert fsm().update(math.inf, 'Red', -0.5, False, True)[0] == S.LANE_FOLLOW


def test_slow_down_for_an_object_in_range():
    assert fsm().update(12.0, 'Green', math.nan, False, True) == (S.OBSTACLE_AVOIDANCE, 3.0)


def test_goal_reached_holds_the_car():
    from ads_pipeline.behavior_node import goal_reached
    assert goal_reached(np.array([[0.0, 0.0], [2.0, 0.0]]), 3.0)
    assert not goal_reached(PATH, 3.0)


def test_goal_hold_is_inside_the_evaluators_arrival_radius():
    """The car must be able to come within 2.0 m of the goal (evaluate_run's
    arrival radius) before the behavior node holds it."""
    from ads_pipeline.behavior_node import goal_reached
    assert not goal_reached(np.array([[0.0, 0.0], [2.5, 0.0]]))
    assert goal_reached(np.array([[0.0, 0.0], [1.0, 0.0]]))


def test_one_waypoint_left_is_the_goal():
    from ads_pipeline.behavior_node import goal_reached
    assert goal_reached(np.array([[5.0, 5.0]]))
    assert not goal_reached(np.zeros((0, 2)))       # no path: the no-route rule decides
