"""GP4 evaluation: route completion, red-light violations, collisions."""
import numpy as np
import pytest

from tools.evaluate_run import collision_count, red_light_violations

ROUTE = np.c_[np.arange(0.0, 101.0, 2.0), np.zeros(51)]       # 100 m straight


def test_red_light_violation_is_crossing_the_stop_line_on_red_while_moving():
    # samples: (t, state, signed distance to the stop line in m, speed in m/s)
    ran = [(0.0, 'Red', 0.22, 6.4), (0.05, 'Red', -0.08, 5.5)]       # video8 at 45.8 s
    waited = [(5.0, 'Red', 0.66, 0.0), (9.0, 'Green', 0.66, 0.0), (9.5, 'Green', -0.2, 2.0)]
    crept = [(0.0, 'Red', 0.1, 0.5), (0.1, 'Red', -0.1, 0.5)]        # 0.5 m/s: not counted
    assert red_light_violations(ran) == 1
    assert red_light_violations(waited) == 0
    assert red_light_violations(crept) == 0
    assert red_light_violations(ran + waited) == 1


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


def test_a_pedestrian_inside_the_moving_car_is_one_hit():
    from tools.evaluate_run import pedestrian_hits
    # The car at the origin heading +x at 6 m/s; a person walks across its
    # nose and through its outline over three frames: one hit, not three.
    frames = [((0.0, 0.0), 0.0, 6.0, [('p1', (2.0, 0.5))]),
              ((0.6, 0.0), 0.0, 6.0, [('p1', (2.0, 0.0))]),
              ((1.2, 0.0), 0.0, 6.0, [('p1', (2.0, -0.5))])]
    assert pedestrian_hits(frames) == [('p1', 6.0)]


def test_people_beside_or_walking_into_a_stopped_car_are_not_hits():
    from tools.evaluate_run import pedestrian_hits
    beside = [((0.0, 0.0), 0.0, 6.0, [('p1', (0.0, 2.0))])]      # 2 m to the left
    stopped = [((0.0, 0.0), 0.0, 0.2, [('p2', (1.0, 0.0))])]     # the car stands still
    assert pedestrian_hits(beside) == [] and pedestrian_hits(stopped) == []


def test_light_messages_take_the_clock_and_speed_of_the_car_message_before_them():
    from tools.evaluate_run import lights_on_sim_clock
    # (recording time, kind, payload): a String has no stamp, so each light
    # message takes the simulated time and speed of the car message before it.
    records = [(100.0, 'ego', (45.75, 6.4)), (100.01, 'light', 'Red 1 0.22'),
               (100.5, 'ego', (45.80, 5.5)), (100.51, 'light', 'Red 1 -0.08')]
    assert lights_on_sim_clock(records) == [(45.75, 'Red', 0.22, 6.4), (45.80, 'Red', -0.08, 5.5)]
