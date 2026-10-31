"""GP4 Task 1 tests: lock step and the command's steering sign.

    cd ads_pipeline && python3 -m pytest test/test_gp4_lockstep.py -v
"""
from ads_pipeline.sensor_manager import ready_to_tick, to_carla_control


def test_waits_for_the_answer_to_the_last_frame():
    assert ready_to_tick(10.00, 9.95, 0.1, 2.0) == (False, False)
    assert ready_to_tick(10.00, 10.00, 0.1, 2.0) == (True, False)


def test_ticks_anyway_after_the_timeout_and_says_so():
    assert ready_to_tick(10.00, 9.95, 2.5, 2.0) == (True, True)


def test_first_tick_needs_no_answer():
    assert ready_to_tick(None, None, 0.0, 2.0) == (True, False)


class Cmd:
    throttle, steer, brake = 0.4, 0.25, 0.0


def test_left_turn_is_negative_steer_in_carla():
    c = to_carla_control(Cmd())
    assert c == {'throttle': 0.4, 'steer': -0.25, 'brake': 0.0}

