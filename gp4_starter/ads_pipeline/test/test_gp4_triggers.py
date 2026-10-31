"""GP4 Task 2 tests: the two re-planning triggers and the hysteresis.

Distances as in the lecture's worked example: waypoints every 2 m on a
straight path, obstacle_clearance 3.0 m, max_deviation 2.0 m.
"""
import numpy as np

from ads_pipeline.planner_core import Hysteresis, obstacle_on_path, off_path

PATH = np.c_[np.arange(0.0, 31.0, 2.0), np.zeros(16)]       # x 0 to 30 m


def test_object_2_m_beside_the_path_fires_trigger_1():
    assert obstacle_on_path(PATH, [(20.0, 2.0)], 3.0)


def test_object_3_5_m_beside_the_path_does_not():
    assert not obstacle_on_path(PATH, [(20.0, 3.5)], 3.0)


def test_no_object_no_trigger():
    assert not obstacle_on_path(PATH, [], 3.0)


def test_car_2_5_m_from_the_path_fires_trigger_2():
    assert off_path(PATH, (10.0, 2.5), 2.0)


def test_car_1_5_m_from_the_path_does_not():
    assert not off_path(PATH, (10.0, 1.5), 2.0)


def test_hysteresis_fires_once_then_waits_for_the_trigger_to_clear():
    h = Hysteresis(clear_s=1.0)
    assert h.should_replan(True, 0.0)
    assert not h.should_replan(True, 0.5)        # still triggered: no new search
    assert not h.should_replan(False, 1.0)       # clear from t = 1.0 ...
    assert not h.should_replan(True, 1.5)        # ... but not for 1.0 s yet
    assert not h.should_replan(False, 2.0)
    assert not h.should_replan(False, 3.0)       # clear for 1.0 s now
    assert h.should_replan(True, 3.5)


def test_path_message_points_along_the_route():
    import math
    from std_msgs.msg import Header
    from ads_pipeline.planner_node import to_path_msg
    msg = to_path_msg(np.array([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]]), Header(frame_id='map'))
    assert len(msg.poses) == 3 and msg.header.frame_id == 'map'
    q = msg.poses[0].pose.orientation
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y ** 2 + q.z ** 2))
    assert abs(yaw - math.pi / 2) < 1e-9          # the route heads along +y


def test_only_the_path_ahead_of_the_car_counts():
    from ads_pipeline.planner_node import path_ahead
    ahead = path_ahead(PATH, (10.2, 0.1))
    assert ahead[0].tolist() == [10.0, 0.0] and len(ahead) == 11


def test_vehicles_queued_at_a_light_are_not_parked_cars():
    from ads_pipeline.planner_node import away_from_lights
    stops = np.array([[100.0, 0.0]])
    objs = [(80.0, 0.0), (40.0, 0.0)]          # 20 m and 60 m before the stop line
    assert away_from_lights(objs, stops, 30.0) == [(40.0, 0.0)]
