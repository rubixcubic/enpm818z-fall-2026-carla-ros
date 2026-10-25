"""The MarkerArrays, built from plain numbers (needs ROS 2 messages, not CARLA)."""

import pytest

rclpy = pytest.importorskip("rclpy")    # skip outside a sourced ROS 2 workspace

from std_msgs.msg import Header                                   # noqa: E402
from visualization_msgs.msg import Marker                         # noqa: E402

from l8_foxglove_demo import cv_predict as cvp                    # noqa: E402
from l8_foxglove_demo.markers import (Vehicle, prediction_markers,  # noqa: E402
                                      vehicle_markers)


def _cars():
    return [Vehicle(1, 0.0, 0.0, 0.8, 0.0, 8.0, 0.0, 4.7, 2.1, 1.5, is_ego=True),
            Vehicle(7, 30.0, 3.5, 0.8, 3.1416, -6.0, 0.0, 4.5, 2.0, 1.4)]


def test_vehicle_markers_clear_then_box_and_label_per_car():
    arr = vehicle_markers(_cars(), Header(frame_id="map"), 0.3)
    assert arr.markers[0].action == Marker.DELETEALL
    boxes = [m for m in arr.markers if m.ns == "vehicles"]
    labels = [m for m in arr.markers if m.ns == "speed"]
    assert len(boxes) == 2 and len(labels) == 2
    assert boxes[0].scale.x == pytest.approx(4.7)
    assert labels[0].text == "AV: 8.0 m/s"
    assert labels[1].text == "car 7: 6.0 m/s"
    assert boxes[0].color != boxes[1].color               # the AV stands out


def test_prediction_markers_line_and_one_dot_per_second():
    times = cvp.horizons(5.0, 0.1)
    paths = {c.actor_id: cvp.cv_path(c.x, c.y, c.vx, c.vy, times) for c in _cars()}
    arr = prediction_markers(paths, times, 1, 0.3, 1.0, Header(frame_id="map"), 0.3)
    lines = [m for m in arr.markers if m.ns == "cv_path"]
    dots = [m for m in arr.markers if m.ns == "cv_dots"]
    assert all(len(m.points) == 50 for m in lines)
    assert all(len(m.points) == 5 for m in dots)          # at 1, 2, 3, 4, 5 s
    assert dots[0].points[-1].x == pytest.approx(40.0)    # the AV, 8 m/s for 5 s
