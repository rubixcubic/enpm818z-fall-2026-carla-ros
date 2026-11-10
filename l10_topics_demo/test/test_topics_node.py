from l10_topics_demo.topics_node import heading_change, ros_yaw
import math


def test_heading_change_wraps():
    assert heading_change(170.0, -170.0) == 20.0
    assert heading_change(0.0, 90.0) == 90.0


def test_ros_yaw_flips_sign():
    assert math.isclose(ros_yaw(90.0), -math.pi / 2)
