"""The L9 planning demo with three parked cars instead of one.

    ros2 launch l9_planning_demo demo_obstacles.launch.py

The cars, as (s, d) along the slides' stretch of road 67:
    s =  78 m, d = 3.5 m    in the left lane: the AV cannot move left until it is past
    s = 118 m, d = -0.3 m   the slides' parked car, in the right lane: the AV passes it
    s = 165 m, d = -0.3 m   in the right lane: the AV stays in the left lane
Every plan along the drive has a collision-free cheapest path (checked offline with the
node's planning loop). A layout can leave the planner with none: it has no speed choice
and no "back right, then left again" candidate, so it would pick a colliding path.

Assumes a CARLA 0.9.16 server is running (start it with carla_start). The node ticks
the server: do not run l2_carla_demo's bridge at the same time.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# [s1, d1, s2, d2, ...] in meters, floats (the parameter is a double array)
OBSTACLES = [78.0, 3.5, 118.0, -0.3, 165.0, -0.3]


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("l9_planning_demo")
    rviz_config = os.path.join(share, "rviz", "l9_planning.rviz")
    return LaunchDescription([
        DeclareLaunchArgument("host", default_value="localhost"),
        DeclareLaunchArgument("port", default_value="2000"),
        DeclareLaunchArgument("rviz", default_value="true",
                              description="show the candidates, the cars and the costs"),
        Node(package="l9_planning_demo", executable="planning_demo", name="planning_demo",
             output="screen",
             parameters=[{"host": LaunchConfiguration("host"),
                          "port": LaunchConfiguration("port"),
                          "obstacles": OBSTACLES}]),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             arguments=["-d", rviz_config],
             condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
