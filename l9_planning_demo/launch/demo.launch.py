"""Start the L9 planning demo and RViz.

    ros2 launch l9_planning_demo demo.launch.py
    ros2 launch l9_planning_demo demo.launch.py port:=2000 rviz:=false

Assumes a CARLA 0.9.16 server is running (it loads Town03 if another town is
loaded). The node ticks the server: do not run l2_carla_demo's bridge at the
same time.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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
                          "port": LaunchConfiguration("port")}]),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             arguments=["-d", rviz_config],
             condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
