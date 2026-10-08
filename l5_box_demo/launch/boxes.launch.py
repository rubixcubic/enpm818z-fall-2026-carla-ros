"""3D boxes from the LiDAR, on top of the L2 bridge.

    ros2 launch l2_carla_demo demo.launch.py rviz:=false      # terminal 1, first
    ros2 run l5_tracking_demo spawn_traffic                    # terminal 2: traffic
    ros2 launch l5_box_demo boxes.launch.py evaluate:=true     # terminal 3

The bridge owns the simulation clock and the AV. Nothing here ticks.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("l5_box_demo")
    arg = LaunchConfiguration
    return LaunchDescription([
        DeclareLaunchArgument("evaluate", default_value="false",
                              description="grade the boxes against CARLA's true boxes"),
        DeclareLaunchArgument("out", default_value="", description="evaluate: JSON file"),
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(package="l5_box_demo", executable="box_detector", name="box_detector",
             output="screen", emulate_tty=True,
             parameters=[os.path.join(share, "config", "boxes.yaml"),
                         {"evaluate": ParameterValue(arg("evaluate"), value_type=bool),
                          "out": arg("out")}]),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             arguments=["-d", os.path.join(share, "rviz", "l5_boxes.rviz")],
             condition=IfCondition(arg("rviz"))),
    ])
