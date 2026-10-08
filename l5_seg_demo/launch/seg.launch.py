"""Semantic segmentation of the front camera, graded against CARLA's truth.

    ros2 launch l2_carla_demo demo.launch.py rviz:=false      # terminal 1, first
    ros2 run l5_tracking_demo spawn_traffic                    # terminal 2: traffic
    ros2 launch l5_seg_demo seg.launch.py evaluate:=true       # terminal 3

    seg_node   the network on the bridge's camera   -> /l5/seg/labels, classes, overlay
    seg_truth  CARLA's semantic camera, same place  -> /l5/seg/truth_tags, truth
    seg_eval   (evaluate:=true) IoU per class and mIoU, logged every 10 s

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
    share = get_package_share_directory("l5_seg_demo")
    params = os.path.join(share, "config", "seg.yaml")
    arg = LaunchConfiguration
    return LaunchDescription([
        DeclareLaunchArgument("evaluate", default_value="false",
                              description="grade the network against CARLA's true classes"),
        DeclareLaunchArgument("out", default_value="", description="evaluate: JSON file"),
        DeclareLaunchArgument("truth", default_value="true",
                              description="attach CARLA's semantic camera (/l5/seg/truth)"),
        DeclareLaunchArgument("instances", default_value="false",
                              description="also run YOLOv8s-seg (/l5/seg/instances)"),
        DeclareLaunchArgument("rviz", default_value="true"),
        Node(package="l5_seg_demo", executable="seg_node", name="seg_node",
             output="screen", emulate_tty=True,
             parameters=[params,
                         {"instances": ParameterValue(arg("instances"), value_type=bool)}]),
        Node(package="l5_seg_demo", executable="seg_truth", name="seg_truth",
             output="screen", emulate_tty=True, parameters=[params],
             condition=IfCondition(arg("truth"))),
        Node(package="l5_seg_demo", executable="seg_eval", name="seg_eval",
             output="screen", emulate_tty=True,
             parameters=[params, {"out": arg("out")}],
             condition=IfCondition(arg("evaluate"))),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             arguments=["-d", os.path.join(share, "rviz", "l5_seg.rviz")],
             condition=IfCondition(arg("rviz"))),
    ])
