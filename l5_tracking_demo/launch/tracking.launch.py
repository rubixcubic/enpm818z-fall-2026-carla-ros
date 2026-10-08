"""Bring up a detector, the tracker and RViz on top of the L2 bridge.

    ros2 launch l2_carla_demo demo.launch.py rviz:=false      # terminal 1, first
    ros2 run l5_tracking_demo spawn_traffic                    # terminal 2: traffic
    ros2 launch l5_tracking_demo tracking.launch.py            # terminal 3

    ros2 launch l5_tracking_demo tracking.launch.py source:=truth association:=nn
    ros2 launch l5_tracking_demo tracking.launch.py source:=truth flip_prob:=0.3 \
        reset_on_class_change:=true evaluate:=true
    ros2 launch l5_tracking_demo tracking.launch.py source:=boxes   # needs l5_box_demo

The bridge owns the simulation clock and the AV. Nothing here ticks.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import EqualsSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("l5_tracking_demo")
    params = os.path.join(share, "config", "tracker.yaml")
    rviz_config = os.path.join(share, "rviz", "l5_tracking.rviz")
    arg = LaunchConfiguration

    # source:=boxes runs l5_box_demo's box_detector, which publishes the fitted
    # box centers on /l5/detections instead of the cluster centroids.
    try:
        box_params = [os.path.join(get_package_share_directory("l5_box_demo"),
                                   "config", "boxes.yaml"),
                      {"output_topic": "/l5/detections"}]
    except Exception:            # l5_box_demo not built: source:=boxes unavailable
        box_params = None

    def node(executable, extra=None, condition=None):
        return Node(package="l5_tracking_demo", executable=executable, name=executable,
                    output="screen", emulate_tty=True,
                    parameters=[params] + ([extra] if extra else []), condition=condition)

    return LaunchDescription([
        DeclareLaunchArgument("source", default_value="lidar",
                              description="lidar: cluster centroids from the bridge's LiDAR; "
                                          "boxes: L-shape box centers (l5_box_demo); "
                                          "truth: CARLA's objects with noise added"),
        DeclareLaunchArgument("association", default_value="gnn", description="gnn or nn"),
        DeclareLaunchArgument("nn_order", default_value="ascending"),
        DeclareLaunchArgument("cost", default_value="epsilon",
                              description="epsilon, or likelihood = epsilon + ln|S|"),
        DeclareLaunchArgument("confirmed_first", default_value="true",
                              description="confirmed tracks choose before tentative ones"),
        DeclareLaunchArgument("reset_on_class_change", default_value="false"),
        DeclareLaunchArgument("flip_prob", default_value="0.0",
                              description="truth source only: chance of a wrong class"),
        DeclareLaunchArgument("evaluate", default_value="false",
                              description="grade detections and tracks against CARLA"),
        DeclareLaunchArgument("out", default_value="", description="evaluate: JSON file"),
        DeclareLaunchArgument("rviz", default_value="true"),
        node("detector", condition=IfCondition(EqualsSubstitution(arg("source"), "lidar"))),
        *([Node(package="l5_box_demo", executable="box_detector", name="box_detector",
                output="screen", emulate_tty=True, parameters=box_params,
                condition=IfCondition(EqualsSubstitution(arg("source"), "boxes")))]
          if box_params else []),
        node("truth_detector",
             {"flip_prob": ParameterValue(arg("flip_prob"), value_type=float)},
             condition=IfCondition(EqualsSubstitution(arg("source"), "truth"))),
        node("tracker", {"association": arg("association"), "nn_order": arg("nn_order"),
                         "cost": arg("cost"),
                         "confirmed_first": ParameterValue(arg("confirmed_first"), value_type=bool),
                         "reset_on_class_change": ParameterValue(
                             arg("reset_on_class_change"), value_type=bool)}),
        node("evaluate", {"out": arg("out")}, condition=IfCondition(arg("evaluate"))),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             arguments=["-d", rviz_config], condition=IfCondition(arg("rviz"))),
    ])
