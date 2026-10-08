"""Bring up ground truth, the logger and the filter together.

    ros2 launch l3_ekf_demo ekf.launch.py
    ros2 launch l3_ekf_demo ekf.launch.py rviz:=false

RViz opens by default showing the estimate against the truth with the
covariance ellipse drawn, and the front camera in its own panel. Pass rviz:=false if l2_carla_demo already has one
open, or if you are running headless.

Assumes l2_carla_demo's carla_bridge is already publishing the sensors. The
filter subscribes to those topics and never touches the simulator, which is
the point: it sees what a vehicle would see. Start the bridge with the GNSS
noise that config/ekf.yaml's R assumes:

    ros2 launch l2_carla_demo demo.launch.py rviz:=false gnss_noise_m:=1.5
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("l3_ekf_demo")
    params = os.path.join(share, "config", "ekf.yaml")
    rviz_config = os.path.join(share, "rviz", "l3_ekf.rviz")
    log_csv = LaunchConfiguration("log_csv")

    return LaunchDescription([
        DeclareLaunchArgument("log_csv", default_value="l3_log.csv",
                              description="Task 1 CSV of gnss, imu and truth"),
        DeclareLaunchArgument(
            "rviz", default_value="true",
            description="show the estimate, the truth and the covariance"),
        DeclareLaunchArgument(
            "role_name", default_value="ego",
            description="role_name of the vehicle to follow; l2_carla_demo uses ego"),
        Node(package="l3_ekf_demo", executable="truth", name="l3_truth",
             output="screen",
             parameters=[{"role_name": LaunchConfiguration("role_name")}]),
        Node(package="l3_ekf_demo", executable="logger", name="l3_logger",
             output="screen", parameters=[{"out": log_csv}]),
        Node(package="l3_ekf_demo", executable="ekf", name="l3_ekf",
             output="screen", parameters=[params]),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             arguments=["-d", rviz_config],
             condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
