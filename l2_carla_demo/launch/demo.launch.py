"""Launch the L2 CARLA demonstration.

    ros2 launch l2_carla_demo demo.launch.py
    ros2 launch l2_carla_demo demo.launch.py town:=Town03 rviz:=false
    ros2 launch l2_carla_demo demo.launch.py gnss_noise_m:=1.5    # L3 EKF hands-on

The CARLA server must already be running. Give it 30-60 s to load the level
before launching this, or the client times out.
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
    share = get_package_share_directory("l2_carla_demo")
    params = os.path.join(share, "config", "sensors.yaml")
    rviz_config = os.path.join(share, "rviz", "l2_demo.rviz")

    arguments = [
        DeclareLaunchArgument("host", default_value="localhost"),
        DeclareLaunchArgument("port", default_value="2000"),
        DeclareLaunchArgument(
            "town", default_value="",
            description="map to load; empty keeps whatever is already loaded"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument(
            "rate_report", default_value="true",
            description="print the delivered rate of every topic"),
        DeclareLaunchArgument(
            "gnss_noise_m", default_value="0.0",
            description="GNSS noise, standard deviation per axis in meters; "
                        "0 keeps CARLA's default"),
    ]

    bridge = Node(
        package="l2_carla_demo",
        executable="carla_bridge",
        name="carla_bridge",
        output="screen",
        emulate_tty=True,
        parameters=[
            params,
            {"host": LaunchConfiguration("host"),
             "port": LaunchConfiguration("port"),
             "town": LaunchConfiguration("town"),
             "gnss_noise_m": ParameterValue(LaunchConfiguration("gnss_noise_m"),
                                            value_type=float)},
        ],
    )

    rate_report = Node(
        package="l2_carla_demo",
        executable="rate_report",
        name="rate_report",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration("rate_report")),
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(arguments + [bridge, rate_report, rviz])
