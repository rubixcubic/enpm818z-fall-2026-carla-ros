"""L10: the controller's five input topics, live from CARLA.

    ros2 launch l10_topics_demo l10_topics.launch.py
    ros2 launch l10_topics_demo l10_topics.launch.py town:=Town03 port:=2000

The CARLA server must already be running. Give it 30 to 60 s to load the level
before launching this, or the client times out. Then, in a second terminal:

    ros2 topic list
    ros2 topic echo /behavior/target_speed --once

The L2 bridge drives the AV on CARLA's autopilot and owns the clock (RViz off).
Its IMU topic is renamed to /carla/imu/data, the name the controller reads.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    l2_launch = os.path.join(get_package_share_directory("l2_carla_demo"),
                             "launch", "demo.launch.py")
    arg = LaunchConfiguration

    return LaunchDescription([
        DeclareLaunchArgument("host", default_value="localhost"),
        DeclareLaunchArgument("port", default_value="2000", description="CARLA's port"),
        DeclareLaunchArgument("town", default_value="Town03",
                              description="map to load; empty keeps the loaded one"),

        GroupAction(scoped=True, actions=[
            SetRemap(src="/carla/ego_vehicle/imu", dst="/carla/imu/data"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(l2_launch),
                launch_arguments={"host": arg("host"), "port": arg("port"),
                                  "town": arg("town"), "rviz": "false",
                                  "rate_report": "false"}.items()),
        ]),

        Node(package="l10_topics_demo", executable="l10_topics", name="l10_topics",
             output="screen", emulate_tty=True,
             parameters=[{
                 "host": arg("host"),
                 "port": ParameterValue(arg("port"), value_type=int),
             }]),
    ])
