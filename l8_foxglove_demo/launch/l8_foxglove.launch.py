"""The L8 Foxglove demo: the L2 bridge, foxglove_bridge and the L8 scene node.

    ros2 launch l8_foxglove_demo l8_foxglove.launch.py
    ros2 launch l8_foxglove_demo l8_foxglove.launch.py town:=Town01 vehicles:=15
    ros2 launch l8_foxglove_demo l8_foxglove.launch.py foxglove_port:=8766

The CARLA server must already be running. Give it 30 to 60 s to load the level
before launching this, or the client times out. Then, in Foxglove: Open
connection, Foxglove WebSocket, ws://localhost:8765, and import the layout
(share/l8_foxglove_demo/foxglove/l8_layout.json).

The bridge owns the clock and the AV (RViz off: Foxglove is the viewer here).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import (AnyLaunchDescriptionSource,
                                               PythonLaunchDescriptionSource)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("l8_foxglove_demo")
    params = os.path.join(share, "config", "l8_scene.yaml")
    l2_launch = os.path.join(get_package_share_directory("l2_carla_demo"),
                             "launch", "demo.launch.py")
    fox_launch = os.path.join(get_package_share_directory("foxglove_bridge"),
                              "launch", "foxglove_bridge_launch.xml")
    arg = LaunchConfiguration

    return LaunchDescription([
        DeclareLaunchArgument("host", default_value="localhost"),
        DeclareLaunchArgument("port", default_value="2000", description="CARLA's port"),
        DeclareLaunchArgument("town", default_value="",
                              description="map to load; empty keeps the loaded one"),
        DeclareLaunchArgument("vehicles", default_value="8",
                              description="traffic cars near the AV"),
        DeclareLaunchArgument("foxglove_port", default_value="8765",
                              description="the WebSocket port Foxglove connects to"),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(l2_launch),
            launch_arguments={"host": arg("host"), "port": arg("port"),
                              "town": arg("town"), "rviz": "false",
                              "rate_report": "false"}.items()),

        # Scoped: foxglove_bridge's launch file also has an argument named "port".
        # Unscoped, it would overwrite CARLA's port for everything launched after it.
        GroupAction(scoped=True, actions=[IncludeLaunchDescription(
            AnyLaunchDescriptionSource(fox_launch),
            launch_arguments={"port": arg("foxglove_port")}.items())]),

        Node(package="l8_foxglove_demo", executable="l8_scene", name="l8_scene",
             output="screen", emulate_tty=True,
             parameters=[params, {
                 "host": arg("host"),
                 "port": ParameterValue(arg("port"), value_type=int),
                 "vehicles": ParameterValue(arg("vehicles"), value_type=int),
             }]),
    ])
