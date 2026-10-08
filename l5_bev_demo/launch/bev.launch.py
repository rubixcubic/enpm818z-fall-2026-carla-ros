"""Bring up the surround cameras and the three bird's-eye views.

    ros2 launch l2_carla_demo demo.launch.py rviz:=false     # first, terminal 1
    ros2 launch l5_bev_demo bev.launch.py                    # then, terminal 2
    ros2 launch l5_bev_demo bev.launch.py rviz:=false

The L2 bridge must be running: it owns the simulation clock, spawns the ego
vehicle and publishes the LiDAR. This launch adds four roof cameras to that
vehicle and three nodes that only subscribe, never touch the simulator:

    lidar_bev     -> /l5/bev/lidar_height, /l5/bev/occupancy
    ipm_bev       -> /l5/bev/ipm
    semantic_bev  -> /l5/bev/semantic

All three share one grid (config/bev.yaml), so the views line up cell for cell.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("l5_bev_demo")
    params = os.path.join(share, "config", "bev.yaml")
    rviz_config = os.path.join(share, "rviz", "l5_bev.rviz")

    def node(executable: str) -> Node:
        return Node(package="l5_bev_demo", executable=executable, name=executable,
                    output="screen", emulate_tty=True, parameters=[params])

    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true",
                              description="show the four views and the point cloud"),
        node("surround_rig"),
        node("lidar_bev"),
        node("ipm_bev"),
        node("semantic_bev"),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             arguments=["-d", rviz_config],
             condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
