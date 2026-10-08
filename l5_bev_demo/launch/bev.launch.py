"""Bring up the surround cameras and the three bird's-eye views.

    ros2 launch l2_carla_demo demo.launch.py rviz:=false     # first, terminal 1
    ros2 launch l5_bev_demo bev.launch.py                    # then, terminal 2
    ros2 launch l5_bev_demo bev.launch.py rviz:=false
    ros2 launch l5_bev_demo bev.launch.py labels:=network    # a network's classes

The L2 bridge must be running: it owns the simulation clock, spawns the ego
vehicle and publishes the LiDAR. This launch adds four roof cameras to that
vehicle and three nodes that only subscribe, never touch the simulator:

    lidar_bev     -> /l5/bev/lidar_height, /l5/bev/occupancy
    ipm_bev       -> /l5/bev/ipm
    semantic_bev  -> /l5/bev/semantic

All three share one grid (config/bev.yaml), so the views line up cell for cell.

labels:=network also starts l5_seg_demo's seg_node on the four roof cameras,
and semantic_bev splats the network's classes instead of CARLA's true ones.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share = get_package_share_directory("l5_bev_demo")
    params = os.path.join(share, "config", "bev.yaml")
    rviz_config = os.path.join(share, "rviz", "l5_bev.rviz")

    def node(executable: str, extra: dict | None = None) -> Node:
        return Node(package="l5_bev_demo", executable=executable, name=executable,
                    output="screen", emulate_tty=True, parameters=[params, extra or {}])

    network = IfCondition(PythonExpression(["'", LaunchConfiguration("labels"),
                                            "' == 'network'"]))

    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true",
                              description="show the four views and the point cloud"),
        node("surround_rig"),
        node("lidar_bev"),
        node("ipm_bev"),
        DeclareLaunchArgument("labels", default_value="truth",
                              description="semantic_bev's classes: truth (CARLA's) "
                                          "or network (l5_seg_demo's SegFormer)"),
        node("semantic_bev", {"labels": LaunchConfiguration("labels")}),
        Node(package="l5_seg_demo", executable="seg_node", name="seg_node_rig",
             output="screen", emulate_tty=True,
             parameters=[{"image_topic": "", "rig": True}], condition=network),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             arguments=["-d", rviz_config],
             condition=IfCondition(LaunchConfiguration("rviz"))),
    ])
