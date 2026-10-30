"""GP3 Task 1: drive on autopilot with traffic and record what GP3 needs.

    ros2 launch ads_pipeline record_gp3_launch.py config:=<your YAML> out:=<bag dir>

The drive stops by itself after `duration` seconds of simulated time
(default 90), and the bag closes with it. Check it with `ros2 bag info`.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

GP3_TOPICS = [
    '/carla/camera/rgb/image',
    '/carla/lidar/points',
    '/carla/gnss/fix',
    '/carla/imu/data',
    '/ground_truth/ego',
    '/ground_truth/objects',
    '/tf',
    '/tf_static',
]


def generate_launch_description():
    pkg_share = get_package_share_directory('ads_pipeline')
    args = [
        DeclareLaunchArgument(
            'config',
            default_value=os.path.join(pkg_share, 'config', 'carla_config.yaml'),
            description='Your team YAML, with the GP3 GNSS mount.'),
        DeclareLaunchArgument(
            'out', default_value=os.path.expanduser('~/gp3_data/drive'),
            description='Bag directory to create (must not exist yet).'),
        DeclareLaunchArgument(
            'duration', default_value='90.0',
            description='Seconds of simulated time to drive.'),
    ]

    sensor_manager = Node(
        package='ads_pipeline', executable='sensor_manager',
        name='sensor_manager', output='screen', emulate_tty=True,
        parameters=[
            LaunchConfiguration('config'),
            {'autopilot': True, 'ignore_lights': True, 'ground_truth': True,
             'max_sim_s': ParameterValue(LaunchConfiguration('duration'),
                                         value_type=float)},
        ])

    record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-s', 'mcap',
             '-o', LaunchConfiguration('out'), *GP3_TOPICS],
        output='screen')

    stop_all = RegisterEventHandler(OnProcessExit(
        target_action=sensor_manager,
        on_exit=[EmitEvent(event=Shutdown(reason='drive finished'))]))

    return LaunchDescription([*args, record, sensor_manager, stop_all])
