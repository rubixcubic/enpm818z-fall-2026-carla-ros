"""GP4: your whole pipeline, live in CARLA, in lock step.

    ros2 launch ads_pipeline gp4_launch.py config:=$PWD/config/carla_config.yaml \\
        route_file:=$PWD/routes/<rig>.yaml out:=$HOME/gp4_data/run_1

GP1's sensor_manager drives the world in synchronous mode and ticks only
after the controller has answered the last frame (lock_step). GP2's
detector, GP3's localization, fusion and tracker, and GP4's planner,
behavior and controller run in between. The run stops after `duration`
seconds of simulated time and the bag closes with it.
"""

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

OUT_TOPICS = ['/planning/path', '/behavior/state', '/behavior/target_speed',
              '/control/vehicle_cmd', '/localization/pose', '/localization/odom',
              '/perception/tracks', '/ground_truth/ego', '/ground_truth/objects',
              '/ground_truth/collision', '/ground_truth/traffic_light']


def generate_launch_description():
    args = [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in (
        ('config', 'config/carla_config.yaml', 'Your GP1 YAML (rig, vehicle, traffic).'),
        ('route_file', 'routes/route.yaml', 'Your route: start spawn point and goal.'),
        ('spawn_index', '', 'The route file\'s spawn_index (overrides your YAML).'),
        ('map_file', 'ads_pipeline/maps/Town01.xodr', 'The town the route is in.'),
        ('points_file', 'routes/town01_points.json', 'Spawn points and traffic light stop lines.'),
        ('pose_topic', '/localization/pose', 'Your EKF, or /ground_truth/ego (report).'),
        ('ekf_config', 'config/ekf_config.yaml', 'GP3 EKF settings.'),
        ('fusion_config', 'config/fusion_config.yaml', 'GP3 fusion settings.'),
        ('planner_config', 'config/planner_config.yaml', 'Triggers and search.'),
        ('controller_config', 'config/controller_config.yaml', 'Pure Pursuit and PID.'),
        ('behavior_config', 'config/behavior_config.yaml', 'States and distances.'),
        ('duration', '300.0', 'Seconds of simulated time before the run stops.'),
        ('cmd_timeout_s', '2.0', 'Lock step: tick anyway after this long (logged).'),
        ('out', 'results/gp4/run', 'Bag of the run (must not exist yet).'),
        ('model_type', 'yolo', "GP2: 'yolo' or 'detr'."),
        ('weights', 'yolov8s.pt', 'GP2: Ultralytics weights.'),
        ('device', 'cuda', 'GP2: cuda or cpu.'),
    )]
    pose = {'pose_topic': LC('pose_topic')}
    sensor_manager = Node(
        package='ads_pipeline', executable='sensor_manager', output='screen',
        parameters=[LC('config'), {
            'lock_step': True, 'ground_truth': True, 'autopilot': False,
            'ignore_lights': False,
            'spawn_index': ParameterValue(LC('spawn_index'), value_type=int),
            'cmd_timeout_s': ParameterValue(LC('cmd_timeout_s'), value_type=float),
            'max_sim_s': ParameterValue(LC('duration'), value_type=float)}])
    nodes = [
        sensor_manager,
        Node(package='ads_pipeline', executable='detector_node', output='screen',
             parameters=[{'model_type': LC('model_type'), 'weights': LC('weights'),
                          'device': LC('device')}]),
        Node(package='ads_pipeline', executable='gnss_outage', output='screen',
             parameters=[{'outage_start_s': 0.0, 'outage_end_s': 0.0}]),
        Node(package='ads_pipeline', executable='localization_node', output='screen',
             parameters=[LC('ekf_config'), {'use_sim_time': False}]),
        Node(package='ads_pipeline', executable='fusion_node', output='screen',
             parameters=[LC('fusion_config'), {'use_sim_time': False,
                                               'rig_config': LC('config')}, pose]),
        Node(package='ads_pipeline', executable='tracker_node', output='screen'),
        Node(package='ads_pipeline', executable='planner_node', output='screen',
             parameters=[LC('planner_config'), {'route_file': LC('route_file'),
                                                'map_file': LC('map_file'),
                                                'points_file': LC('points_file')}, pose]),
        Node(package='ads_pipeline', executable='behavior_node', output='screen',
             parameters=[LC('behavior_config'), {'rig_config': LC('config')}, pose]),
        Node(package='ads_pipeline', executable='controller_node', output='screen',
             parameters=[LC('controller_config'), pose]),
    ]
    record = ExecuteProcess(cmd=['ros2', 'bag', 'record', '-s', 'mcap', '-o', LC('out'),
                                 *OUT_TOPICS], output='screen')
    stop = RegisterEventHandler(OnProcessExit(
        target_action=sensor_manager,
        on_exit=[EmitEvent(event=Shutdown(reason='run finished'))]))
    return LaunchDescription([*args, record, *nodes, stop])
