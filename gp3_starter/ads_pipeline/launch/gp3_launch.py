"""GP3: replay your drive through GP2's detector and GP3's three nodes.

    ros2 launch ads_pipeline gp3_launch.py bag:=<your drive> config:=<your GP1 YAML> \\
        ekf_config:=<ekf_config.yaml> fusion_config:=<fusion_config.yaml> \\
        outage_start_s:=<from the table> outage_end_s:=<from the table> out:=results/gp3/run_1

The bag plays at `rate` (try 0.25 if the detector cannot keep up). The
outputs and the ground truth are recorded to `out`, which the evaluators
in tools/ read. Everything stops when the bag ends.
"""

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

OUT_TOPICS = ['/perception/detections', '/perception/fused_objects',
              '/perception/tracks', '/localization/pose', '/localization/odom',
              '/gnss/fix_used', '/carla/gnss/fix', '/ground_truth/ego',
              '/ground_truth/objects']


def generate_launch_description():
    args = [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in (
        ('bag', '', 'Your GP3 drive (Task 1).'),
        ('rate', '1.0', 'Replay speed; lower it if the detector falls behind.'),
        ('pose_topic', '/localization/pose', 'Pose for fusion: your EKF or /ground_truth/ego.'),
        ('config', 'config/carla_config.yaml', 'Your GP1 YAML (camera and mounts).'),
        ('ekf_config', 'config/ekf_config.yaml', 'Q, R, antenna and map origin.'),
        ('fusion_config', 'config/fusion_config.yaml', 'Range window and DBSCAN.'),
        ('outage_start_s', '0.0', 'GNSS outage start, s after the first fix.'),
        ('outage_end_s', '0.0', 'GNSS outage end.'),
        ('out', 'results/gp3/run', 'Bag of the outputs (must not exist yet).'),
        ('nis_csv', 'results/gp3/nis.csv', 'Where the EKF writes its NIS values.'),
        ('model_type', 'yolo', "GP2: 'yolo' or 'detr'."),
        ('weights', 'yolov8s.pt', 'GP2: Ultralytics weights.'),
        ('device', 'cuda', 'GP2: cuda or cpu.'),
    )]
    sim = {'use_sim_time': True}
    nodes = [
        Node(package='ads_pipeline', executable='detector_node', output='screen',
             parameters=[sim, {'model_type': LC('model_type'), 'weights': LC('weights'),
                               'device': LC('device')}]),
        Node(package='ads_pipeline', executable='gnss_outage', output='screen',
             parameters=[sim, {
                 'outage_start_s': ParameterValue(LC('outage_start_s'), value_type=float),
                 'outage_end_s': ParameterValue(LC('outage_end_s'), value_type=float)}]),
        Node(package='ads_pipeline', executable='localization_node', output='screen',
             parameters=[LC('ekf_config'), sim, {'nis_csv': LC('nis_csv')}]),
        Node(package='ads_pipeline', executable='fusion_node', output='screen',
             parameters=[LC('fusion_config'), sim, {'rig_config': LC('config'),
                                                    'pose_topic': LC('pose_topic')}]),
        Node(package='ads_pipeline', executable='tracker_node', output='screen',
             parameters=[sim]),
    ]
    record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-s', 'mcap', '--use-sim-time',
             '-o', LC('out'), *OUT_TOPICS], output='screen')
    play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', LC('bag'), '--clock', '--rate', LC('rate'),
             '--delay', '3'],
        output='screen')
    stop = RegisterEventHandler(OnProcessExit(
        target_action=play, on_exit=[EmitEvent(event=Shutdown(reason='bag ended'))]))
    return LaunchDescription([*args, *nodes, record, play, stop])
