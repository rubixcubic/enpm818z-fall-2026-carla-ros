"""GP4: the launch file that runs GP1 to GP4 live, in lock step.

Skipped without ROS 2 sourced.
"""
import importlib.util
from pathlib import Path

import pytest


def load():
    pytest.importorskip('launch')
    path = Path(__file__).resolve().parents[1] / 'launch' / 'gp4_launch.py'
    spec = importlib.util.spec_from_file_location('gp4_launch', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.generate_launch_description().entities


def test_launch_declares_its_arguments():
    from launch.actions import DeclareLaunchArgument
    names = {e.name for e in load() if isinstance(e, DeclareLaunchArgument)}
    assert names >= {'config', 'route_file', 'spawn_index', 'pose_topic', 'ekf_config', 'fusion_config',
                     'planner_config', 'controller_config', 'behavior_config', 'out',
                     'duration'}


def test_launch_starts_gp1_to_gp4():
    pytest.importorskip('launch_ros')
    from launch_ros.actions import Node
    exes = {e.node_executable for e in load() if isinstance(e, Node)}
    assert exes == {'sensor_manager', 'detector_node', 'gnss_outage', 'localization_node',
                    'fusion_node', 'tracker_node', 'planner_node', 'behavior_node',
                    'controller_node'}
