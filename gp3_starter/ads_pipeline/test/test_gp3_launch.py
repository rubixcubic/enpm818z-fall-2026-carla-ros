"""GP3: the launch file that replays a drive through GP2's and GP3's nodes.

Skipped without ROS 2 sourced.
"""
import importlib.util
from pathlib import Path

import pytest


def load():
    pytest.importorskip('launch')
    path = Path(__file__).resolve().parents[1] / 'launch' / 'gp3_launch.py'
    spec = importlib.util.spec_from_file_location('gp3_launch', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.generate_launch_description().entities


def test_launch_declares_its_arguments():
    from launch.actions import DeclareLaunchArgument
    names = {e.name for e in load() if isinstance(e, DeclareLaunchArgument)}
    assert names >= {'bag', 'rate', 'pose_topic', 'config', 'ekf_config',
                     'fusion_config', 'outage_start_s', 'outage_end_s', 'out'}


def test_launch_starts_the_five_nodes():
    pytest.importorskip('launch_ros')
    from launch_ros.actions import Node
    exes = {e.node_executable for e in load() if isinstance(e, Node)}
    assert exes == {'detector_node', 'gnss_outage', 'localization_node',
                    'fusion_node', 'tracker_node'}
