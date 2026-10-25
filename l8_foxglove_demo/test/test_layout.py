"""The Foxglove layout: valid JSON, every panel placed, every topic one we publish."""

import json
import re
from pathlib import Path

LAYOUT = Path(__file__).resolve().parents[1] / "foxglove" / "l8_layout.json"

# What the L2 bridge and the l8_scene node publish (l2_carla_demo/carla_bridge_node.py,
# l8_foxglove_demo/l8_scene_node.py), plus /rosout for the Log panel.
PUBLISHED = {
    "/carla/ego_vehicle/rgb_front/image", "/carla/ego_vehicle/rgb_front/camera_info",
    "/carla/ego_vehicle/lidar", "/carla/ego_vehicle/radar_front", "/carla/ego_vehicle/imu",
    "/carla/ego_vehicle/gnss", "/carla/ego_vehicle/marker", "/l8/vehicles",
    "/l8/predictions", "/l8/ego/speed", "/l8/nearest/speed", "/l8/nearest/distance",
}


def _leaves(node):
    if isinstance(node, str):
        return [node]
    return _leaves(node["first"]) + _leaves(node["second"])


def test_every_panel_is_placed_once():
    d = json.loads(LAYOUT.read_text())
    placed = _leaves(d["layout"])
    assert sorted(placed) == sorted(d["configById"])


def test_every_topic_is_published():
    text = LAYOUT.read_text()
    topics = set(re.findall(r'"(/[a-z0-9_/]+?)(?:\.data)?"', text))
    assert topics, "no topics found"
    assert topics <= PUBLISHED, topics - PUBLISHED
