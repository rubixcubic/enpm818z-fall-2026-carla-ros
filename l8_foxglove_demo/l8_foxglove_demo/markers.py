"""MarkerArray builders for the vehicles and their CV predictions.

No CARLA import: every input is plain numbers already in the ROS map frame
(x east, y north, yaw counterclockwise from x, in radians), so the tests can
build and check the messages without a simulator.

Colors follow L8's figures: the AV dark, the other cars orange (car A's color),
CV predictions blue.
"""

from __future__ import annotations

from dataclasses import dataclass

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from l2_carla_demo.conversions import quaternion_from_euler

MAP_FRAME = "map"


def rgba(hex_rgb: str, a: float = 1.0) -> ColorRGBA:
    h = hex_rgb.lstrip("#")
    return ColorRGBA(r=int(h[0:2], 16) / 255.0, g=int(h[2:4], 16) / 255.0,
                     b=int(h[4:6], 16) / 255.0, a=a)


AV_COLOR = "#1E2939"       # L8's ink: the AV
CAR_COLOR = "#D9694A"      # L8's car A
CV_COLOR = "#2D6CA2"       # L8's CV prediction
AV_PATH_COLOR = "#7A828C"  # the AV's own CV path, gray


@dataclass
class Vehicle:
    """One vehicle in the ROS map frame."""
    actor_id: int
    x: float           # center of the bounding box, m
    y: float
    z: float
    yaw: float         # rad, counterclockwise from x (east)
    vx: float          # m/s
    vy: float
    length: float      # full size, m
    width: float
    height: float
    is_ego: bool = False


def _lifetime(seconds: float) -> Duration:
    sec = int(seconds)
    return Duration(sec=sec, nanosec=int((seconds - sec) * 1e9))


def vehicle_markers(vehicles: list[Vehicle], header: Header,
                    lifetime: float) -> MarkerArray:
    """A box and a speed label for every vehicle, after a DELETEALL.

    DELETEALL first, so a car that left (or was destroyed) does not stay on
    screen; the lifetime is a second guard if a message is lost.
    """
    out = MarkerArray()
    clear = Marker(header=header, action=Marker.DELETEALL)
    out.markers.append(clear)
    for v in vehicles:
        box = Marker(header=header, ns="vehicles", id=v.actor_id, type=Marker.CUBE,
                     action=Marker.ADD, lifetime=_lifetime(lifetime))
        box.pose.position.x, box.pose.position.y, box.pose.position.z = v.x, v.y, v.z
        box.pose.orientation = quaternion_from_euler(0.0, 0.0, v.yaw)
        box.scale.x, box.scale.y, box.scale.z = v.length, v.width, v.height
        box.color = rgba(AV_COLOR if v.is_ego else CAR_COLOR, 0.85)
        out.markers.append(box)

        label = Marker(header=header, ns="speed", id=v.actor_id,
                       type=Marker.TEXT_VIEW_FACING, action=Marker.ADD,
                       lifetime=_lifetime(lifetime))
        label.pose.position.x, label.pose.position.y = v.x, v.y
        label.pose.position.z = v.z + v.height / 2 + 1.0
        label.pose.orientation.w = 1.0
        label.scale.z = 1.2                       # text height, m
        label.color = rgba("#FFFFFF")
        name = "AV" if v.is_ego else f"car {v.actor_id}"
        label.text = f"{name}: {(v.vx ** 2 + v.vy ** 2) ** 0.5:.1f} m/s"
        out.markers.append(label)
    return out


def prediction_markers(paths: dict[int, list[tuple[float, float]]],
                       times: list[float], ego_id: int | None, z: float,
                       dot_every: float, header: Header,
                       lifetime: float) -> MarkerArray:
    """Each vehicle's CV path as a line, with a dot every `dot_every` seconds."""
    out = MarkerArray()
    out.markers.append(Marker(header=header, action=Marker.DELETEALL))
    for actor_id, path in paths.items():
        color = rgba(AV_PATH_COLOR if actor_id == ego_id else CV_COLOR)
        line = Marker(header=header, ns="cv_path", id=actor_id, type=Marker.LINE_STRIP,
                      action=Marker.ADD, lifetime=_lifetime(lifetime))
        line.pose.orientation.w = 1.0
        line.scale.x = 0.25                       # line width, m
        line.color = color
        line.points = [Point(x=x, y=y, z=z) for x, y in path]
        out.markers.append(line)

        dots = Marker(header=header, ns="cv_dots", id=actor_id, type=Marker.SPHERE_LIST,
                      action=Marker.ADD, lifetime=_lifetime(lifetime))
        dots.pose.orientation.w = 1.0
        dots.scale.x = dots.scale.y = dots.scale.z = 0.8
        dots.color = color
        dots.points = [Point(x=x, y=y, z=z) for (x, y), t in zip(path, times)
                       if abs(t / dot_every - round(t / dot_every)) < 1e-6]
        out.markers.append(dots)
    return out
