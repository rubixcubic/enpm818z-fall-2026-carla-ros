"""CARLA's true vehicle boxes, tick by tick, in the ROS map frame.

Extends l5_tracking_demo's TruthWatcher (which records each tick's actor poses
keyed by simulation time, and never ticks) with each vehicle's bounding box:
its size and its center, which CARLA gives relative to the actor's origin.
Parked cars that are part of the map are not actors; with include_parked they
are added from world.get_environment_objects, as boxes that never move.

CARLA is left-handed with y to the right; ROS has y to the left: y and the yaw
change sign (the L2 axis trap).
"""

from __future__ import annotations

import math

import carla

from l5_tracking_demo.carla_truth import TruthWatcher
from l5_tracking_demo.ros_util import time_key


def ros_box(center, extent, yaw_deg: float, ident) -> dict:
    return {"id": ident, "x": center.x, "y": -center.y, "z": center.z,
            "yaw": -math.radians(yaw_deg),
            "size": (2.0 * extent.x, 2.0 * extent.y, 2.0 * extent.z)}


class BoxTruth(TruthWatcher):

    PARKED_LABELS = ("Car", "Truck", "Bus")

    def __init__(self, host: str, port: int, role_name: str = "ego",
                 include_parked: bool = True, keep_frames: int = 400) -> None:
        self.bboxes: dict[int, carla.BoundingBox] = {}
        self.box_frames: dict[int, list[dict]] = {}
        super().__init__(host, port, role_name, keep_frames)
        self.parked: list[dict] = []
        if include_parked:
            for name in self.PARKED_LABELS:
                label = getattr(carla.CityObjectLabel, name, None)
                if label is None:
                    continue
                for obj in self.world.get_environment_objects(label):
                    bb = obj.bounding_box          # world frame
                    self.parked.append(ros_box(bb.location, bb.extent, bb.rotation.yaw,
                                               f"parked-{obj.id}"))

    def _refresh_actors(self) -> None:
        super()._refresh_actors()
        for a in self.world.get_actors().filter("vehicle.*"):
            if a.id not in self.bboxes:
                self.bboxes[a.id] = a.bounding_box

    def _on_tick(self, snapshot) -> None:
        super()._on_tick(snapshot)
        boxes = []
        for a in snapshot:
            bb = self.bboxes.get(a.id)
            if bb is None or a.id == self.ego_id:
                continue
            tf = a.get_transform()
            # Transform.transform() changes the Location it is given (CARLA 0.9.16):
            # transform a copy, or the cached offset drifts by a pose every tick.
            center = tf.transform(carla.Location(bb.location.x, bb.location.y, bb.location.z))
            boxes.append(ros_box(center, bb.extent, tf.rotation.yaw, a.id))
        key = time_key(snapshot.timestamp.elapsed_seconds)
        with self.lock:
            self.box_frames[key] = boxes
            while len(self.box_frames) > self.keep:
                self.box_frames.pop(next(iter(self.box_frames)))

    def boxes_at(self, t: float) -> list[dict] | None:
        with self.lock:
            moving = self.box_frames.get(time_key(t))
        if moving is None:
            return None
        return moving + self.parked

    def ego_at(self, t: float):
        """(x, y, yaw) of the AV in the map frame at time t, from the actor list."""
        with self.lock:
            frame = self.frames.get(time_key(t))
        if frame is None:
            return None
        for o in frame:
            if o[0] == self.ego_id:
                return o[1], o[2], o[5]
        return None
