"""CARLA's ground truth, recorded tick by tick, for the truth detector and the evaluator.

The bridge owns the clock: this client never ticks. It asks CARLA to call
back on every tick (world.on_tick) and keeps each tick's actor poses, keyed
by simulation time, so a detection or a track stamped t is compared with the
world as it was at t, not one tick later.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import carla

from l5_tracking_demo.ros_util import actor_kind, time_key, truth_from_snapshot


class TruthWatcher:

    def __init__(self, host: str, port: int, role_name: str = "ego",
                 keep_frames: int = 400) -> None:
        self.client = carla.Client(host, port)
        self.client.set_timeout(20.0)
        self.world = self.client.get_world()
        self.role_name = role_name
        self.keep = keep_frames
        self.lock = threading.Lock()
        self.frames: OrderedDict[int, list] = OrderedDict()   # time key -> truth list
        self.new: list[tuple[float, list]] = []               # not yet handed out
        self.kinds: dict[int, str] = {}
        self.ego_id: int | None = None
        self._refresh_actors()
        self._ticks = 0
        self.callback_id = self.world.on_tick(self._on_tick)

    def _refresh_actors(self) -> None:
        kinds, ego = {}, None
        for a in self.world.get_actors():
            kind = actor_kind(a.type_id)
            if kind is None:
                continue
            kinds[a.id] = kind
            if a.attributes.get("role_name") == self.role_name:
                ego = a.id
        self.kinds, self.ego_id = kinds, ego

    def _on_tick(self, snapshot) -> None:
        """Called by CARLA's client thread on every tick: copy what we need, fast."""
        self._ticks += 1
        if self._ticks % 40 == 0 or self.ego_id is None:    # new actors appear
            try:
                self._refresh_actors()
            except RuntimeError:
                return
        t = snapshot.timestamp.elapsed_seconds
        truth = truth_from_snapshot(snapshot, self.kinds)
        with self.lock:
            self.frames[time_key(t)] = truth
            while len(self.frames) > self.keep:
                self.frames.popitem(last=False)
            self.new.append((t, truth))

    def take_new(self) -> list[tuple[float, list]]:
        with self.lock:
            out, self.new = self.new, []
        return out

    def at(self, t: float) -> list | None:
        with self.lock:
            return self.frames.get(time_key(t))

    def close(self) -> None:
        try:
            self.world.remove_on_tick(self.callback_id)
        except RuntimeError:
            pass
