#!/usr/bin/env python3
"""L8 in Foxglove: every car's box and its constant-velocity (CV) prediction, live.

A second CARLA client next to the L2 bridge. The bridge owns the clock and the
AV; this node never ticks. It waits for the AV (role_name "ego"), spawns traffic
at the spawn points nearest it, and then, every 0.1 s of wall time, reads the
latest world snapshot and publishes:

    /l8/vehicles         visualization_msgs/MarkerArray  each car's true box and speed
    /l8/predictions      visualization_msgs/MarkerArray  each car's CV path, 5 s ahead
    /l8/ego/speed        std_msgs/Float32                the AV's speed, m/s
    /l8/nearest/speed    std_msgs/Float32                the nearest other car's speed
    /l8/nearest/distance std_msgs/Float32                its distance to the AV, m

and a warning in the log (Foxglove's Log panel, /rosout) when a car's CV path
comes within warn_distance of the AV's own CV path at the same time.

The CV predictions start from CARLA's TRUE velocity, not from a tracker: the
point here is the viewer, not the estimate (L8 fits the velocity from a track).

    ros2 launch l8_foxglove_demo l8_foxglove.launch.py

Ctrl+C removes every car this node spawned.
"""

from __future__ import annotations

import math
import random

import carla
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from visualization_msgs.msg import MarkerArray

from l2_carla_demo.conversions import header as make_header
from l8_foxglove_demo import cv_predict as cvp
from l8_foxglove_demo.markers import MAP_FRAME, Vehicle, prediction_markers, vehicle_markers


class L8Scene(Node):

    def __init__(self) -> None:
        super().__init__("l8_scene")
        p = self.declare_parameter
        p("host", "localhost")
        p("port", 2000)
        p("tm_port", 8000)
        p("vehicles", 8)            # traffic cars to spawn (chosen)
        p("seed", 7)
        p("min_spawn_gap", 10.0)    # m: no traffic spawned this close to the AV (chosen)
        p("rate_hz", 10.0)          # publish rate: L8's 0.1 s step
        p("horizon", 5.0)           # s: L8's horizon
        p("step", 0.1)              # s between predicted points: L8's
        p("dot_every", 1.0)         # s between dots on a path: as in L8's figures
        p("watch_radius", 50.0)     # m: cars farther from the AV are not checked
        p("warn_distance", 2.0)     # m: half of a 4 m lane, L8's miss threshold
        p("warn_every", 2.0)        # s: at most one warning per car this often
        g = lambda name: self.get_parameter(name).value
        self.g = g

        self.client = carla.Client(g("host"), int(g("port")))
        # Short: a call that blocks here blocks the timer. A busy server (the bridge
        # loading a map) only costs a retry on the next tick.
        self.client.set_timeout(5.0)
        self.world = None                               # fetched in _tick
        self.episode = None                             # world.id: changes when a map loads
        self.times = cvp.horizons(float(g("horizon")), float(g("step")))

        self.ego_id: int | None = None
        self.boxes: dict[int, carla.BoundingBox] = {}   # actor id -> bounding box
        self.spawned: list[int] = []
        self.last_frame = -1
        self.last_refresh = -math.inf
        self.last_warn: dict[int, float] = {}

        self.pub_vehicles = self.create_publisher(MarkerArray, "/l8/vehicles", 10)
        self.pub_predictions = self.create_publisher(MarkerArray, "/l8/predictions", 10)
        self.pub_ego_speed = self.create_publisher(Float32, "/l8/ego/speed", 10)
        self.pub_near_speed = self.create_publisher(Float32, "/l8/nearest/speed", 10)
        self.pub_near_dist = self.create_publisher(Float32, "/l8/nearest/distance", 10)

        self.timer = self.create_timer(1.0 / float(g("rate_hz")), self._tick)
        self.get_logger().info("waiting for the AV (role_name 'ego') from the L2 bridge")

    # ---------------------------------------------------------------- actors
    def _refresh_actors(self) -> None:
        """Every vehicle's bounding box, and which one is the AV."""
        boxes, ego = {}, None
        for actor in self.world.get_actors().filter("vehicle.*"):
            boxes[actor.id] = actor.bounding_box
            if actor.attributes.get("role_name") == "ego":
                ego = actor.id
        self.boxes, self.ego_id = boxes, ego

    def _spawn_traffic(self) -> None:
        """N cars on the Traffic Manager's autopilot, at the spawn points nearest the AV."""
        g = self.g
        ego = self.world.get_actor(self.ego_id)
        here = ego.get_transform().location
        points = [pt for pt in self.world.get_map().get_spawn_points()
                  if pt.location.distance(here) >= float(g("min_spawn_gap"))]
        points.sort(key=lambda pt: pt.location.distance(here))
        rng = random.Random(int(g("seed")))
        blueprints = [b for b in self.world.get_blueprint_library().filter("vehicle.*")
                      if int(b.get_attribute("number_of_wheels")) == 4]
        tm = self.client.get_trafficmanager(int(g("tm_port")))
        tm.set_random_device_seed(int(g("seed")))
        batch = []
        for point in points[:int(g("vehicles"))]:
            bp = rng.choice(blueprints)
            bp.set_attribute("role_name", "l8_traffic")
            batch.append(carla.command.SpawnActor(bp, point).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())))
        responses = self.client.apply_batch_sync(batch, False)
        self.spawned = [r.actor_id for r in responses if not r.error]
        self.get_logger().info(
            f"spawned {len(self.spawned)} cars on autopilot near the AV "
            f"({len(batch) - len(self.spawned)} spawn points were taken)")

    # ------------------------------------------------------------------ loop
    def _tick(self) -> None:
        try:
            self._step()
        except RuntimeError as exc:
            # The server did not answer in time: the bridge may be loading a map,
            # which also replaces the world. Fetch it again on the next tick.
            self.world, self.ego_id = None, None
            self.get_logger().warn(f"CARLA is busy, trying again ({exc})",
                                   throttle_duration_sec=5.0)

    def _step(self) -> None:
        if self.world is None:
            self.world = self.client.get_world()
            if self.world.id != self.episode:           # a new map: the old cars are gone
                self.episode, self.spawned, self.last_frame = self.world.id, [], -1
        snapshot = self.world.get_snapshot()
        if snapshot.frame == self.last_frame:
            return                                    # the bridge has not ticked yet
        self.last_frame = snapshot.frame
        now = snapshot.timestamp.elapsed_seconds

        if self.ego_id is None or now - self.last_refresh > 2.0:
            self._refresh_actors()                    # new cars appear, old ones leave
            self.last_refresh = now
            if self.ego_id is None:
                return
            if not self.spawned:
                self._spawn_traffic()
                self._refresh_actors()

        vehicles = []
        for actor_id, box in self.boxes.items():
            snap = snapshot.find(actor_id)
            if snap is None:
                continue
            tf = snap.get_transform()
            # Transform.transform() also OVERWRITES the point it is given (0.9.16):
            # pass a copy, or the cached box drifts by the car's position every tick.
            loc = box.location
            center = tf.transform(carla.Location(loc.x, loc.y, loc.z))   # CARLA world frame
            vel = snap.get_velocity()
            # CARLA is left-handed: y and yaw change sign on the way to ROS.
            vehicles.append(Vehicle(
                actor_id=actor_id, x=center.x, y=-center.y, z=center.z,
                yaw=math.radians(-tf.rotation.yaw), vx=vel.x, vy=-vel.y,
                length=2 * box.extent.x, width=2 * box.extent.y, height=2 * box.extent.z,
                is_ego=(actor_id == self.ego_id)))
        ego = next((v for v in vehicles if v.is_ego), None)
        if ego is None:
            self.ego_id = None                        # the bridge stopped: wait again
            return

        header = make_header(MAP_FRAME, now)
        life = 3.0 / float(self.g("rate_hz"))
        paths = {v.actor_id: cvp.cv_path(v.x, v.y, v.vx, v.vy, self.times) for v in vehicles}
        self.pub_vehicles.publish(vehicle_markers(vehicles, header, life))
        self.pub_predictions.publish(prediction_markers(
            paths, self.times, self.ego_id, ego.z - ego.height / 2 + 0.3,
            float(self.g("dot_every")), header, life))
        self.pub_ego_speed.publish(Float32(data=cvp.speed(ego.vx, ego.vy)))

        others = [(math.hypot(v.x - ego.x, v.y - ego.y), v) for v in vehicles if not v.is_ego]
        if others:
            dist, near = min(others, key=lambda dv: dv[0])
            self.pub_near_speed.publish(Float32(data=cvp.speed(near.vx, near.vy)))
            self.pub_near_dist.publish(Float32(data=dist))
        self._warn(others, paths, now)

    def _warn(self, others, paths, now: float) -> None:
        """Log when a car's CV path comes close to the AV's CV path at the same time."""
        g = self.g
        for dist, v in others:
            if dist > float(g("watch_radius")):
                continue
            d, t = cvp.closest_approach(paths[v.actor_id], paths[self.ego_id], self.times)
            if d < float(g("warn_distance")) and \
                    now - self.last_warn.get(v.actor_id, -math.inf) >= float(g("warn_every")):
                self.last_warn[v.actor_id] = now
                self.get_logger().warn(
                    f"car {v.actor_id}: its CV path comes {d:.2f} m from the AV's "
                    f"in {t:.1f} s")

    # --------------------------------------------------------------- cleanup
    def cleanup(self) -> None:
        if self.spawned:
            self.client.apply_batch([carla.command.DestroyActor(i) for i in self.spawned])
            print(f"removed {len(self.spawned)} cars", flush=True)
            self.spawned = []


def main(args=None) -> None:
    rclpy.init(args=args)
    node = L8Scene()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
