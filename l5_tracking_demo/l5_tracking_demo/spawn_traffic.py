#!/usr/bin/env python3
"""Fill the map with traffic for the tracker: N vehicles on CARLA's autopilot.

The L2 bridge spawns only the AV, and a tracker with nothing to track teaches
nothing. Start the bridge first (it puts the server and the Traffic Manager
in synchronous mode and owns the clock), then:

    ros2 run l5_tracking_demo spawn_traffic --ros-args -p vehicles:=40

This client never ticks. Ctrl+C removes every vehicle it spawned.
"""

from __future__ import annotations

import random
import time

import carla
import rclpy
from rclpy.node import Node

from l5_tracking_demo.ros_util import spin


class SpawnTraffic(Node):

    def __init__(self) -> None:
        super().__init__("spawn_traffic")
        p = self.declare_parameter
        p("host", "localhost")
        p("port", 2000)
        p("tm_port", 8000)
        p("vehicles", 40)
        p("seed", 7)
        g = lambda name: self.get_parameter(name).value
        self.client = carla.Client(g("host"), int(g("port")))
        self.client.set_timeout(20.0)
        world = self.client.get_world()
        if not world.get_settings().synchronous_mode:
            self.get_logger().warn("the server is not in synchronous mode: "
                                   "start the L2 bridge first")
        tm = self.client.get_trafficmanager(int(g("tm_port")))
        tm.set_random_device_seed(int(g("seed")))
        rng = random.Random(int(g("seed")))
        blueprints = [b for b in world.get_blueprint_library().filter("vehicle.*")
                      if int(b.get_attribute("number_of_wheels")) == 4]
        points = world.get_map().get_spawn_points()
        rng.shuffle(points)
        batch = []
        for point in points[:int(g("vehicles"))]:
            bp = rng.choice(blueprints)
            bp.set_attribute("role_name", "traffic")
            batch.append(carla.command.SpawnActor(bp, point).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, tm.get_port())))
        # Spawn points already taken (by the AV, for one) fail; the rest succeed.
        responses = self.client.apply_batch_sync(batch, False)
        self.ids = [r.actor_id for r in responses if not r.error]
        self.get_logger().info(f"spawned {len(self.ids)} vehicles on autopilot "
                               f"({len(batch) - len(self.ids)} spawn points were taken)")

    def cleanup(self) -> None:
        if self.ids:
            self.client.apply_batch([carla.command.DestroyActor(i) for i in self.ids])
            time.sleep(0.5)
            print(f"removed {len(self.ids)} vehicles", flush=True)
            self.ids = []


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SpawnTraffic()
    try:
        spin(node)
    finally:
        node.cleanup()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
