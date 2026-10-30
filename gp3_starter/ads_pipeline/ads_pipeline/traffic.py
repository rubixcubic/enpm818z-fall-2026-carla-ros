"""Traffic for GP3 drives: the same seed gives the same traffic. Given.

The same steps as GP2's tools/record_gp2.py: four-wheeled vehicles on
autopilot on the other spawn points, and pedestrians on the sidewalks, each
with an AI controller.
"""
import random

import carla


def spawn(client, world, tm_port, vehicles=80, walkers=150, seed=818,
          exclude_index=None):
    """Spawn the traffic and return every actor, controllers included."""
    random.seed(seed)
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)
    bp = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()

    actors = []
    cars = [b for b in bp.filter('vehicle.*')
            if int(b.get_attribute('number_of_wheels')) == 4]
    free = [s for i, s in enumerate(spawns) if i != exclude_index]
    random.shuffle(free)
    for s in free[:vehicles]:
        v = world.try_spawn_actor(random.choice(cars), s)
        if v is not None:
            v.set_autopilot(True, tm_port)
            actors.append(v)

    peds, controllers = [], []
    ped_bps = bp.filter('walker.pedestrian.*')
    ctrl_bp = bp.find('controller.ai.walker')
    for _ in range(walkers):
        loc = world.get_random_location_from_navigation()
        if loc is None:
            continue
        w = world.try_spawn_actor(random.choice(ped_bps), carla.Transform(loc))
        if w is not None:
            peds.append(w)
    world.tick()
    for w in peds:
        controllers.append(
            world.spawn_actor(ctrl_bp, carla.Transform(), attach_to=w))
    world.tick()
    for c in controllers:
        c.start()
        c.go_to_location(world.get_random_location_from_navigation())
        c.set_max_speed(1.4)
    return actors + peds + controllers
