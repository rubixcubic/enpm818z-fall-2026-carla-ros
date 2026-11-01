"""Scores one GP4 closed-loop run. Given.

    python3 tools/evaluate_run.py --bag results/gp4/run_X [--out results/gp4/metrics.json]

From the run's bag, all from CARLA's ground truth:
- route completion: 1 when the car came within 2.0 m of the goal; else
  1 minus the road distance left to the goal (on the map's graph, from
  where the car ended) over the first route's length, so a detour counts;
- collisions: one per other actor, events within 1 s of each other counted once;
- pedestrian hits: CARLA's collision sensor misses people, so a hit is a
  person whose center enters the car's outline (2.5 m ahead and behind,
  1.1 m to each side) while the car moves faster than 0.5 m/s; one per person;
- red-light violations: the car crosses a stop line (the signed distance to
  it goes from 0 or more to below 0) while the light is red and the car
  moves faster than 1 m/s;
- cross-track error: the car's distance to the current planned path, mean
  and max; and the mean speed.
Everything is counted up to the moment the car first comes within 2.0 m of
the goal: a car parked at its goal is no longer driving.
"""
import argparse
import json
import math
import os

import numpy as np


def pedestrian_hits(frames, half_length=2.5, half_width=1.1, min_speed=0.5):
    """[(person ID, car speed at the first contact)], one per person.
    frames: [((x, y), yaw, speed, [(person ID, (x, y))])], map frame."""
    hits = {}
    for (cx, cy), yaw, v, people in frames:
        if v <= min_speed:
            continue
        c, s = math.cos(yaw), math.sin(yaw)
        for pid, (px, py) in people:
            dx, dy = px - cx, py - cy
            if (pid not in hits and abs(c * dx + s * dy) <= half_length
                    and abs(-s * dx + c * dy) <= half_width):
                hits[pid] = v
    return list(hits.items())


def completion_by_distance_left(arrived, left_m, route_m):
    """1 at arrival; else the share of the first route's length no longer
    left to drive (the road distance from where the car ended to the goal),
    so a detour counts as progress."""
    if arrived:
        return 1.0
    return float(min(1.0, max(0.0, 1.0 - left_m / route_m)))


def arrival_index(route_xy, driven_xy, radius=2.0):
    """Index of the first driven point within radius of the route's goal;
    the last index when the car never arrived. A run is scored up to there:
    what happens to a car parked at its goal is not part of the drive."""
    goal = np.asarray(route_xy)[-1]
    d = np.hypot(*(np.asarray(driven_xy) - goal).T)
    hit = np.flatnonzero(d < radius)
    return int(hit[0]) if len(hit) else len(d) - 1


def red_light_violations(samples):
    """samples: [(t, state, signed distance to the stop line, speed)] in
    time order; the distance is nan when no light applies."""
    count, prev = 0, float('nan')
    for _, state, dist, speed in samples:
        if prev >= 0.0 > dist and state == 'Red' and speed > 1.0:
            count += 1
        prev = dist
    return count


def lights_on_sim_clock(records):
    """[(simulated t, state, signed distance to the stop line, speed)] from
    [(recording time, 'ego' or 'light', payload)] in recording order. The
    light message is a String with no stamp, so it takes the simulated time
    and the speed of the car message recorded just before it."""
    from ads_pipeline.ground_truth import parse_light
    out, car = [], None
    for _, kind, payload in records:
        if kind == 'ego':
            car = payload
        elif car is not None:
            state, _, dist = parse_light(payload)
            out.append((car[0], state, dist, car[1]))
    return out


def read_light_records(bag_dir):
    """The car and light messages of a bag with their recording times."""
    import rosbag2_py
    from nav_msgs.msg import Odometry
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import String
    from tools.bag_io import _storage_id
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id=_storage_id(bag_dir)),
           rosbag2_py.ConverterOptions('', ''))
    r.set_filter(rosbag2_py.StorageFilter(topics=['/ground_truth/ego',
                                                  '/ground_truth/traffic_light']))
    out = []
    while r.has_next():
        topic, data, t_rec = r.read_next()
        if topic == '/ground_truth/ego':
            m = deserialize_message(data, Odometry)
            out.append((t_rec, 'ego', (m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                                       math.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y))))
        else:
            out.append((t_rec, 'light', deserialize_message(data, String).data))
    out.sort(key=lambda x: x[0])
    return out


def collision_count(events, window=1.0):
    """events: [(t, other actor type)] in time order."""
    last, count = {}, 0
    for t, actor in events:
        if actor not in last or t - last[actor] > window:
            count += 1
        last[actor] = t
    return count


def distance_left(map_file, from_xy, goal_xy):
    """Road distance (m) from a point to the goal on the map's graph."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    from ads_pipeline.route_graph import RouteGraph
    g = RouteGraph.from_xodr(map_file)
    n = len(g.xy)
    rows, cols, w = zip(*[(i, j, c) for i in range(n) for j, c in g.neighbors(i)])
    dist = dijkstra(csr_matrix((w, (rows, cols)), shape=(n, n)), indices=g.nearest(*from_xy))
    return float(dist[g.nearest(*goal_xy)])


def main():
    from ads_pipeline.bicycle_sim import cross_track
    from ads_pipeline.ground_truth import parse_light
    from tools import bag_io
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--out', default='results/gp4/metrics.json')
    ap.add_argument('--map-file', default='ads_pipeline/maps/Town01.xodr')
    a = ap.parse_args()
    d = bag_io.read(a.bag, ['/ground_truth/ego',
                            '/ground_truth/collision', '/planning/path',
                            '/ground_truth/objects'])
    ego = d['/ground_truth/ego']
    driven = np.array([[m.pose.pose.position.x, m.pose.pose.position.y] for _, m in ego])
    speed = np.array([math.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y) for _, m in ego])
    paths = [(t, np.array([[q.pose.position.x, q.pose.position.y] for q in m.poses]))
             for t, m in d['/planning/path']]
    paths = [(t, p) for t, p in paths if len(p) >= 2]
    route = paths[0][1] if paths else np.zeros((0, 2))
    arrived = False
    if len(route):                          # score the drive up to arrival
        k = arrival_index(route, driven)
        arrived = math.dist(driven[k], route[-1]) < 2.0
        t_end = ego[k][0]
        ego, driven, speed = ego[:k + 1], driven[:k + 1], speed[:k + 1]
        paths = [(t, p) for t, p in paths if t <= t_end]
    else:
        t_end = float('inf')
    lights = [x for x in lights_on_sim_clock(read_light_records(a.bag)) if x[0] <= t_end]
    events = [(float(m.data.split()[0]), m.data.split()[1]) for _, m in d['/ground_truth/collision']]
    events = [e for e in events if e[0] <= t_end]     # CARLA's clock, as the stamps
    people = {round(t, 3): [(o.id, (o.bbox.center.position.x, o.bbox.center.position.y))
                            for o in m.detections
                            if o.results and o.results[0].hypothesis.class_id == 'pedestrian']
              for t, m in d['/ground_truth/objects']}
    frames = []
    for (t, m), v in zip(ego, speed):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        frames.append(((m.pose.pose.position.x, m.pose.pose.position.y), yaw, v,
                       people.get(round(t, 3), [])))
    ctes, j = [], 0
    for (t, _), (x, y) in zip(ego, driven):
        while j + 1 < len(paths) and paths[j + 1][0] <= t:
            j += 1
        if paths:
            ctes.append(cross_track(paths[j][1], x, y))
    out = {
        'route_completion': (completion_by_distance_left(
            arrived, distance_left(a.map_file, driven[-1], route[-1]),
            float(np.hypot(*np.diff(route, axis=0).T).sum())) if len(route) else 0.0),
        'collisions': collision_count(events),
        'pedestrian_hits': len(pedestrian_hits(frames)),
        'red_light_violations': red_light_violations(lights),
        'cross_track_mean_m': float(np.mean(ctes)) if ctes else float('nan'),
        'cross_track_max_m': float(np.max(ctes)) if ctes else float('nan'),
        'mean_speed_mps': float(speed.mean()) if len(speed) else float('nan'),
        'replans': max(0, len(paths) - 1),
        'duration_s': float(ego[-1][0] - ego[0][0]) if ego else 0.0,
    }
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    for k, v in out.items():
        print(f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}')


if __name__ == '__main__':
    main()
