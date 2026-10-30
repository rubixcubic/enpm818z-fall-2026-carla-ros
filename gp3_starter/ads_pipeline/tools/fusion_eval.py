"""Fused objects against the true boxes, for GP3 Task 3. Given.

    python3 tools/fusion_eval.py --bag results/gp3/run_X [--match-radius 2.0]

Per stamp and per class, each fused object is matched to at most one true
box (Hungarian on center distance, at most match_radius apart). Prints, per
class, the mean position error of the matches, overall and by distance from
the car (0 to 20, 20 to 40, 40 to 80 m), and the share of true boxes
matched. Distances are on the ground (x, y). Only true objects the sensors
can see count: within the camera's horizontal field of view and max_range
of it (in_view), with at least MIN_POINTS LiDAR points of the same sweep
inside the box (visible_ids, as nuScenes drops boxes with no LiDAR points).
The match radius is 3 m: the LiDAR sees a car's near side, so the cluster's
center falls short of the box's center, by up to half the car's length.
"""
import argparse
import math
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

BANDS = (('0-20', 0.0, 20.0), ('20-40', 20.0, 40.0), ('40-80', 40.0, 80.0))
CLASSES = ('vehicle', 'pedestrian')
OLD_NAMES = {'car': 'vehicle', 'person': 'pedestrian'}   # first test bag
MIN_POINTS = 5            # = fusion's DBSCAN minimum
BOX_MARGIN = 0.5          # m: a sweep takes 0.05 s, the car moves up to 0.4 m in it


def in_view(xy, ego_xy, ego_yaw, cam_xy, fov_deg, max_range):
    """True when a map point is inside the camera's horizontal field of view
    and within max_range of it. cam_xy: the camera's (x, y) on the car
    (x forward, y left); the camera looks straight ahead."""
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    cam = ego_xy + np.array([c * cam_xy[0] - s * cam_xy[1], s * cam_xy[0] + c * cam_xy[1]])
    d = np.asarray(xy) - cam
    bearing = (math.atan2(d[1], d[0]) - ego_yaw + math.pi) % (2 * math.pi) - math.pi
    return abs(bearing) <= math.radians(fov_deg) / 2 and np.hypot(*d) <= max_range


def points_in_box(pts, center, size, yaw, margin=0.0):
    """How many (N, 3) map points fall inside a box turned by yaw, grown by
    margin (m) in length and width."""
    d = np.asarray(pts) - np.asarray(center)
    c, s = np.cos(yaw), np.sin(yaw)
    lx = c * d[:, 0] + s * d[:, 1]
    ly = -s * d[:, 0] + c * d[:, 1]
    return int(np.sum((np.abs(lx) <= (size[0] + margin) / 2)
                      & (np.abs(ly) <= (size[1] + margin) / 2)
                      & (np.abs(d[:, 2]) <= size[2] / 2)))


def visible_ids(drive_bag, rig):
    """{stamp: set of CARLA IDs with at least MIN_POINTS LiDAR points in
    their box}, from the drive bag (LiDAR, true pose, true boxes)."""
    import yaml
    import sensor_msgs_py.point_cloud2 as pc2
    from tools import bag_io
    m = yaml.safe_load(open(rig))['sensor_manager']['ros__parameters']['mounts']['lidar_top']
    mount = np.array([m['x'], -m['y'], m['z']])
    d = bag_io.read(drive_bag, ['/carla/lidar/points', '/ground_truth/ego',
                                '/ground_truth/objects'])
    ego = {round(t, 3): e for t, e in d['/ground_truth/ego']}
    boxes = {round(t, 3): b for t, b in d['/ground_truth/objects']}
    out = {}
    for t, cloud in d['/carla/lidar/points']:
        k = round(t, 3)
        if k not in ego or k not in boxes:
            continue
        p, yaw = ego[k].pose.pose.position, _yaw(ego[k].pose.pose.orientation)
        pts = pc2.read_points_numpy(cloud, field_names=('x', 'y', 'z'), skip_nans=True) + mount
        c, s = math.cos(yaw), math.sin(yaw)
        world = np.c_[p.x + c * pts[:, 0] - s * pts[:, 1],
                      p.y + s * pts[:, 0] + c * pts[:, 1], p.z + pts[:, 2]]
        seen = set()
        for b in boxes[k].detections:
            bc, bs = b.bbox.center.position, b.bbox.size
            near = world[np.hypot(world[:, 0] - bc.x, world[:, 1] - bc.y) < 6.0]
            if points_in_box(near, (bc.x, bc.y, bc.z), (bs.x, bs.y, bs.z),
                             _yaw(b.bbox.center.orientation),
                             margin=BOX_MARGIN) >= MIN_POINTS:
                seen.add(b.id)
        out[k] = seen
    return out


def _band(d):
    for name, lo, hi in BANDS:
        if lo <= d < hi:
            return name
    return None


def score_frames(frames, match_radius):
    """frames: [(ego_xy, fused, gt)], fused and gt lists of {'xy', 'cls'}."""
    acc = {c: {'matched': 0, 'gt': 0, 'false': 0, 'err': [],
               'band': defaultdict(lambda: {'matched': 0, 'gt': 0, 'err': []})}
           for c in CLASSES}
    for ego, fused, gt in frames:
        for c in CLASSES:
            f = [o['xy'] for o in fused if o['cls'] == c]
            g = [o['xy'] for o in gt if o['cls'] == c]
            a = acc[c]
            a['gt'] += len(g)
            for gx in g:
                b = _band(np.linalg.norm(gx - ego))
                if b:
                    a['band'][b]['gt'] += 1
            if not f or not g:
                a['false'] += len(f)
                continue
            C = np.linalg.norm(np.array(f)[:, None, :] - np.array(g)[None, :, :], axis=2)
            rows, cols = linear_sum_assignment(C)
            ok = [(r, k) for r, k in zip(rows, cols) if C[r, k] <= match_radius]
            a['matched'] += len(ok)
            a['false'] += len(f) - len(ok)
            for r, k in ok:
                a['err'].append(C[r, k])
                b = _band(np.linalg.norm(g[k] - ego))
                if b:
                    a['band'][b]['matched'] += 1
                    a['band'][b]['err'].append(C[r, k])
    out = {}
    for c, a in acc.items():
        out[c] = {
            'matched': a['matched'], 'gt': a['gt'], 'false': a['false'],
            'mean_error': float(np.mean(a['err'])) if a['err'] else float('nan'),
            'match_rate': a['matched'] / a['gt'] if a['gt'] else float('nan'),
            'by_band': {n: {'matched': a['band'][n]['matched'], 'gt': a['band'][n]['gt'],
                            'mean_error': float(np.mean(a['band'][n]['err']))
                            if a['band'][n]['err'] else float('nan')}
                        for n, _, _ in BANDS},
        }
    return out


def _yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def view_from_rig(path, max_range):
    """(cam_xy, fov_deg, max_range) from the GP1 YAML; y flipped to the left."""
    import yaml
    p = yaml.safe_load(open(path))['sensor_manager']['ros__parameters']
    m = p['mounts']['camera_rgb_front']
    return (m['x'], -m['y']), float(p['camera']['fov']), max_range


def _objects(msg):
    out = []
    for d in msg.detections:
        cls = d.results[0].hypothesis.class_id if d.results else ''
        p = d.bbox.center.position
        out.append({'xy': np.array([p.x, p.y]), 'cls': OLD_NAMES.get(cls, cls), 'id': d.id})
    return out


def main():
    from tools import bag_io
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--match-radius', type=float, default=3.0)
    ap.add_argument('--rig', required=True, help='your GP1 YAML (camera mount and FOV)')
    ap.add_argument('--drive', required=True, help='your drive bag (Task 1)')
    ap.add_argument('--max-range', type=float, default=60.0)
    a = ap.parse_args()
    view = view_from_rig(a.rig, a.max_range)
    vis = visible_ids(a.drive, a.rig)
    d = bag_io.read(a.bag, ['/perception/fused_objects', '/ground_truth/objects',
                            '/ground_truth/ego'])
    gt = {round(t, 3): m for t, m in d['/ground_truth/objects']}
    ego = {round(t, 3): m for t, m in d['/ground_truth/ego']}
    frames = []
    for t, m in d['/perception/fused_objects']:
        k = round(t, 3)
        if k in gt and k in ego:
            p, q = ego[k].pose.pose.position, ego[k].pose.pose.orientation
            e, yaw = np.array([p.x, p.y]), _yaw(q)
            seen = [o for o in _objects(gt[k]) if in_view(o['xy'], e, yaw, *view)
                    and o['id'] in vis.get(k, ())]
            frames.append((e, _objects(m), seen))
    s = score_frames(frames, a.match_radius)
    print(f'frames: {len(frames)}, match radius {a.match_radius} m')
    for c, r in s.items():
        print(f"{c}: matched {r['matched']} of {r['gt']} ({100 * r['match_rate']:.1f}%), "
              f"false {r['false']}, mean error {r['mean_error']:.2f} m")
        for n, b in r['by_band'].items():
            print(f"    {n} m: {b['matched']} of {b['gt']}, mean error {b['mean_error']:.2f} m")


if __name__ == '__main__':
    main()
