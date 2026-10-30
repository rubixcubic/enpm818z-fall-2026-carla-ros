"""Tracks on the map against the truth, for GP3 Task 4. Given.

    python3 tools/track_eval.py --bag results/gp3/run_X [--match-radius 2.0]

GP2's count_id_switches with the box overlap replaced by center distance:
per stamp and class, each true object is matched to at most one track
(Hungarian on distance, at most match_radius apart). An ID switch is an
object whose matched track ID differs from the one it had the last time it
was matched. MOTA = 1 - (missed + false tracks + ID switches) / true objects.
The true speed is the change in the object's center between stamps. Only
true objects the sensors can see count (fusion_eval.in_view and
visible_ids); the match radius is 3 m, as in fusion_eval.
"""
import argparse
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

OLD_NAMES = {'car': 'vehicle', 'person': 'pedestrian'}   # first test bag


def count_on_map(frames, match_radius):
    """frames: [(tracks, gts)]; tracks [{'track_id','cls','xy','v'}],
    gts [{'id','cls','xy','v'}]."""
    last = {}
    out = {'id_switches': 0, 'matched': 0, 'missed': 0, 'false_tracks': 0, 'gt': 0}
    speed_err = []
    for tracks, gts in frames:
        out['gt'] += len(gts)
        for c in {g['cls'] for g in gts} | {t['cls'] for t in tracks}:
            g = [x for x in gts if x['cls'] == c]
            t = [x for x in tracks if x['cls'] == c]
            ok = []
            if g and t:
                C = np.linalg.norm(np.array([x['xy'] for x in g])[:, None, :]
                                   - np.array([x['xy'] for x in t])[None, :, :], axis=2)
                rows, cols = linear_sum_assignment(C)
                ok = [(r, k) for r, k in zip(rows, cols) if C[r, k] <= match_radius]
            out['matched'] += len(ok)
            out['missed'] += len(g) - len(ok)
            out['false_tracks'] += len(t) - len(ok)
            for r, k in ok:
                aid, tid = g[r]['id'], t[k]['track_id']
                if aid in last and last[aid] != tid:
                    out['id_switches'] += 1
                last[aid] = tid
                speed_err.append(float(np.linalg.norm(t[k]['v'] - g[r]['v'])))
    out['mota'] = (1.0 - (out['missed'] + out['false_tracks'] + out['id_switches'])
                   / out['gt']) if out['gt'] else float('nan')
    out['mean_speed_error'] = float(np.mean(speed_err)) if speed_err else float('nan')
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
    from tools.fusion_eval import _yaw, in_view, view_from_rig, visible_ids
    view = view_from_rig(a.rig, a.max_range)
    vis = visible_ids(a.drive, a.rig)
    d = bag_io.read(a.bag, ['/perception/tracks', '/ground_truth/objects',
                            '/ground_truth/ego'])
    ego = {round(t, 3): m for t, m in d['/ground_truth/ego']}
    gt_frames, prev = {}, {}
    for t, m in d['/ground_truth/objects']:
        objs = []
        for det in m.detections:
            p = det.bbox.center.position
            xy = np.array([p.x, p.y])
            v = np.zeros(2)
            if det.id in prev:
                tp, xp = prev[det.id]
                v = (xy - xp) / (t - tp) if t > tp else v
            prev[det.id] = (t, xy)
            cls = det.results[0].hypothesis.class_id
            objs.append({'id': det.id, 'cls': OLD_NAMES.get(cls, cls), 'xy': xy, 'v': v})
        k = round(t, 3)
        if k in ego:
            p, q = ego[k].pose.pose.position, ego[k].pose.pose.orientation
            e, yaw = np.array([p.x, p.y]), _yaw(q)
            objs = [o for o in objs if in_view(o['xy'], e, yaw, *view)
                    and o['id'] in vis.get(k, ())]
        gt_frames[k] = objs
    frames = []
    for t, m in d['/perception/tracks']:
        k = round(t, 3)
        if k not in gt_frames:
            continue
        tracks = [{'track_id': o.id, 'cls': o.class_id,
                   'xy': np.array([o.position.x, o.position.y]),
                   'v': np.array([o.velocity.x, o.velocity.y])} for o in m.objects]
        frames.append((tracks, gt_frames[k]))
    out = count_on_map(frames, a.match_radius)
    print(f'frames: {len(frames)}, match radius {a.match_radius} m')
    for key in ('gt', 'matched', 'missed', 'false_tracks', 'id_switches'):
        print(f'{key}: {out[key]}')
    print(f"MOTA: {out['mota']:.3f}; mean speed error: {out['mean_speed_error']:.2f} m/s")


if __name__ == '__main__':
    main()
