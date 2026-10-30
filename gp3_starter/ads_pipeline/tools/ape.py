"""APE and the NIS check for GP3 Task 2. Given.

    python3 tools/ape.py --bag results/gp3/run_X [--outage 30 50] [--nis results/gp3/nis.csv]

APE compares the EKF's /localization/pose with /ground_truth/ego, both the
car's origin, at the same stamps. The outage window is in seconds from the
first GNSS fix, as gnss_outage counts it. "With GNSS" leaves out the outage
and the settle seconds after it, while the first fixes remove the error
the outage left.
"""
import argparse
import csv
import os

import numpy as np

# The NIS band for a fix with 2 values, from l3_ekf_demo/config/ekf.yaml:
NIS_LO, NIS_HI, NIS_DOF = 0.051, 7.378, 2


def match_by_stamp(est, ref, tol=0.026):
    """Pairs (t, est_xy, ref_xy) for each estimate with a reference within
    tol seconds. 0.026 s is just over half the 0.05 s tick."""
    if not ref:
        return []
    ts = np.array([t for t, _ in ref])
    pairs = []
    for t, e in est:
        i = int(np.argmin(np.abs(ts - t)))
        if abs(ts[i] - t) <= tol:
            pairs.append((t, np.asarray(e), np.asarray(ref[i][1])))
    return pairs


def ape_rmse(pairs, window=None, outside=False):
    """RMSE of the position error over the pairs, or only those whose
    time is in [window[0], window[1]), or, with outside, only the others."""
    errs = [np.linalg.norm(e - r) for t, e, r in pairs
            if window is None or (window[0] <= t < window[1]) != outside]
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else float('nan')


def nis_verdict(values, min_inside=85.0):
    """The NIS verdict of l3_ekf_demo/ekf.py: consistent if at least
    min_inside percent of the values are inside the band, else the mean
    says which way the filter is wrong."""
    a = np.asarray(values, dtype=float)
    inside = float(np.mean((a >= NIS_LO) & (a <= NIS_HI))) * 100.0
    mean = float(a.mean())
    if inside >= min_inside:
        verdict = 'consistent'
    elif mean > NIS_DOF:
        verdict = 'overconfident'
    else:
        verdict = 'underconfident'
    return verdict, mean, inside


def _xy(msg):
    p = msg.pose.pose.position if hasattr(msg.pose, 'pose') else msg.pose.position
    return (p.x, p.y)


def main():
    from tools import bag_io
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--outage', nargs=2, type=float, metavar=('START', 'END'))
    ap.add_argument('--settle', type=float, default=3.0,
                    help='seconds after the outage left out of the with-GNSS APE: '
                         'the first fixes still carry the outage error')
    ap.add_argument('--nis')
    ap.add_argument('--csv', default='results/gp3/ape.csv')
    a = ap.parse_args()
    d = bag_io.read(a.bag, ['/localization/pose', '/ground_truth/ego', '/carla/gnss/fix'])
    est = [(t, _xy(m)) for t, m in d['/localization/pose']]
    ref = [(t, _xy(m)) for t, m in d['/ground_truth/ego']]
    pairs = match_by_stamp(est, ref)
    print(f'pairs: {len(pairs)}')
    print(f'APE RMSE all: {ape_rmse(pairs):.3f} m')
    if a.outage:
        t0 = d['/carla/gnss/fix'][0][0]
        w = (t0 + a.outage[0], t0 + a.outage[1])
        print(f'APE RMSE outage: {ape_rmse(pairs, w):.3f} m')
        w_settle = (w[0], w[1] + a.settle)
        print(f'APE RMSE with GNSS (outside the outage and the {a.settle:g} s after it): '
              f'{ape_rmse(pairs, w_settle, outside=True):.3f} m')
    os.makedirs(os.path.dirname(a.csv) or '.', exist_ok=True)
    with open(a.csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['t', 'est_x', 'est_y', 'ref_x', 'ref_y', 'err'])
        for t, e, r in pairs:
            w.writerow([f'{t:.3f}', *e, *r, np.linalg.norm(e - r)])
    if a.nis:
        with open(a.nis) as fh:
            vals = [float(r['nis']) for r in csv.DictReader(fh)]
        v, mean, inside = nis_verdict(vals)
        print(f'NIS: {v}, mean {mean:.2f} (an honest filter averages {NIS_DOF}), '
              f'{inside:.1f}% inside [{NIS_LO}, {NIS_HI}]')


if __name__ == '__main__':
    main()
