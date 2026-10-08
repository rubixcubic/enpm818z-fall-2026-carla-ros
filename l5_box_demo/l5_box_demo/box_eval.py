"""Grading boxes against true boxes, the nuScenes way (no ROS here).

From the nuScenes devkit (python-sdk/nuscenes/eval, commit b40adc4):
  - a detection matches a true object when their centers are within a
    distance on the ground (center_distance, x and y only); the thresholds
    are 0.5, 1, 2 and 4 m (dist_ths), and the errors of matched boxes are
    measured at 2 m (dist_th_tp);
  - size error, ASE = 1 - scale IoU: the 3D IoU of the two boxes after
    aligning their centers and headings (scale_iou);
  - heading error, AOE: the smallest angle between the two headings. The
    devkit takes it modulo 360 degrees (180 for barriers); a box fitted to one
    LiDAR sweep has no front or back, so here it is modulo 180 degrees.
The devkit matches by confidence, highest first; these boxes have none, so
pairs are matched closest first, one detection per object.
"""

from __future__ import annotations

import math

import numpy as np

from l5_box_demo.lshape import heading_error_mod_half_turn

THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
TP_THRESHOLD = 2.0


def scale_iou(size_a, size_b) -> float:
    a, b = np.asarray(size_a, float), np.asarray(size_b, float)
    inter = float(np.prod(np.minimum(a, b)))
    return inter / (float(np.prod(a)) + float(np.prod(b)) - inter)


def greedy_pairs(truth_xy: np.ndarray, det_xy: np.ndarray, max_dist: float):
    """Closest pairs first, each object and each detection used once."""
    if len(truth_xy) == 0 or len(det_xy) == 0:
        return []
    d = np.linalg.norm(truth_xy[:, None, :] - det_xy[None, :, :], axis=2)
    pairs, used_t, used_d = [], set(), set()
    for flat in np.argsort(d, axis=None):
        r, c = divmod(int(flat), d.shape[1])
        if d[r, c] > max_dist:
            break
        if r in used_t or c in used_d:
            continue
        used_t.add(r); used_d.add(c)
        pairs.append((r, c, float(d[r, c])))
    return pairs


class BoxScore:
    """Accumulates one run. truth: dicts with x, y, yaw, size (l, w, h), visible.
    boxes: dicts with x, y, yaw, size, and centroid (x, y) of their cluster."""

    def __init__(self) -> None:
        self.frames = 0
        self.n_truth = 0
        self.n_visible = 0
        self.n_boxes = 0
        self.tp = {t: 0 for t in THRESHOLDS}
        self.tp_visible = {t: 0 for t in THRESHOLDS}
        self.ate, self.ase, self.aoe, self.centroid_err = [], [], [], []
        self.len_err, self.wid_err = [], []

    def add(self, truth: list[dict], boxes: list[dict]) -> None:
        self.frames += 1
        self.n_truth += len(truth)
        self.n_visible += sum(1 for o in truth if o["visible"])
        self.n_boxes += len(boxes)
        txy = np.array([[o["x"], o["y"]] for o in truth]).reshape(-1, 2)
        bxy = np.array([[b["x"], b["y"]] for b in boxes]).reshape(-1, 2)
        for th in THRESHOLDS:
            pairs = greedy_pairs(txy, bxy, th)
            self.tp[th] += len(pairs)
            self.tp_visible[th] += sum(1 for r, _, _ in pairs if truth[r]["visible"])
            if th == TP_THRESHOLD:
                for r, c, dist in pairs:
                    o, b = truth[r], boxes[c]
                    self.ate.append(dist)
                    self.ase.append(1.0 - scale_iou(o["size"], b["size"]))
                    self.aoe.append(math.degrees(heading_error_mod_half_turn(o["yaw"], b["yaw"])))
                    self.centroid_err.append(math.hypot(b["centroid"][0] - o["x"],
                                                        b["centroid"][1] - o["y"]))
                    self.len_err.append(b["size"][0] - o["size"][0])
                    self.wid_err.append(b["size"][1] - o["size"][1])

    def summary(self) -> dict:
        r = lambda a, b: round(a / b, 3) if b else None
        mean = lambda v: round(float(np.mean(v)), 3) if v else None
        median = lambda v: round(float(np.median(v)), 3) if v else None
        return {
            "frames": self.frames,
            "true_vehicles_per_frame": r(self.n_truth, self.frames),
            "visible_vehicles_per_frame": r(self.n_visible, self.frames),
            "boxes_per_frame": r(self.n_boxes, self.frames),
            "precision": {str(t): r(self.tp[t], self.n_boxes) for t in THRESHOLDS},
            "recall_all": {str(t): r(self.tp[t], self.n_truth) for t in THRESHOLDS},
            "recall_visible": {str(t): r(self.tp_visible[t], self.n_visible) for t in THRESHOLDS},
            "matched_at_2m": len(self.ate),
            "center_error_m": {"mean": mean(self.ate), "median": median(self.ate)},
            "centroid_error_m": {"mean": mean(self.centroid_err),
                                 "median": median(self.centroid_err)},
            "size_error_ase": mean(self.ase),
            "length_error_m_mean": mean(self.len_err),
            "width_error_m_mean": mean(self.wid_err),
            "heading_error_deg_mod180": {"mean": mean(self.aoe), "median": median(self.aoe)},
        }
