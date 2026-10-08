"""Grading a segmentation: the confusion matrix, IoU per class, mIoU.

For one class c, over every graded pixel of every frame:

    TP  truth c, network c        (true positive)
    FP  truth not c, network c    (false positive)
    FN  truth c, network not c    (false negative)

    IoU(c) = TP / (TP + FP + FN)

the pixels both call c, divided by the pixels either calls c. mIoU is the mean
of IoU(c) over the classes that appear in the truth during the run: a class
CARLA never showed cannot be graded. (Cityscapes averages over all 19 classes
because its test set has all of them.) The counts are summed over frames
first, then divided, as the Cityscapes benchmark does: a class seen in a few
pixels of one frame does not weigh as much as one frame's full IoU.
"""

from __future__ import annotations

import numpy as np

from l5_seg_demo.classes import CITYSCAPES, IGNORE


class ConfusionMatrix:

    def __init__(self, n_classes: int = len(CITYSCAPES)) -> None:
        self.n = n_classes
        # m[t, p]: pixels whose truth is t and whose prediction is p.
        self.m = np.zeros((n_classes, n_classes), dtype=np.int64)
        self.frames = 0

    def add(self, truth: np.ndarray, pred: np.ndarray) -> None:
        """Count one frame. Both are class numbers 0 to n-1; truth may be IGNORE."""
        if truth.shape != pred.shape:
            raise ValueError(f"truth {truth.shape} and prediction {pred.shape} differ")
        keep = (truth != IGNORE) & (pred != IGNORE)
        idx = truth[keep].astype(np.int64) * self.n + pred[keep].astype(np.int64)
        self.m += np.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)
        self.frames += 1

    def counts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tp = np.diag(self.m).copy()
        fp = self.m.sum(axis=0) - tp
        fn = self.m.sum(axis=1) - tp
        return tp, fp, fn

    def iou(self) -> np.ndarray:
        """IoU per class; NaN for a class with no truth pixel."""
        tp, fp, fn = self.counts()
        out = np.full(self.n, np.nan)
        seen = (tp + fn) > 0
        out[seen] = tp[seen] / (tp + fp + fn)[seen]
        return out

    def miou(self) -> float:
        iou = self.iou()
        return float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else float("nan")

    def pixel_accuracy(self) -> float:
        total = self.m.sum()
        return float(np.trace(self.m) / total) if total else float("nan")

    def report(self, names=CITYSCAPES) -> dict:
        tp, fp, fn = self.counts()
        iou = self.iou()
        total = self.m.sum()
        classes = {}
        for c in range(self.n):
            if tp[c] + fn[c] == 0:
                continue
            row = self.m[c] / self.m[c].sum()          # what the network called class c
            top = [k for k in np.argsort(-row)[:2] if row[k] > 0]
            classes[names[c]] = {
                "iou": round(float(iou[c]), 4),
                "truth_share": round(float((tp[c] + fn[c]) / total), 5),
                "tp": int(tp[c]), "fp": int(fp[c]), "fn": int(fn[c]),
                "network_called_it": {names[k]: round(float(row[k]), 4) for k in top},
            }
        return {"frames": self.frames, "graded_pixels": int(total),
                "miou": round(self.miou(), 4),
                "pixel_accuracy": round(self.pixel_accuracy(), 4),
                "classes": classes}
