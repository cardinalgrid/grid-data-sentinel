"""Detection metrics against a 0/1 label vector."""

from __future__ import annotations

import numpy as np


def _dilate(x: np.ndarray, tolerance: int) -> np.ndarray:
    out = x.copy()
    for d in range(1, tolerance + 1):
        out[d:] |= x[:-d]
        out[:-d] |= x[d:]
    return out


def score_labels(label: np.ndarray, flags: np.ndarray, tolerance: int = 0) -> dict[str, float]:
    """Precision, recall, F1 and Matthews correlation coefficient.

    With ``tolerance`` > 0 a flag within that many readings of a labelled reading is a true
    positive, and a labelled reading with a flag within that distance counts as found. This is
    useful for detectors that fire on the edge of a run rather than on every reading in it.
    """
    y = np.asarray(label, dtype=bool)
    p = np.asarray(flags, dtype=bool)
    if tolerance > 0:
        y_wide = _dilate(y, tolerance)
        p_wide = _dilate(p, tolerance)
        tp = int((p & y_wide).sum())
        fp = int((p & ~y_wide).sum())
        fn = int((y & ~p_wide).sum())
        tn = int((~p & ~y).sum())
    else:
        tp = int((p & y).sum())
        fp = int((p & ~y).sum())
        fn = int((~p & y).sum())
        tn = int((~p & ~y).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": float(mcc),
        "flagged": int(p.sum()),
        "labelled": int(y.sum()),
    }
