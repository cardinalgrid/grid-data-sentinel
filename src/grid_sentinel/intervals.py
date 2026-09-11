"""Anomaly intervals: runs of flagged readings with a duration class.

Working with intervals rather than individual readings has two uses. Repair can choose its method
by duration (a straight line for one hour, equivalent days for a day-long gap, nothing for a month),
and a table of intervals per class is a compact description of the quality of a series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = (
    ("up to 1 h", 1),
    ("up to 1 day", 24),
    ("up to 1 week", 24 * 7),
    ("up to 1 month", 24 * 30),
    ("over 1 month", np.inf),
)


def duration_class(n_hours: float) -> str:
    for name, limit in CLASSES:
        if n_hours <= limit:
            return name
    return CLASSES[-1][0]


def intervals_from_mask(mask: np.ndarray | pd.Series, index: pd.Index | None = None, samples_per_hour: float = 1.0,
                        reason: str | np.ndarray | None = None) -> pd.DataFrame:
    """Group consecutive flagged readings into intervals.

    Returns a frame with ``start`` and ``end`` (positions, or index labels when ``index`` is given),
    ``i0``/``i1`` positions (inclusive), ``n`` readings, ``hours`` and ``duration_class``. When ``reason`` is
    an array of per-reading labels, the most frequent label inside each interval is reported.
    """
    m = np.asarray(mask, dtype=bool)
    if index is None and isinstance(mask, pd.Series):
        index = mask.index
    if m.size == 0 or not m.any():
        return pd.DataFrame(columns=["start", "end", "i0", "i1", "n", "hours", "duration_class", "reason"])
    edges = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    rows = []
    for i0, i1 in zip(starts, ends):
        n = int(i1 - i0 + 1)
        hours = n / samples_per_hour
        r = ""
        if reason is not None:
            if isinstance(reason, str):
                r = reason
            else:
                vals, counts = np.unique(np.asarray(reason)[i0 : i1 + 1], return_counts=True)
                r = str(vals[np.argmax(counts)])
        rows.append({
            "start": index[i0] if index is not None else int(i0),
            "end": index[i1] if index is not None else int(i1),
            "i0": int(i0), "i1": int(i1), "n": n, "hours": hours, "duration_class": duration_class(hours), "reason": r,
        })
    return pd.DataFrame(rows)


def summarize_intervals(iv: pd.DataFrame) -> pd.DataFrame:
    """Count and total hours per duration class, in class order."""
    order = [c for c, _ in CLASSES]
    if iv.empty:
        return pd.DataFrame({"duration_class": order, "intervals": 0, "hours": 0.0})
    g = iv.groupby("duration_class").agg(intervals=("n", "size"), hours=("hours", "sum"))
    return g.reindex(order).fillna(0).astype({"intervals": int}).reset_index()
