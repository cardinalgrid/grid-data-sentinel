"""Repair of flagged readings with an audit trail, and anomaly intervals with a duration class.

Detection never modifies a series. ``Repairer.transform`` takes the readings and the flags, returns the
corrected readings and an ``Audit`` that records, reading by reading, the original value, the new value,
the method and the reason. Intervals longer than ``max_gap_hours`` are left open (NaN) rather than
invented. No pandas here; ``Audit.to_pandas`` imports it lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from grid_sentinel.types import as_series
from grid_sentinel.types import intervals_from_mask as _runs

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


def intervals_from_mask(mask: np.ndarray) -> np.ndarray:
    """Runs of flagged readings: structured array with ``i0``, ``i1`` (inclusive), ``n`` and ``duration_class``."""
    runs = _runs(mask)
    dtype = np.dtype([("i0", "int64"), ("i1", "int64"), ("n", "int64"), ("duration_class", "<U16")])
    out = np.empty(len(runs), dtype=dtype)
    out["i0"], out["i1"], out["n"] = runs["i0"], runs["i1"], runs["n"]
    out["duration_class"] = [duration_class(int(n)) for n in runs["n"]]
    return out


def summarize_intervals(iv: np.ndarray) -> dict[str, dict[str, int]]:
    """Per duration class, in class order: number of intervals and total hours."""
    out = {name: {"intervals": 0, "hours": 0} for name, _ in CLASSES}
    for row in iv:
        out[str(row["duration_class"])]["intervals"] += 1
        out[str(row["duration_class"])]["hours"] += int(row["n"])
    return out


@dataclass
class Audit:
    """One row per changed reading."""

    index: np.ndarray
    timestamp: np.ndarray | None
    original: np.ndarray
    repaired: np.ndarray
    method: np.ndarray
    reason: np.ndarray

    def __len__(self) -> int:
        return len(self.index)

    def to_pandas(self):
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover
            raise ImportError("Audit.to_pandas needs pandas: pip install grid-data-sentinel[pandas]") from e
        idx = pd.DatetimeIndex(self.timestamp) if self.timestamp is not None else pd.Index(self.index)
        return pd.DataFrame({"original": self.original, "repaired": self.repaired, "method": self.method,
                             "reason": self.reason}, index=idx)


def _linear(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linear interpolation across flagged readings, only between two good readings (no extrapolation)."""
    out = values.copy()
    out[mask] = np.nan
    good = np.isfinite(out)
    if good.sum() < 2:
        return out
    pos = np.arange(len(out))
    filled = np.interp(pos, pos[good], out[good])
    first, last = pos[good][0], pos[good][-1]
    inside = (pos >= first) & (pos <= last)
    out[~good & inside] = filled[~good & inside]
    return out


def _equivalent_days(values: np.ndarray, mask: np.ndarray, i0: int, i1: int, offsets_days: tuple[int, ...],
                     samples_per_day: int) -> np.ndarray | None:
    """Mean of the same interval ``d`` days before and after, over the first offset for which at least one
    side exists and contains no flagged reading, shifted linearly so that its ends meet the neighbouring
    good readings."""
    n = i1 - i0 + 1
    for d in offsets_days:
        step = d * samples_per_day
        sides = []
        for sign in (-1, 1):
            a, b = i0 + sign * step, i1 + sign * step
            if a < 0 or b >= len(values):
                continue
            seg = values[a : b + 1]
            if np.isnan(seg).any() or mask[a : b + 1].any():
                continue
            sides.append(seg)
        if sides:
            base = np.mean(sides, axis=0)
            before = values[i0 - 1] if i0 > 0 and not mask[i0 - 1] and np.isfinite(values[i0 - 1]) else np.nan
            after = values[i1 + 1] if i1 + 1 < len(values) and not mask[i1 + 1] and np.isfinite(values[i1 + 1]) else np.nan
            if np.isfinite(before) and np.isfinite(after):
                w = np.linspace(0, 1, n + 2)[1:-1]
                return base + (1 - w) * (before - base[0]) + w * (after - base[-1])
            if np.isfinite(before):
                return base + (before - base[0])
            if np.isfinite(after):
                return base + (after - base[-1])
            return base
    return None


class Repairer:
    """``method``: ``"linear"`` (a line between the nearest good readings), ``"equivalent_days"`` (a line
    for intervals up to one hour; longer ones take the mean of the same interval one to four weeks before
    and after, shifted to meet the neighbouring good readings, or a line when no clean equivalent exists) or
    ``"nan"`` (leave a hole). Intervals longer than ``max_gap_hours`` are left open and logged as such."""

    def __init__(self, method: str = "equivalent_days", max_gap_hours: int = 48,
                 offsets_days: tuple[int, ...] = (7, 14, 21, 28), reason: str = "flagged"):
        if method not in ("linear", "equivalent_days", "nan"):
            raise ValueError("method must be 'linear', 'equivalent_days' or 'nan'")
        self.method = method
        self.max_gap_hours = int(max_gap_hours)
        self.offsets_days = tuple(int(d) for d in offsets_days)
        self.reason = reason

    def transform(self, values: Any, timestamps: Any = None, flags: Any = None) -> tuple[np.ndarray, Audit]:
        v, t = as_series(values, timestamps)
        mask = np.asarray(flags, dtype=bool)
        if len(mask) != len(v):
            raise ValueError("flags must have the length of the series")
        out = v.copy()
        out[mask] = np.nan
        reasons = np.full(len(v), self.reason, dtype="<U16")
        methods = np.full(len(v), "", dtype="<U32")
        iv = intervals_from_mask(mask)
        if self.method == "linear":
            out = _linear(v, mask)
            methods[mask] = "linear"
        elif self.method == "equivalent_days":
            lin = _linear(v, mask)
            filled = out.copy()
            for row in iv:
                i0, i1 = int(row["i0"]), int(row["i1"])
                if row["n"] <= 1:
                    filled[i0 : i1 + 1] = lin[i0 : i1 + 1]
                    methods[i0 : i1 + 1] = "linear"
                    continue
                rep = _equivalent_days(v, mask, i0, i1, self.offsets_days, 24)
                if rep is None:
                    filled[i0 : i1 + 1] = lin[i0 : i1 + 1]
                    methods[i0 : i1 + 1] = "linear (no clean equivalent)"
                else:
                    filled[i0 : i1 + 1] = np.maximum(rep, 0.0)
                    methods[i0 : i1 + 1] = "equivalent days"
            out = filled
        if self.method != "nan" and self.max_gap_hours > 0:
            for row in iv[iv["n"] > self.max_gap_hours]:
                i0, i1 = int(row["i0"]), int(row["i1"])
                out[i0 : i1 + 1] = np.nan
                reasons[i0 : i1 + 1] = "gap too long"
                methods[i0 : i1 + 1] = "left open"
        idx = np.flatnonzero(mask)
        audit = Audit(idx, None if t is None else t[idx], v[idx], out[idx], methods[idx], reasons[idx])
        return out, audit
