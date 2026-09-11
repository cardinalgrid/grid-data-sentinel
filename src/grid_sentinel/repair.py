"""Repair of flagged readings with an audit trail."""

from __future__ import annotations

import numpy as np
import pandas as pd

from grid_sentinel.intervals import intervals_from_mask


def _linear(out: pd.Series, mask: np.ndarray) -> pd.Series:
    tmp = out.copy()
    tmp[mask] = np.nan
    return tmp.interpolate(method="linear", limit_area="inside")


def _equivalent_days(values: np.ndarray, mask: np.ndarray, i0: int, i1: int, offsets_days: tuple[int, ...],
                     samples_per_day: int) -> np.ndarray | None:
    """Mean of the same interval ``d`` days before and after, over the first offset for which both sides
    (or at least one) exist and contain no flagged reading; the result is shifted linearly so that its
    ends meet the last good reading before and the first good reading after the interval."""
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
                d1 = before - base[0]
                d2 = after - base[-1]
                w = np.linspace(0, 1, n + 2)[1:-1]
                return base + (1 - w) * d1 + w * d2
            if np.isfinite(before):
                return base + (before - base[0])
            if np.isfinite(after):
                return base + (after - base[-1])
            return base
    return None


def repair(
    series: pd.Series,
    flags: pd.Series | np.ndarray,
    method: str = "linear",
    reason: str = "flagged",
    max_gap: int = 48,
    samples_per_hour: float = 1.0,
    equivalent_offsets_days: tuple[int, ...] = (7, 14, 21, 28),
) -> tuple[pd.Series, pd.DataFrame]:
    """Replace flagged readings and return (repaired series, audit log).

    ``method``:
      ``"linear"``          interpolation between the nearest unflagged neighbours;
      ``"equivalent_days"`` intervals up to 1 hour by a straight line; longer intervals by the mean of
                            the same interval one or more weeks before and after, shifted so that the
                            edges meet the neighbouring good readings (falls back to linear when no
                            clean equivalent interval exists);
      ``"nan"``             leave a hole.
    Intervals longer than ``max_gap`` readings are left as NaN and logged with reason ``"gap too long"``.
    The audit log has one row per changed reading: ``original``, ``repaired``, ``reason``, ``method``.
    """
    s = series.astype(float).copy()
    mask = np.asarray(flags, dtype=bool)
    out = s.copy()
    out[mask] = np.nan
    reasons = np.array([reason] * len(s), dtype=object)
    methods = np.array([""] * len(s), dtype=object)
    iv = intervals_from_mask(mask, samples_per_hour=samples_per_hour)

    if method == "linear":
        out = _linear(s, mask)
        methods[mask] = "linear"
    elif method == "equivalent_days":
        values = s.to_numpy()
        filled = out.to_numpy().copy()
        spd = int(round(24 * samples_per_hour))
        lin = _linear(s, mask).to_numpy()
        for _, r in iv.iterrows():
            i0, i1 = int(r["i0"]), int(r["i1"])
            if r["n"] <= samples_per_hour:
                filled[i0 : i1 + 1] = lin[i0 : i1 + 1]
                methods[i0 : i1 + 1] = "linear"
                continue
            rep = _equivalent_days(values, mask, i0, i1, equivalent_offsets_days, spd)
            if rep is None:
                filled[i0 : i1 + 1] = lin[i0 : i1 + 1]
                methods[i0 : i1 + 1] = "linear (no clean equivalent)"
            else:
                filled[i0 : i1 + 1] = np.maximum(rep, 0.0)
                methods[i0 : i1 + 1] = "equivalent days"
        out = pd.Series(filled, index=s.index)
    elif method != "nan":
        raise ValueError("method must be 'linear', 'equivalent_days' or 'nan'")

    if method != "nan" and max_gap > 0 and not iv.empty:
        for _, r in iv[iv["n"] > max_gap].iterrows():
            i0, i1 = int(r["i0"]), int(r["i1"])
            out.iloc[i0 : i1 + 1] = np.nan
            reasons[i0 : i1 + 1] = "gap too long"
            methods[i0 : i1 + 1] = "left open"

    log = pd.DataFrame(
        {"original": s[mask].to_numpy(), "repaired": out[mask].to_numpy(), "reason": reasons[mask], "method": methods[mask]},
        index=s.index[mask],
    )
    return out, log
