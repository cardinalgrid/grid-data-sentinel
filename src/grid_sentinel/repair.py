"""Repair of flagged readings with an audit trail."""

from __future__ import annotations

import numpy as np
import pandas as pd


def repair(
    series: pd.Series,
    flags: pd.Series | np.ndarray,
    method: str = "linear",
    reason: str = "flagged",
    max_gap: int = 48,
) -> tuple[pd.Series, pd.DataFrame]:
    """Replace flagged readings and return (repaired series, audit log).

    ``method`` is ``"linear"`` (interpolation between the nearest unflagged neighbours) or
    ``"nan"`` (leave a hole). Gaps longer than ``max_gap`` readings are left as NaN and logged with
    reason ``"gap too long"`` so that a long outage is never silently invented.
    The audit log has one row per changed reading: ``original``, ``repaired``, ``reason``.
    """
    s = series.astype(float).copy()
    mask = np.asarray(flags, dtype=bool)
    out = s.copy()
    out[mask] = np.nan
    if method == "linear":
        out = out.interpolate(method="linear", limit_area="inside")
    elif method != "nan":
        raise ValueError("method must be 'linear' or 'nan'")

    # long gaps back to NaN
    reasons = np.array([reason] * len(s), dtype=object)
    if method == "linear" and max_gap > 0:
        run_id = (~mask).cumsum()
        run_len = pd.Series(mask).groupby(run_id).transform("sum").to_numpy()
        too_long = mask & (run_len > max_gap)
        out[too_long] = np.nan
        reasons[too_long] = "gap too long"

    log = pd.DataFrame(
        {"original": s[mask].to_numpy(), "repaired": out[mask].to_numpy(), "reason": reasons[mask]},
        index=s.index[mask],
    )
    return out, log
