"""Cross-checks that decide whether a high reading is a genuine extreme or a fault.

Each check returns, per timestamp, 1.0 (the reading is confirmed as genuine), 0.0 (not confirmed) or
NaN (the check is not available). ``preserve_extremes`` applies the rule: a flagged reading above the
expected profile is kept as a fault only if at least two of the available checks fail to confirm it;
with exactly two available it must fail both; with fewer than two the original flag stands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from grid_sentinel.profile import regime_from_temperature
from grid_sentinel.weather import daily_mean_f, temperature_tail


def _normalised(s: pd.Series, index: pd.DatetimeIndex, window_days: int) -> pd.Series:
    """Load divided by the causal mean of the same hour over the previous ``window_days`` days."""
    x = s.astype(float).reindex(index)
    hour = pd.DatetimeIndex(index).hour
    ref = x.groupby(hour).transform(lambda v: v.rolling(window_days, min_periods=7).mean().shift(1))
    return x / ref


def confirm_neighbors(
    neighbor_series: dict[str, pd.Series], index: pd.DatetimeIndex, window_days: int = 14, pct: float = 95.0,
    recent_days: int = 28, min_neighbors: int = 2,
) -> pd.Series:
    """1.0 when the median normalised load of the neighbours at that hour is above its own recent
    ``pct`` percentile; 0.0 otherwise; NaN when fewer than ``min_neighbors`` neighbours have data."""
    if len(neighbor_series) < min_neighbors:
        return pd.Series(np.nan, index=index)
    norms = pd.concat([_normalised(s, index, window_days) for s in neighbor_series.values()], axis=1)
    med = norms.median(axis=1, skipna=True)
    med[norms.notna().sum(axis=1) < min_neighbors] = np.nan
    thr = med.rolling(24 * recent_days, min_periods=24 * 7).quantile(pct / 100.0).shift(1)
    out = pd.Series(np.where(med > thr, 1.0, 0.0), index=index)
    out[med.isna() | thr.isna()] = np.nan
    return out


def confirm_weather(temp_f: pd.Series, half_weeks: int = 3, heat_f: float = 59.0, cool_f: float = 72.0) -> pd.Series:
    """1.0 when the hour's temperature is in the BA's own seasonal tail for the day's regime (below the
    5th percentile on a heating day, above the 95th on a cooling day); 0.0 otherwise; NaN without data."""
    t = temp_f.astype(float)
    idx = pd.DatetimeIndex(t.index)
    tail = temperature_tail(t, half_weeks=half_weeks)
    regime = regime_from_temperature(daily_mean_f(t), heat_f=heat_f, cool_f=cool_f).reindex(idx.normalize())
    regime.index = idx
    cold = (regime == 0) & (t < tail["p05"])
    hot = (regime == 2) & (t > tail["p95"])
    out = pd.Series(np.where(cold | hot, 1.0, 0.0), index=idx)
    out[t.isna() | tail["p05"].isna() | regime.isna()] = np.nan
    return out


def preserve_extremes(flagged: np.ndarray, above: np.ndarray, checks: list) -> np.ndarray:
    """Flags after the two-of-three rule for readings that are ``flagged`` and ``above`` the expectation."""
    flagged = np.asarray(flagged, dtype=bool)
    above = np.asarray(above, dtype=bool)
    if not checks:
        return flagged.copy()
    c = np.vstack([np.asarray(pd.Series(x).to_numpy(), dtype=float) for x in checks])
    available = np.isfinite(c).sum(axis=0)
    failed = (c == 0.0).sum(axis=0)
    # fewer than two checks is not independent evidence: the original decision stands
    keep_fault = np.where(available < 2, True, failed >= 2)
    out = flagged.copy()
    sel = flagged & above
    out[sel] = keep_fault[sel]
    return out
