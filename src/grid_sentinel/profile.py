"""Typical daily profile and residual detector.

The expected value of an hour is built the way Cardinal Grid Note 2 found to work best on U.S.
balancing-authority load: days are grouped by day type (workday, Saturday, Sunday, with special days
mapped to the weekend type they resemble) and, optionally, by a regime label (for instance a
heating/cooling regime from temperature); the reference set for a day is the same-group days in the
previous ``recent_weeks`` weeks plus the same-group days within ``half_days`` of the same calendar date
one year earlier. The expected shape is the median shape of the reference days, and the level is the
median daily mean of the most recent same-group days, so that the expectation is causal and in MW.

The residual detector standardises the residual by the median absolute deviation of recent residuals
and flags readings beyond ``k`` MADs. Used alone it is a calendar-aware baseline; used as a
cross-check it removes the alarms that a level detector raises on genuine steep ramps.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from grid_sentinel.calendar_us import day_types

MAD_TO_SIGMA = 1.4826


def _daily_frame(series: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    """Pivot an hourly series into days x 24 (NaN where missing). Returns (wide frame, ordinal days)."""
    s = series.astype(float)
    idx = pd.DatetimeIndex(s.index)
    df = pd.DataFrame({"v": s.to_numpy(), "day": idx.normalize(), "hour": idx.hour})
    wide = df.pivot_table(index="day", columns="hour", values="v", aggfunc="mean").reindex(columns=range(24))
    ords = np.array([d.toordinal() for d in wide.index.date])
    return wide, ords


def regime_from_peak_hour(series: pd.Series) -> pd.Series:
    """Load-derived regime per day: 1 when the daily peak falls at local hour 12 or earlier (a heating
    morning), else 0. Causal use: shift by one day (``regime.shift(1)``)."""
    wide, _ = _daily_frame(series)
    ok = wide.notna().sum(axis=1) >= 20
    peak = wide.fillna(-np.inf).to_numpy().argmax(axis=1)
    reg = pd.Series(np.where(peak <= 11, 1, 0), index=wide.index)
    reg[~ok] = np.nan
    return reg


def regime_from_temperature(tmean_f: pd.Series, heat_f: float = 59.0, cool_f: float = 72.0) -> pd.Series:
    """Three-class regime from daily mean temperature in Fahrenheit: 0 heating, 1 mild, 2 cooling."""
    t = tmean_f.astype(float)
    return pd.Series(np.where(t < heat_f, 0, np.where(t > cool_f, 2, 1)), index=t.index).where(t.notna())


def expected_profile(
    series: pd.Series,
    regime: pd.Series | None = None,
    recent_weeks: int = 2,
    analog_years: int = 1,
    half_days: int = 21,
    min_ref: int = 2,
    level_days: int = 5,
) -> pd.Series:
    """Causal expected value, in the units of the series, for every timestamp."""
    wide, ords = _daily_frame(series)
    shapes = wide.to_numpy()
    level = np.nanmean(shapes, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        shapes = shapes / level[:, None]
    dtype = day_types(wide.index)
    if regime is not None:
        r = regime.reindex(wide.index).to_numpy()
        group = np.where(np.isnan(r), np.nan, dtype * 10 + r)
    else:
        group = dtype.astype(float)
    valid_day = np.isfinite(level) & (np.isfinite(shapes).sum(axis=1) >= 20)
    n = len(wide)
    exp_shape = np.full((n, 24), np.nan)
    exp_level = np.full(n, np.nan)
    span = 7 * recent_weeks
    for i in range(n):
        if np.isnan(group[i]):
            continue
        m = (ords >= ords[i] - span) & (ords < ords[i])
        for y in range(1, analog_years + 1):
            c = ords[i] - int(round(365.25 * y))
            m |= (ords >= c - half_days) & (ords <= c + half_days)
        m &= (group == group[i]) & valid_day
        if m.sum() >= min_ref:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                exp_shape[i] = np.nanmedian(shapes[m], axis=0)
        recent = (ords >= ords[i] - 7 * 4) & (ords < ords[i]) & (group == group[i]) & valid_day
        if recent.sum() >= 1:
            exp_level[i] = np.median(level[recent][-level_days:])
    expected = exp_shape * exp_level[:, None]
    flat = pd.Series(expected.reshape(-1), index=pd.DatetimeIndex(np.repeat(wide.index.to_numpy(), 24)) + pd.to_timedelta(np.tile(np.arange(24), len(wide)), unit="h"))
    out = flat
    return out.reindex(pd.DatetimeIndex(series.index))


def profile_residual(
    series: pd.Series,
    regime: pd.Series | None = None,
    k: float = 4.0,
    mad_window_days: int = 28,
    **profile_kwargs,
) -> pd.DataFrame:
    """Flag readings whose residual against the expected profile exceeds ``k`` robust standard deviations.

    Returns the usual detector frame plus ``expected`` and ``residual`` (in series units).
    """
    x = series.astype(float)
    expected = expected_profile(x, regime=regime, **profile_kwargs)
    resid = x - expected
    scale = (
        resid.abs().rolling(f"{mad_window_days}D", min_periods=48).median().shift(1) * MAD_TO_SIGMA
        if isinstance(x.index, pd.DatetimeIndex)
        else resid.abs().rolling(24 * mad_window_days, min_periods=48).median().shift(1) * MAD_TO_SIGMA
    )
    z = (resid / scale).to_numpy()
    score = np.abs(z)
    flags = np.nan_to_num(score, nan=0.0) > k
    return pd.DataFrame(
        {"value": x.to_numpy(), "expected": expected.to_numpy(), "residual": resid.to_numpy(), "score": score,
         "threshold": k, "is_anomaly": flags},
        index=x.index,
    )
