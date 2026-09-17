"""Deterministic rules for failure modes that statistical detectors miss, and a composite detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from grid_sentinel.profile import profile_residual, regime_from_peak_hour, regime_from_temperature
from grid_sentinel.teda import RecursiveTEDA


def _frame(series: pd.Series | np.ndarray, score: np.ndarray, thr: float, flags: np.ndarray) -> pd.DataFrame:
    index = series.index if isinstance(series, pd.Series) else pd.RangeIndex(len(score))
    return pd.DataFrame(
        {"value": np.asarray(series, dtype=float), "score": score, "threshold": thr, "is_anomaly": flags},
        index=index,
    )


def stuck_values(series: pd.Series | np.ndarray, min_run: int = 3) -> pd.DataFrame:
    """Flag runs of ``min_run`` or more identical consecutive readings (a frozen telemetry value).

    Load never repeats to the last megawatt for hours; a run of identical values is a stale point,
    not a measurement. The score is the run length; the first reading of the run is kept, since it
    is usually the last good value.
    """
    v = np.asarray(series, dtype=float)
    same = np.zeros(len(v), dtype=bool)
    same[1:] = (v[1:] == v[:-1]) & ~np.isnan(v[1:])
    run = np.zeros(len(v), dtype=int)
    for i in range(1, len(v)):
        run[i] = run[i - 1] + 1 if same[i] else 0
    # propagate the final run length back over the whole run
    length = run.copy()
    for i in range(len(v) - 2, -1, -1):
        if same[i + 1]:
            length[i] = length[i + 1]
    flags = same & (length >= min_run - 1)
    return _frame(series, length.astype(float), float(min_run), flags)


def sentinel(
    series: pd.Series | np.ndarray,
    m_level: float = 4.0,
    m_diff: float = 3.0,
    min_run: int = 3,
    robust: bool = False,
) -> pd.DataFrame:
    """Composite v0.1 detector: recursive TEDA on levels, recursive TEDA on differences, stuck-value rule.

    A reading is flagged if any component flags it. The score is the largest of the three
    component scores, each expressed as a ratio to its own threshold.
    """
    a = RecursiveTEDA(m=m_level, diff=False, robust=robust).detect(series)
    b = RecursiveTEDA(m=m_diff, diff=True, robust=robust).detect(series)
    c = stuck_values(series, min_run=min_run)
    score = np.nanmax(
        np.vstack([a["score"].to_numpy(), b["score"].to_numpy(), c["score"].to_numpy() / c["threshold"].to_numpy()]),
        axis=0,
    )
    flags = a["is_anomaly"].to_numpy() | b["is_anomaly"].to_numpy() | c["is_anomaly"].to_numpy()
    return _frame(series, score, 1.0, flags)


def sentinel_v2(
    series: pd.Series,
    regime: pd.Series | None = None,
    m_level: float = 4.0,
    m_diff: float = 3.0,
    min_run: int = 3,
    k_profile: float = 5.0,
    band: float = 0.35,
    scale_ratio: float = 3.0,
    use_load_regime: bool = True,
    robust: bool = False,
) -> pd.DataFrame:
    """Composite v0.2: the v0.1 detectors confirmed by a calendar-aware profile.

    A reading flagged by TEDA (levels or differences) is kept only if it is also implausible against the
    expected profile: residual beyond ``k_profile`` robust standard deviations, or more than ``band`` (35%)
    away from the expected value, or a ratio to it outside [1/scale_ratio, scale_ratio]. Frozen runs (stuck rule), non-positive readings and readings
    with no available expectation are kept without the check. The profile uses day types with special
    days, the last two weeks plus last year's analogs, and a regime label: the one passed in ``regime``
    (for example from temperature), else yesterday's load-derived regime when ``use_load_regime``.
    ``robust=True`` uses the winsorised TEDA update in the base detectors (more recall on long faults, less
    precision on injected ones; see the benchmark). Requires a DatetimeIndex in local time.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("sentinel_v2 needs a DatetimeIndex in local time")
    base = sentinel(series, m_level=m_level, m_diff=m_diff, min_run=min_run, robust=robust)
    if regime is None and use_load_regime:
        regime = regime_from_peak_hour(series).shift(1)
    # the expectation is built from the series with the readings the base detectors flagged set to missing,
    # so that a fault does not enter the reference set of the days that follow it; residuals are then
    # computed on the original series
    masked = series.astype(float).mask(base["is_anomaly"].to_numpy())
    prof = profile_residual(masked, regime=regime, k=k_profile)
    prof["residual"] = series.astype(float).to_numpy() - prof["expected"].to_numpy()
    prof["score"] = np.abs(prof["residual"].to_numpy() / (prof["residual"].abs().rolling(24 * 28, min_periods=48).median().shift(1) * 1.4826).to_numpy())
    stuck = stuck_values(series, min_run=min_run)["is_anomaly"].to_numpy()
    x = series.astype(float).to_numpy()
    expected = prof["expected"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = x / expected
    implausible = ((np.nan_to_num(prof["score"].to_numpy(), nan=0.0) > k_profile) | (np.abs(np.nan_to_num(ratio, nan=1.0) - 1.0) > band)
                   | (ratio > scale_ratio) | (ratio < 1.0 / scale_ratio))
    # a level or difference detector may fire on the reading next to the fault (the return to normal):
    # accept the flag if the reading itself or a neighbour is implausible against the profile
    wide = implausible.copy()
    wide[1:] |= implausible[:-1]
    wide[:-1] |= implausible[1:]
    implausible = wide
    no_expectation = ~np.isfinite(expected)
    nonsense = ~(x > 0)
    keep = stuck | nonsense | no_expectation | implausible
    grossly = (ratio > scale_ratio) | (ratio < 1.0 / scale_ratio)  # far outside any plausible band: a fault whatever TEDA says
    flags = (base["is_anomaly"].to_numpy() & keep) | grossly | nonsense
    out = base.copy()
    out["is_anomaly"] = flags
    out["expected"] = expected
    out["profile_score"] = prof["score"].to_numpy()
    out["confirmed_by"] = np.where(~flags, np.where(base["is_anomaly"].to_numpy(), "dropped", ""),
                                   np.where(nonsense, "non-positive", np.where(stuck, "stuck", np.where(grossly, "gross ratio",
                                   np.where(no_expectation, "no expectation", "profile")))))
    return out


def sentinel_v3(
    series: pd.Series,
    temperature_f: pd.Series | None = None,
    neighbors: dict[str, pd.Series] | None = None,
    checks: tuple[str, ...] = ("profile", "neighbors", "weather"),
    extremes: str = "preserve",
    profile_band: float = 0.35,
    **v2_kwargs,
) -> pd.DataFrame:
    """Composite v0.3: v0.2 with a temperature-derived regime and preservation of genuine extremes.

    A reading flagged by v0.2 *above* its expected value is kept as a fault only if at least two of the
    available cross-checks fail to confirm it as genuine: the profile (the reading is within
    ``profile_band`` of the expected value, the same band v0.2 uses, so it confirms only readings that the
    robust-residual rule flagged while the ratio to the expectation stayed plausible), the
    neighbours (the BA's interchange partners rose at the same hour, each against its own recent normal)
    and the weather (the hour's temperature is in the BA's own seasonal tail for the heating or cooling
    regime). With two checks available the reading must fail both; with one, or none, the v0.2 decision
    stands. Readings below the expected value, frozen runs and non-positive readings follow the v0.2
    rules unchanged. ``extremes="off"`` returns the v0.2 result with the check columns attached.
    The profile regime comes from the daily mean temperature when ``temperature_f`` is given.
    """
    from grid_sentinel.crosscheck import confirm_neighbors, confirm_weather, preserve_extremes
    from grid_sentinel.weather import daily_mean_f

    regime = v2_kwargs.pop("regime", None)
    if regime is None and temperature_f is not None:
        regime = regime_from_temperature(daily_mean_f(temperature_f.reindex(series.index))).shift(1)
    out = sentinel_v2(series, regime=regime, **v2_kwargs)
    x = series.astype(float).to_numpy()
    expected = out["expected"].to_numpy()
    above = np.isfinite(expected) & (x > expected)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = x / expected
    check_profile = pd.Series(
        np.where(np.isfinite(ratio), (np.abs(np.nan_to_num(ratio, nan=np.inf) - 1.0) <= profile_band).astype(float), np.nan),
        index=series.index,
    )
    idx = pd.DatetimeIndex(series.index)
    check_neighbors = confirm_neighbors(neighbors, idx) if neighbors else pd.Series(np.nan, index=series.index)
    check_weather = (confirm_weather(temperature_f.reindex(series.index)) if temperature_f is not None
                     else pd.Series(np.nan, index=series.index))
    out["above_expected"] = above
    out["check_profile"] = check_profile.to_numpy()
    out["check_neighbors"] = check_neighbors.to_numpy()
    out["check_weather"] = check_weather.to_numpy()
    if extremes == "off":
        return out
    named = (("profile", check_profile), ("neighbors", check_neighbors), ("weather", check_weather))
    used = [c for name, c in named if name in checks]
    before = out["is_anomaly"].to_numpy().copy()
    hard = out["confirmed_by"].isin(["stuck", "non-positive"]).to_numpy()
    flags = preserve_extremes(before, above & ~hard, used)
    out["is_anomaly"] = flags
    out.loc[before & ~flags, "confirmed_by"] = "extreme kept"
    return out
