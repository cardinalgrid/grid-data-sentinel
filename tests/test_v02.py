import numpy as np
import pandas as pd

from grid_sentinel import (
    day_types,
    expected_profile,
    intervals_from_mask,
    modified_zscore,
    profile_residual,
    regime_from_peak_hour,
    regime_from_temperature,
    relative_deviation,
    repair,
    summarize_intervals,
)


def hourly_series(days: int = 120, seed: int = 2) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days * 24, freq="h")
    t = np.arange(days * 24)
    wd = idx.weekday.to_numpy()
    base = 10_000 + 2_500 * np.sin(2 * np.pi * (t - 6) / 24)
    base = base * np.where(wd >= 5, 0.85, 1.0)
    return pd.Series(base + rng.normal(0, 80, len(t)), index=idx)


def test_intervals_and_classes():
    mask = np.zeros(200, dtype=bool)
    mask[10] = True
    mask[50:60] = True
    mask[100:180] = True
    iv = intervals_from_mask(mask)
    assert list(iv["n"]) == [1, 10, 80]
    assert list(iv["duration_class"]) == ["up to 1 h", "up to 1 day", "up to 1 week"]
    summ = summarize_intervals(iv)
    assert summ.set_index("duration_class").loc["up to 1 day", "intervals"] == 1
    assert intervals_from_mask(np.zeros(5, dtype=bool)).empty


def test_day_types_map_holidays():
    idx = pd.DatetimeIndex(["2024-11-28", "2024-05-27", "2024-01-15", "2024-06-05", "2024-06-08"])
    dt = day_types(idx)
    assert list(dt) == [1, 2, 0, 0, 1]  # Thanksgiving->Sat, Memorial->Sun, MLK->workday, Wed, Sat


def test_expected_profile_tracks_weekend_level():
    s = hourly_series()
    exp = expected_profile(s, recent_weeks=2, analog_years=0)
    tail = exp.iloc[-24 * 14 :]
    assert tail.notna().mean() > 0.95
    err = ((tail - s.iloc[-24 * 14 :]).abs() / s.iloc[-24 * 14 :]).mean()
    assert err < 0.03


def test_profile_residual_flags_dip_not_ramp():
    s = hourly_series()
    s.iloc[-24 * 10 + 12] *= 0.7  # 30% dip at noon on a workday
    res = profile_residual(s, analog_years=0)
    assert res["is_anomaly"].iloc[-24 * 10 + 12]
    assert res["is_anomaly"].iloc[-24 * 30 :].sum() <= 3


def test_regimes():
    s = hourly_series(days=30)
    reg = regime_from_peak_hour(s)
    assert set(reg.dropna().unique()) <= {0, 1}
    t = pd.Series([40.0, 65.0, 80.0, np.nan])
    assert list(regime_from_temperature(t).iloc[:3]) == [0, 1, 2]
    assert np.isnan(regime_from_temperature(t).iloc[3])


def test_repair_equivalent_days_keeps_shape():
    s = hourly_series(days=60)
    flags = np.zeros(len(s), dtype=bool)
    flags[24 * 40 + 6 : 24 * 40 + 18] = True  # a 12-hour gap on day 40 (a Sunday? whatever it is)
    truth = s.copy()
    fixed, log = repair(s, flags, method="equivalent_days")
    err_eq = (fixed[flags] - truth[flags]).abs().mean()
    lin, _ = repair(s, flags, method="linear")
    err_lin = (lin[flags] - truth[flags]).abs().mean()
    assert (log["method"] == "equivalent days").all()
    assert err_eq < err_lin


def test_new_baselines_shape():
    s = hourly_series(days=40)
    s.iloc[500] *= 3
    for fn in (modified_zscore, relative_deviation):
        res = fn(s)
        assert list(res.columns) == ["value", "score", "threshold", "is_anomaly"]
        assert res["is_anomaly"].iloc[500]
