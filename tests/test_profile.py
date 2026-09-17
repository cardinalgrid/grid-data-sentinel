import numpy as np
import pandas as pd

from grid_sentinel.detectors import ProfileResidual
from grid_sentinel.profile import profile_residual, regime_from_peak_hour  # v0.3 functions
from grid_sentinel.state import state_from_json, state_to_json
from grid_sentinel.types import Context


def with_faults(synthetic_year):
    v, t = synthetic_year
    v = v.copy()
    v[24 * 100 + 12] *= 0.7   # a 30% dip at noon
    v[24 * 200 + 18] *= 1.5   # a 50% spike in the evening
    v[24 * 250 : 24 * 250 + 5] = np.nan  # five missing hours
    return v, t


def legacy_series(v, t):
    return pd.Series(v, index=pd.DatetimeIndex(t))


def test_profile_matches_the_v03_function_without_regime(synthetic_year):
    v, t = with_faults(synthetic_year)
    new = ProfileResidual(k=4.0, regime=None).predict(v, t)
    old = profile_residual(legacy_series(v, t), regime=None, k=4.0)
    assert np.allclose(new.expected, old["expected"].to_numpy(), equal_nan=True)
    assert np.allclose(new.score, old["score"].to_numpy(), equal_nan=True)
    assert np.array_equal(new.is_anomaly, old["is_anomaly"].to_numpy())
    assert new.is_anomaly[24 * 100 + 12] and new.is_anomaly[24 * 200 + 18]


def test_profile_matches_the_v03_function_with_the_load_regime(synthetic_year):
    v, t = with_faults(synthetic_year)
    s = legacy_series(v, t)
    new = ProfileResidual(k=4.0, regime="load").predict(v, t)
    old = profile_residual(s, regime=regime_from_peak_hour(s).shift(1), k=4.0)
    assert np.allclose(new.expected, old["expected"].to_numpy(), equal_nan=True)
    assert np.array_equal(new.is_anomaly, old["is_anomaly"].to_numpy())


def test_profile_temperature_regime_changes_the_group(synthetic_year):
    v, t = with_faults(synthetic_year)
    doy = (t.astype("datetime64[D]") - t[0].astype("datetime64[D]")).astype(int)
    temp = 50 - 30 * np.cos(2 * np.pi * doy / 365.25)  # cold in January, warm in July
    ctx = Context(temperature_f=(temp, t))
    a = ProfileResidual(k=4.0, regime="temperature").predict(v, t, ctx)
    b = ProfileResidual(k=4.0, regime=None).predict(v, t)
    assert np.isfinite(a.expected).sum() > 24 * 300
    assert not np.allclose(a.expected, b.expected, equal_nan=True)  # the regime changes the reference set


def test_profile_stream_equals_batch_and_state_roundtrip(synthetic_year):
    v, t = with_faults(synthetic_year)
    d = ProfileResidual(k=4.0, regime="load")
    batch = d.predict(v, t)
    d.reset()
    half = 24 * 180 + 7  # in the middle of a day
    for i in range(half):
        d.update(t[i], v[i])
    e = ProfileResidual(k=4.0, regime="load").set_state(state_from_json(state_to_json(d.get_state())))
    rest = [e.update(t[i], v[i]) for i in range(half, len(v))]
    assert np.array_equal(batch.is_anomaly[half:], np.array([r.is_anomaly for r in rest]))
    assert np.allclose(batch.expected[half:], np.array([r.expected for r in rest]), equal_nan=True)


def test_profile_exclude_keeps_a_fault_out_of_the_reference(synthetic_year):
    v, t = synthetic_year
    v = v.copy()
    run = slice(24 * 120 + 8, 24 * 120 + 16)
    v[run] *= 10.0  # a unit error for eight hours
    plain = ProfileResidual(k=4.0, regime=None).predict(v, t)
    d = ProfileResidual(k=4.0, regime=None)
    masked = np.zeros(len(v), dtype=bool)
    masked[run] = True
    out = [d.update(ti, x, exclude=bool(m)) for ti, x, m in zip(t, v, masked)]
    exp_masked = np.array([o.expected for o in out])
    week_later = slice(24 * 127 + 8, 24 * 127 + 16)
    assert np.nanmean(plain.expected[week_later]) > np.nanmean(exp_masked[week_later])  # the fault polluted the plain reference
    assert np.nanmax(np.abs(exp_masked[week_later] - v[week_later]) / v[week_later]) < 0.15
