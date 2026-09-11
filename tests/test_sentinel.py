import numpy as np
import pandas as pd
import pytest

from grid_sentinel import (
    RecursiveTEDA,
    SparseAutoencoder,
    hampel,
    inject_anomalies,
    iqr,
    repair,
    rolling_zscore,
    score_labels,
    sentinel,
    stuck_values,
)


def daily_series(days: int = 60, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(days * 24)
    base = 10_000 + 2_500 * np.sin(2 * np.pi * (t - 6) / 24) + 400 * np.sin(2 * np.pi * t / (24 * 7))
    return pd.Series(base + rng.normal(0, 80, len(t)))


def test_teda_recursion_matches_batch_statistics():
    x = np.array([3.0, 5.0, 9.0, 4.0, 7.0])
    d = RecursiveTEDA(m=3.0)
    for v in x:
        r = d.update(v)
    assert r.k == 5
    assert r.mean == pytest.approx(x.mean())
    assert r.variance == pytest.approx(x.var(ddof=0))  # TEDA recursion gives the population variance


def test_teda_flags_injected_spike_in_diff_mode():
    s = daily_series()
    s.iloc[700] *= 1.6
    res = RecursiveTEDA(m=3.0, diff=True).detect(s)
    assert res["is_anomaly"].iloc[700]
    assert not res["is_anomaly"].iloc[701]  # the return to normal is not a second anomaly
    assert res["is_anomaly"].sum() < 20


def test_teda_handles_nan_and_constant_start():
    s = pd.Series([5.0] * 10 + [np.nan] + [5.0, 50.0])
    res = RecursiveTEDA(m=2.0).detect(s)
    assert res["is_anomaly"].iloc[-1]
    assert not res["is_anomaly"].iloc[10]


def test_autoencoder_trains_and_flags_spike():
    s = daily_series(days=90)
    s.iloc[1000] *= 1.8
    ae = SparseAutoencoder(window=4, encoding_dim=2, epochs=15, seed=0)
    res = ae.detect(s)
    assert ae.history[-1] < ae.history[0]
    assert res["is_anomaly"].iloc[1000]
    assert res["is_anomaly"].mean() < 0.05


def test_autoencoder_refuses_constant_series():
    with pytest.raises(ValueError):
        SparseAutoencoder().fit(pd.Series(np.ones(100)))


def test_baselines_return_expected_frame():
    s = daily_series(days=20)
    s.iloc[100] = 0.0
    for fn in (rolling_zscore, hampel, iqr):
        res = fn(s)
        assert list(res.columns) == ["value", "score", "threshold", "is_anomaly"]
        assert len(res) == len(s)
    assert hampel(s)["is_anomaly"].iloc[100]
    assert iqr(s)["is_anomaly"].iloc[100]


def test_inject_labels_match_changes():
    s = daily_series(days=120)
    inj = inject_anomalies(s, rate=0.01, seed=3)
    changed = ~np.isclose(inj["clean"], inj["value"])
    assert set(inj.loc[changed, "kind"].unique()) <= {"spike", "dip", "zero", "stuck", "scale"}
    assert inj["label"].sum() >= 0.009 * len(s)
    # stuck readings can coincide with the clean value; every other kind must differ
    other = inj["kind"].isin(["spike", "dip", "zero", "scale"]).to_numpy()
    assert changed[other].all()
    assert not changed[inj["label"].to_numpy() == 0].any()


def test_score_labels_with_tolerance():
    y = np.array([0, 0, 1, 0, 0, 0, 1, 0])
    p = np.array([0, 1, 0, 0, 0, 0, 1, 0])
    exact = score_labels(y, p)
    loose = score_labels(y, p, tolerance=1)
    assert exact["tp"] == 1 and exact["fp"] == 1
    assert loose["tp"] == 2 and loose["fp"] == 0 and loose["recall"] == 1.0


def test_repair_interpolates_and_logs():
    s = pd.Series([1.0, 2.0, 30.0, 4.0, 5.0])
    flags = np.array([False, False, True, False, False])
    fixed, log = repair(s, flags, reason="test")
    assert fixed.iloc[2] == pytest.approx(3.0)
    assert len(log) == 1 and log["reason"].iloc[0] == "test"


def test_repair_leaves_long_gaps_open():
    s = pd.Series(np.arange(100, dtype=float))
    flags = np.zeros(100, dtype=bool)
    flags[10:70] = True
    fixed, log = repair(s, flags, max_gap=48)
    assert fixed.iloc[10:70].isna().all()
    assert (log["reason"] == "gap too long").all()


def test_stuck_rule_flags_frozen_run_but_keeps_first_value():
    s = daily_series(days=10)
    s.iloc[50:56] = s.iloc[49]
    res = stuck_values(s, min_run=3)
    assert not res["is_anomaly"].iloc[49]
    assert res["is_anomaly"].iloc[50:56].all()
    assert res["is_anomaly"].sum() == 6


def test_sentinel_composite_catches_spike_scale_and_stuck():
    s = daily_series(days=90)
    s.iloc[300] *= 1.7
    s.iloc[900:930] *= 10
    s.iloc[1500:1508] = s.iloc[1499]
    res = sentinel(s)
    assert res["is_anomaly"].iloc[300]
    assert res["is_anomaly"].iloc[900:930].mean() > 0.9
    assert res["is_anomaly"].iloc[1500:1508].all()
    assert res["is_anomaly"].mean() < 0.05
