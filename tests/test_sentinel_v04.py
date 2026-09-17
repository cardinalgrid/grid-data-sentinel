from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grid_sentinel.detectors import Sentinel
from grid_sentinel.state import state_from_json, state_to_json
from grid_sentinel.types import Context

TIDY = Path(__file__).resolve().parents[2] / "ba-forecast-scorecard" / "data" / "tidy"
ISD = Path(__file__).resolve().parents[2] / "notes" / "03-daily-load-profiles" / "data" / "isd"
NEIGHBORS = Path(__file__).resolve().parents[1] / "docs" / "neighbors.csv"


def context_from(temp, t, nb):
    return Context(temperature_f=(temp, t), neighbors={k: (v, t) for k, v in nb.items()})


def test_sentinel_keeps_the_cold_morning_drops_the_lone_spike_and_keeps_unit_errors(arctic_fixture):
    load, t, temp, nb, win, spike = arctic_fixture
    ctx = context_from(temp, t, nb)
    res = Sentinel().predict(load, t, ctx)
    assert not res.is_anomaly[win].any()
    assert (res.reason[win] == "extreme_kept").any()
    assert res.is_anomaly[spike] and res.reason[spike] in ("teda_level", "teda_diff", "gross_ratio")
    assert set(res.checks) == {"profile", "neighbors", "weather"}
    off = Sentinel(extremes="off").predict(load, t, ctx)
    assert off.is_anomaly[win].any()
    load2 = load.copy()
    run = (t >= np.datetime64("2024-01-17T12", "h")) & (t <= np.datetime64("2024-01-17T17", "h"))
    load2[run] *= 10.0
    res2 = Sentinel().predict(load2, t, ctx)
    assert res2.is_anomaly[run].all() and (res2.reason[run] == "gross_ratio").all()
    assert not res2.is_anomaly[win].any()


def test_sentinel_without_cross_checks_reproduces_v02_up_to_declared_differences(arctic_fixture):
    from grid_sentinel.rules import sentinel_v2

    load, t, _temp, _nb, _win, _spike = arctic_fixture
    new = Sentinel(cross_checks=(), regime="load", extremes="off").predict(load, t)
    old = sentinel_v2(pd.Series(load, index=pd.DatetimeIndex(t)))
    old_flags = old["is_anomaly"].to_numpy()
    assert not (new.is_anomaly & ~old_flags).any()  # the causal composite never flags what v0.2 did not
    extra = old_flags & ~new.is_anomaly
    assert extra.mean() < 0.005
    assert set(old["confirmed_by"].to_numpy()[extra]) <= {"stuck", "profile"}  # stuck semantics, no look-ahead


def test_sentinel_stream_equals_batch_and_state_roundtrip(arctic_fixture):
    load, t, temp, nb, _win, _spike = arctic_fixture
    ctx = context_from(temp, t, nb)
    d = Sentinel()
    batch = d.predict(load, t, ctx)
    d.reset()
    half = 24 * 400 + 5
    aligned = ctx.aligned(t)
    for i in range(half):
        d.update(t[i], load[i], **aligned.at(i))
    e = Sentinel().set_state(state_from_json(state_to_json(d.get_state())))
    rest = [e.update(t[i], load[i], **aligned.at(i)) for i in range(half, len(t))]
    assert np.array_equal(batch.is_anomaly[half:], np.array([r.is_anomaly for r in rest]))
    assert list(batch.reason[half:]) == [r.reason for r in rest]


@pytest.mark.skipif(not (TIDY.exists() and ISD.exists() and NEIGHBORS.exists()), reason="public data not on this machine")
def test_sentinel_matches_v03_on_pjm_2024_up_to_declared_differences():
    from grid_sentinel.benchmark import make_context
    from grid_sentinel.data import load_series_local
    from grid_sentinel.rules import sentinel_v3

    s = load_series_local(TIDY, "PJM", 2024)
    s = s[s > 0]
    ctx3 = make_context(TIDY, ISD, NEIGHBORS)("PJM", 2024)
    old = sentinel_v3(s, temperature_f=ctx3["temperature_f"], neighbors=ctx3["neighbors"])
    # the public series has gaps (zeros removed, a missing DST hour): the v0.4 contract wants a regular
    # grid, so the series is reindexed with NaN and the comparison is made on the readings that exist
    grid = pd.date_range(s.index.min(), s.index.max(), freq="h")
    t = grid.to_numpy().astype("datetime64[ns]")
    temp = ctx3["temperature_f"].reindex(grid).to_numpy()
    nb = {k: (v.reindex(grid).to_numpy(), t) for k, v in ctx3["neighbors"].items()}
    full = Sentinel().predict(s.reindex(grid).to_numpy(), t, Context(temperature_f=(temp, t), neighbors=nb))
    pos = grid.get_indexer(s.index)
    new_flags, new_reason = full.is_anomaly[pos], full.reason[pos]

    class _New:
        is_anomaly = new_flags
        reason = new_reason

    new = _New()
    old_flags = old["is_anomaly"].to_numpy()
    assert not (new.is_anomaly & ~old_flags).any()
    extra = old_flags & ~new.is_anomaly
    assert extra.mean() < 0.005
    assert set(old["confirmed_by"].to_numpy()[extra]) <= {"stuck", "profile"}
    assert (new.reason == "extreme_kept").sum() >= (old["confirmed_by"].to_numpy() == "extreme kept").sum() * 0.8
