import numpy as np
import pandas as pd

from grid_sentinel.crosscheck import confirm_neighbors, confirm_weather, preserve_extremes  # v0.3 functions
from grid_sentinel.detectors.crosscheck import NeighborCheck, WeatherCheck, preserve_one
from grid_sentinel.state import state_from_json, state_to_json


def test_preserve_one_matches_the_truth_table():
    profile = [1, 1, 0, 0, 1, np.nan, np.nan, 0]
    neighbours = [1, 0, 0, 1, np.nan, 1, np.nan, 0]
    weather = [1, 1, 0, 0, np.nan, 0, np.nan, 0]
    flagged = np.array([True] * 8)
    above = np.array([True] * 7 + [False])
    batch = preserve_extremes(flagged, above, [np.array(profile), np.array(neighbours), np.array(weather)])
    one = [preserve_one(True, bool(above[i]), [profile[i], neighbours[i], weather[i]]) for i in range(8)]
    assert list(batch) == one == [False, False, True, True, True, False, True, True]


def neighbours_fixture():
    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="h")
    rng = np.random.default_rng(3)
    base = 100 + 20 * np.sin(2 * np.pi * (np.arange(len(idx)) - 6) / 24)
    n1 = pd.Series(base + rng.normal(0, 0.5, len(idx)), index=idx)  # noise avoids exact ties at the percentile
    n2 = pd.Series(2 * base + rng.normal(0, 1.0, len(idx)), index=idx)
    n3 = pd.Series(0.5 * base + rng.normal(0, 0.25, len(idx)), index=idx)
    for s in (n1, n2):
        s.iloc[-30] *= 1.5
    n3.iloc[-10] *= 1.5
    return idx, {"a": n1, "b": n2, "c": n3}


def test_neighbor_check_matches_the_v03_function_and_streams():
    idx, nb = neighbours_fixture()
    old = confirm_neighbors(nb, idx).to_numpy()
    t = idx.to_numpy().astype("datetime64[ns]")
    chk = NeighborCheck()
    new = np.array([chk.update(ti, {k: float(s.iloc[i]) for k, s in nb.items()}) for i, ti in enumerate(t)])
    assert np.allclose(new, old, equal_nan=True)
    assert new[-30] == 1.0 and new[-10] == 0.0
    chk2 = NeighborCheck()
    for i in range(len(t) // 2):
        chk2.update(t[i], {k: float(s.iloc[i]) for k, s in nb.items()})
    chk3 = NeighborCheck().set_state(state_from_json(state_to_json(chk2.get_state())))
    rest = np.array([chk3.update(t[i], {k: float(s.iloc[i]) for k, s in nb.items()}) for i in range(len(t) // 2, len(t))])
    assert np.allclose(rest, old[len(t) // 2 :], equal_nan=True)


def test_weather_check_matches_the_v03_function_and_streams():
    idx = pd.date_range("2020-01-01", "2024-12-31 23:00", freq="h")
    doy = idx.dayofyear.to_numpy()
    temp = pd.Series(50 - 30 * np.cos(2 * np.pi * doy / 365.25), index=idx)
    temp.loc["2024-01-20"] = -5.0
    # v0.3 used the same day's mean temperature for the regime; the streaming check can only know
    # yesterday's, so the reference is the v0.3 computation with the regime shifted by one day
    from grid_sentinel.profile import regime_from_temperature
    from grid_sentinel.weather import daily_mean_f, temperature_tail

    tail = temperature_tail(temp)
    regime = regime_from_temperature(daily_mean_f(temp)).shift(1).reindex(idx.normalize())
    regime.index = idx
    cold = (regime == 0) & (temp < tail["p05"])
    hot = (regime == 2) & (temp > tail["p95"])
    old = np.where(cold | hot, 1.0, 0.0)
    old[(temp.isna() | tail["p05"].isna() | regime.isna()).to_numpy()] = np.nan
    v03 = confirm_weather(temp).to_numpy()
    differs = ~((v03 == old) | (np.isnan(v03) & np.isnan(old)))
    assert differs.sum() < 24 * 10  # only regime-transition days differ from the v0.3 function
    t = idx.to_numpy().astype("datetime64[ns]")
    chk = WeatherCheck()
    new = np.array([chk.update(ti, float(x)) for ti, x in zip(t, temp.to_numpy())])
    assert np.allclose(new, old, equal_nan=True)
    assert (new[idx.normalize() == pd.Timestamp("2024-01-20")] == 1.0).all()
