"""Shared fixtures: synthetic hourly series as (values, timestamps) numpy arrays."""

import numpy as np
import pytest


def _hourly(start: str, hours: int) -> np.ndarray:
    return (np.datetime64(start, "h") + np.arange(hours).astype("timedelta64[h]")).astype("datetime64[ns]")


@pytest.fixture
def synthetic_year():
    """One year of hourly load with a daily cycle, lighter weekends, a seasonal level and noise."""
    rng = np.random.default_rng(1)
    hours = 24 * 365
    t = _hourly("2024-01-01T00", hours)
    h = np.arange(hours)
    weekday = ((t.astype("datetime64[D]").astype("int64") + 3) % 7)  # 1970-01-01 was a Thursday
    doy = (t.astype("datetime64[D]") - np.datetime64("2024-01-01", "D")).astype(int)
    daily = 10_000 + 2_500 * np.sin(2 * np.pi * (h - 6) / 24)
    season = 1 + 0.15 * np.cos(2 * np.pi * doy / 365.25)
    weekend = np.where(weekday >= 5, 0.85, 1.0)
    v = daily * season * weekend + rng.normal(0, 80, hours)
    return v, t


@pytest.fixture
def arctic_fixture():
    """Two years and two months of hourly load, its temperature (F) and two neighbours, with one genuine
    arctic morning (load up everywhere, temperature in the cold tail) and one lone spike on a mild afternoon.
    Returns (values, timestamps, temperature, neighbours, window_slice, spike_index)."""
    rng = np.random.default_rng(0)
    t = _hourly("2022-01-01T00", 24 * (365 + 365 + 60))
    n = len(t)
    h = np.arange(n)
    doy = (t.astype("datetime64[D]") - t[0].astype("datetime64[D]")).astype(int) % 365
    daily = 10_000 + 2_500 * np.sin(2 * np.pi * (h - 6) / 24)
    season = 1 + 0.15 * np.cos(2 * np.pi * doy / 365.25)
    load = daily * season + rng.normal(0, 60, n)
    temp = 50 - 30 * np.cos(2 * np.pi * doy / 365.25) + rng.normal(0, 3, n)
    nb = {"n1": load * 0.5 + rng.normal(0, 30, n), "n2": load * 2.0 + rng.normal(0, 100, n)}
    day = np.datetime64("2024-01-17", "D")
    win = (t >= np.datetime64("2024-01-17T06", "h")) & (t <= np.datetime64("2024-01-17T09", "h"))
    for s in (load, nb["n1"], nb["n2"]):
        s[win] *= 1.6
    temp[t.astype("datetime64[D]") == day] = -10.0
    spike = int(np.flatnonzero(t == np.datetime64("2024-02-10T15", "h"))[0])
    load[spike] *= 1.6
    return load, t, temp, nb, win, spike
