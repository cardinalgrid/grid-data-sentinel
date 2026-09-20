import numpy as np
import pandas as pd

from grid_sentinel.alignment import alignment_check, alignment_table, change_point


def series(days: int = 800, switch: str | None = "2021-02-01"):
    """A daily load cycle with noise and a forecast of it that, from ``switch`` on, is stamped one hour early."""
    idx = pd.date_range("2020-01-01", periods=24 * days, freq="h")
    rng = np.random.default_rng(0)
    load = pd.Series(100 + 25 * np.sin(2 * np.pi * (idx.hour - 6) / 24) + rng.normal(0, 1.0, len(idx)), index=idx)
    forecast = load + rng.normal(0, 1.0, len(idx))
    if switch is not None:
        late = forecast.index >= pd.Timestamp(switch)
        moved = forecast[late].copy()
        moved.index = moved.index - pd.Timedelta(hours=1)  # the value for hour t is now labelled t - 1
        forecast = pd.concat([forecast[~late], moved])
        forecast = forecast[~forecast.index.duplicated()]
    return load, forecast


def test_aligned_series_raise_no_flag():
    load, forecast = series(days=400, switch=None)
    t = alignment_table(load, forecast)
    assert (t["best_shift_h"] == 0).all()
    assert alignment_check(load, forecast).empty


def test_one_hour_label_change_is_found_with_its_direction_and_year():
    load, forecast = series()
    t = alignment_table(load, forecast)
    assert t.loc["2020-01-01", "best_shift_h"] == 0
    assert t.loc["2021-01-01", "best_shift_h"] == -1          # the load at t matches the forecast stamped t - 1 h
    assert t.loc["2022-01-01", "best_shift_h"] == -1
    flagged = alignment_check(load, forecast)
    assert list(flagged.index.year) == [2021, 2022]
    assert (flagged["gain"] > 2).all()


def test_change_point_lands_within_a_week_of_the_switch():
    load, forecast = series()
    cp = change_point(load, forecast, shift=-1)
    assert cp is not None
    assert abs((cp - pd.Timestamp("2021-02-01")).days) <= 7


def test_reporting_faults_and_short_periods_are_left_out():
    load, forecast = series(days=400, switch=None)
    forecast.iloc[1000:1010] = 10 * forecast.iloc[1000:1010]   # ten hours off by a factor of ten
    t = alignment_table(load, forecast)
    assert t["mape_shift_+0h"].max() < 2.0
    load2, forecast2 = series(days=40, switch=None)
    assert alignment_table(load2, forecast2, min_hours=24 * 60).empty
