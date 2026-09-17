import numpy as np
import pandas as pd

from grid_sentinel.calendar_us import day_types, special_days


def test_day_types_map_holidays_without_pandas():
    days = np.array(["2024-11-28", "2024-05-27", "2024-01-15", "2024-06-05", "2024-06-08", "2024-02-11"], dtype="datetime64[D]")
    assert list(day_types(days)) == [1, 2, 0, 0, 1, 1]  # Thanksgiving->Sat, Memorial->Sun, MLK->workday, Wed, Sat, Super Bowl->Sat
    assert day_types(days).dtype == np.int8


def test_day_types_accepts_hourly_timestamps_and_matches_the_v03_function():
    t = (np.datetime64("2023-01-01T00", "h") + np.arange(24 * 730).astype("timedelta64[h]")).astype("datetime64[ns]")
    ours = day_types(t)
    from grid_sentinel.calendar_us import (
        day_types_index,  # the pandas-based v0.3 function, kept for the comparison
    )

    theirs = day_types_index(pd.DatetimeIndex(t))
    assert np.array_equal(ours, theirs)
    assert len(special_days(2024)) >= 10
