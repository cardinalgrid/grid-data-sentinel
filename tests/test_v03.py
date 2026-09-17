import numpy as np
import pandas as pd

from grid_sentinel.data import load_series_local, utc_offset


def make_tidy(tmp_path, ba="PJM", year=2024, hours=48, offset=5):
    local_end = pd.date_range(f"{year}-01-01 01:00", periods=hours, freq="h")
    df = pd.DataFrame(
        {
            "ba": ba,
            "region": "MIDA",
            "date": local_end.normalize(),
            "hour": local_end.hour,
            "local_end": local_end,
            "utc_end": (local_end + pd.Timedelta(hours=offset)).tz_localize("UTC"),
            "forecast": 100.0,
            "demand": np.arange(hours, dtype=float) + 1000.0,
            "demand_adjusted": 100.0,
            "demand_imputed": False,
        }
    )
    df.to_parquet(tmp_path / f"{year}H1.parquet", index=False)
    return df


def test_load_series_local_hour_beginning(tmp_path):
    make_tidy(tmp_path)
    s = load_series_local(tmp_path, "PJM", 2024)
    assert s.index[0] == pd.Timestamp("2024-01-01 00:00")
    assert s.iloc[0] == 1000.0 and len(s) == 48


def test_utc_offset_matches_index(tmp_path):
    make_tidy(tmp_path, offset=5)
    s = load_series_local(tmp_path, "PJM", 2024)
    off = utc_offset(tmp_path, "PJM", 2024)
    assert off.index.equals(s.index)
    assert (off == 5).all()
