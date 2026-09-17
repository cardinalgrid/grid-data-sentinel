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


# ---- stations ----

HISTORY = (
    "USAF,WBAN,STATION NAME,CTRY,STATE,ICAO,LAT,LON,ELEV(M),BEGIN,END\n"
    "724080,13739,PHILADELPHIA INTL,US,PA,KPHL,39.873,-75.227,2.0,19410101,20261231\n"
    "722590,03927,DALLAS FT WORTH INTL,US,TX,KDFW,32.898,-97.019,171.0,19730101,20261231\n"
)


def test_ba_station_map_is_the_note_2_map():
    from grid_sentinel.stations import BA_STATION

    assert len(BA_STATION) == 44
    assert BA_STATION["PJM"] == "KPHL" and BA_STATION["ERCO"] == "KDFW"


def test_distance_km_known_pair():
    from grid_sentinel.stations import distance_km

    # Philadelphia (KPHL) to Dallas (KDFW) is about 2,090 km
    assert abs(distance_km(39.87, -75.24, 32.90, -97.04) - 2090) < 40


def test_station_table_from_local_history(tmp_path):
    from grid_sentinel.stations import ba_distances, station_table

    (tmp_path / "isd-history.csv").write_text(HISTORY, encoding="utf-8")
    t = station_table(tmp_path)
    assert set(t.columns) == {"icao", "usaf", "wban", "name", "state", "lat", "lon"}
    assert set(t["icao"]) == {"KPHL", "KDFW"}
    d = ba_distances(tmp_path)
    assert abs(d.loc["PJM", "ERCO"] - 2090) < 40 and d.loc["PJM", "PJM"] == 0
