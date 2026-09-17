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


# ---- weather ----


def isd_rows(day: str, temp_tenths: int) -> list[str]:
    y, m, d = day.split("-")
    return [f"{y} {m} {d} {h:02d} {temp_tenths:6d}   -80 10132 270  40 -9999 -9999 -9999" for h in range(24)]


def write_gz(path, rows):
    import gzip

    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(rows) + "\n")


def test_read_isd_lite_parses_tenths_and_drops_missing(tmp_path):
    from grid_sentinel.weather import read_isd_lite

    rows = [
        "2024 01 01 00   -55   -80 10132 270  40 -9999 -9999 -9999",
        "2024 01 01 01 -9999   -80 10132 270  40 -9999 -9999 -9999",
        "2024 01 01 02    12   -80 10132 270  40 -9999 -9999 -9999",
    ]
    write_gz(tmp_path / "x.gz", rows)
    df = read_isd_lite(tmp_path / "x.gz")
    assert list(df["temp_c"]) == [-5.5, 1.2]
    assert df["utc"].iloc[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")


def test_daily_mean_requires_18_hours():
    from grid_sentinel.weather import daily_mean_f

    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    t = pd.Series(50.0, index=idx)
    t.iloc[:10] = np.nan  # first day has 14 hours
    d = daily_mean_f(t)
    assert list(d.index) == [pd.Timestamp("2024-01-02")]


def seasonal_temperature(start="2020-01-01", end="2024-12-31 23:00"):
    idx = pd.date_range(start, end, freq="h")
    doy = idx.dayofyear.to_numpy()
    return pd.Series(50 - 30 * np.cos(2 * np.pi * doy / 365.25), index=idx)


def test_temperature_tail_uses_previous_years_same_weeks():
    from grid_sentinel.weather import temperature_tail

    t = seasonal_temperature()
    tail = temperature_tail(t)
    assert tail.loc["2024-01-15", "p05"].iloc[0] < 30 and tail.loc["2024-07-15", "p95"].iloc[0] > 70
    assert tail.loc["2020-03-01", "p05"].notna().all()  # fallback: previous 8 weeks
    assert tail.loc["2020-01-02", "p05"].isna().all()  # nothing behind it yet


def test_hourly_temperature_f_aligns_to_local_index(tmp_path):
    from grid_sentinel.weather import hourly_temperature_f

    make_tidy(tmp_path, ba="PJM", year=2024, hours=48, offset=5)
    isd = tmp_path / "isd"
    isd.mkdir()
    (isd / "isd-history.csv").write_text(HISTORY, encoding="utf-8")
    rows = isd_rows("2024-01-01", 100) + isd_rows("2024-01-02", 200) + isd_rows("2024-01-03", 300)
    write_gz(isd / "724080-13739-2024.gz", rows)
    t = hourly_temperature_f("PJM", 2024, tmp_path, isd)
    s = load_series_local(tmp_path, "PJM", 2024)
    assert t.index.equals(s.index)
    assert abs(t.iloc[0] - 50.0) < 1e-9  # local 00:00 Jan 1 = UTC 05:00 Jan 1 -> 10 C
    assert abs(t.loc["2024-01-01 19:00"] - 68.0) < 1e-9  # local 19:00 Jan 1 = UTC 00:00 Jan 2 -> 20 C
