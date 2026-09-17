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


# ---- neighbours ----

INTERCHANGE_CSV = (
    '"Balancing Authority","Data Date","Hour Number","Directly Interconnected Balancing Authority",'
    '"Interchange (MW)","Local Time at End of Hour","UTC Time at End of Hour","Region","DIBA_Region"\n'
    "AECI,01/01/2024,1,MISO,74,01/01/2024 1:00:00 AM,01/01/2024 7:00:00 AM,MIDW,MIDW\n"
    "AECI,01/01/2024,2,MISO,-100,01/01/2024 2:00:00 AM,01/01/2024 8:00:00 AM,MIDW,MIDW\n"
    'AECI,01/01/2024,3,MISO,"1,200",01/01/2024 3:00:00 AM,01/01/2024 9:00:00 AM,MIDW,MIDW\n'
    "AECI,01/01/2024,1,SPA,-233,01/01/2024 1:00:00 AM,01/01/2024 7:00:00 AM,MIDW,CENT\n"
)


def test_interchange_pairs_median_abs(tmp_path):
    from grid_sentinel.neighbors import interchange_pairs

    p = tmp_path / "EIA930_INTERCHANGE_2024_Jan_Jun.csv"
    p.write_text(INTERCHANGE_CSV, encoding="utf-8")
    pairs = interchange_pairs([p]).set_index(["ba", "neighbor"])
    assert pairs.loc[("AECI", "MISO"), "weight_mw"] == 100.0 and pairs.loc[("AECI", "MISO"), "hours"] == 3
    assert pairs.loc[("AECI", "SPA"), "weight_mw"] == 233.0


def test_neighbor_table_uses_interchange_then_distance():
    from grid_sentinel.neighbors import neighbor_table, neighbors_of

    pairs = pd.DataFrame(
        {"ba": ["A", "A", "B", "E", "E", "E"], "neighbor": ["B", "X", "A", "A", "B", "C"],
         "weight_mw": [500.0, 50.0, 500.0, 300.0, 700.0, 0.0], "hours": [100] * 6}
    )
    dist = pd.DataFrame(
        [[0, 100, 300, 900, 50], [100, 0, 250, 900, 60], [300, 250, 0, 900, 350], [900, 900, 900, 0, 900],
         [50, 60, 350, 900, 0]],
        index=list("ABCDE"), columns=list("ABCDE"), dtype=float,
    )
    t = neighbor_table(pairs, dist, has_load={"A", "B", "C", "D", "E"})
    # E has two interchange partners with load data -> interchange, heaviest first
    assert neighbors_of(t, "E") == ["B", "A"] and set(t[t["ba"] == "E"]["source"]) == {"interchange"}
    # A has one partner with load data (X has none) -> distance fallback within 400 km, nearest first
    assert neighbors_of(t, "A") == ["E", "B", "C"] and set(t[t["ba"] == "A"]["source"]) == {"distance"}
    # D has nothing within 400 km and no interchange -> no neighbours
    assert neighbors_of(t, "D") == []


# ---- cross-checks ----


def test_preserve_extremes_two_of_three_truth_table():
    from grid_sentinel.crosscheck import preserve_extremes

    flagged = np.array([True] * 8)
    above = np.array([True] * 7 + [False])
    profile = np.array([1, 1, 0, 0, 1, np.nan, np.nan, 0], dtype=float)
    neighbours = np.array([1, 0, 0, 1, np.nan, 1, np.nan, 0], dtype=float)
    weather = np.array([1, 1, 0, 0, np.nan, 0, np.nan, 0], dtype=float)
    keep = preserve_extremes(flagged, above, [profile, neighbours, weather])
    # 0 all confirm -> drop; 1 one fails -> drop; 2 all fail -> keep; 3 two fail -> keep;
    # 4 one available (not independent evidence) -> keep; 5 two available, one fails -> drop;
    # 6 none available -> keep; 7 not above -> unchanged
    assert list(keep) == [False, False, True, True, True, False, True, True]


def test_confirm_neighbors_flags_shared_rise():
    from grid_sentinel.crosscheck import confirm_neighbors

    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="h")
    base = 100 + 20 * np.sin(2 * np.pi * (np.arange(len(idx)) - 6) / 24)
    n1 = pd.Series(base.copy(), index=idx)
    n2 = pd.Series(2 * base, index=idx)  # a bigger neighbour, same shape
    n3 = pd.Series(0.5 * base, index=idx)
    for s in (n1, n2):
        s.iloc[-30] *= 1.5  # two of three rise at the same hour
    n3.iloc[-10] *= 1.5  # only one rises
    c = confirm_neighbors({"a": n1, "b": n2, "c": n3}, idx)
    assert c.iloc[-30] == 1.0 and c.iloc[-10] == 0.0
    assert np.isnan(c.iloc[0])  # no history yet


def test_confirm_weather_cold_tail_in_heating_regime():
    from grid_sentinel.crosscheck import confirm_weather

    t = seasonal_temperature()
    t.loc["2024-01-20"] = -5.0  # an arctic day
    c = confirm_weather(t)
    assert (c.loc["2024-01-20"] == 1.0).all()
    assert (c.loc["2024-01-10"] == 0.0).all()
    assert (c.loc["2024-07-15"] == 0.0).all()  # warm but ordinary


# ---- composite v0.3 ----


def synthetic_ba(seed=0, factor=1.6):
    """Two years of hourly load with a seasonal level, its temperature, two neighbours, one genuine arctic
    morning (load up everywhere, temperature in the cold tail) and one lone spike on a mild afternoon."""
    from grid_sentinel.rules import sentinel_v2

    idx = pd.date_range("2022-01-01", "2024-03-01 23:00", freq="h")
    t = np.arange(len(idx))
    doy = idx.dayofyear.to_numpy()
    rng = np.random.default_rng(seed)
    daily = 10_000 + 2_500 * np.sin(2 * np.pi * (t - 6) / 24)
    season = 1 + 0.15 * np.cos(2 * np.pi * doy / 365.25)
    load = pd.Series(daily * season + rng.normal(0, 60, len(t)), index=idx)
    temp = pd.Series(50 - 30 * np.cos(2 * np.pi * doy / 365.25) + rng.normal(0, 3, len(t)), index=idx)
    nb = {"n1": load * 0.5 + rng.normal(0, 30, len(t)), "n2": load * 2.0 + rng.normal(0, 100, len(t))}
    win = slice("2024-01-17 06:00", "2024-01-17 09:00")
    for s in (load, nb["n1"], nb["n2"]):
        s.loc[win] *= factor
    temp.loc["2024-01-17"] = -10.0
    load.loc["2024-02-10 15:00"] *= factor
    assert sentinel_v2(load).loc[win, "is_anomaly"].any(), "the fixture must be flagged by v0.2 to be meaningful"
    return load, temp, nb, win


def test_sentinel_v3_keeps_shared_cold_peak_and_drops_lone_spike():
    from grid_sentinel import sentinel_v3

    load, temp, nb, win = synthetic_ba()
    res = sentinel_v3(load, temperature_f=temp, neighbors=nb)
    assert not res.loc[win, "is_anomaly"].any()
    assert (res.loc[win, "confirmed_by"] == "extreme kept").any()
    assert res.loc["2024-02-10 15:00", "is_anomaly"]
    assert {"above_expected", "check_profile", "check_neighbors", "check_weather"} <= set(res.columns)
    off = sentinel_v3(load, temperature_f=temp, neighbors=nb, extremes="off")
    assert off.loc[win, "is_anomaly"].any()  # without the rule the cold morning is a fault
