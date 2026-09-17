"""Hourly temperature per balancing authority from NOAA ISD-Lite, aligned to the BA's local hours.

ISD-Lite files (https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/) are fixed-width text with one row
per hour in UTC; air temperature is in tenths of a degree Celsius and -9999 is missing. Local time is
taken from the UTC offset that the EIA-930 record itself carries for each hour, so the temperature
series shares the index of the load series.
"""

from __future__ import annotations

import gzip
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from grid_sentinel.data import load_series_local, utc_offset
from grid_sentinel.stations import BA_STATION, station_table

ISD_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{usaf}-{wban}-{year}.gz"


def read_isd_lite(path: Path) -> pd.DataFrame:
    """Columns ``utc`` (tz-aware) and ``temp_c``; rows with missing temperature are dropped."""
    with gzip.open(path, "rt") as fh:
        txt = fh.read()
    df = pd.read_csv(
        io.StringIO(txt), sep=r"\s+", header=None, usecols=[0, 1, 2, 3, 4],
        names=["y", "m", "d", "h", "temp"], engine="python",
    )
    df = df[df["temp"] != -9999]
    utc = pd.to_datetime({"year": df["y"], "month": df["m"], "day": df["d"], "hour": df["h"]}, utc=True)
    return pd.DataFrame({"utc": utc.to_numpy(), "temp_c": df["temp"].to_numpy() / 10.0})


def fetch_isd_lite(usaf: str, wban: str, year: int, isd_dir: Path) -> Path | None:
    """Download one station-year into ``isd_dir`` unless cached; ``None`` when the file does not exist."""
    dest = Path(isd_dir) / f"{usaf}-{wban}-{year}.gz"
    if dest.exists():
        return dest
    for attempt in range(3):
        r = requests.get(ISD_URL.format(year=year, usaf=usaf, wban=wban), timeout=120)
        if r.status_code == 200:
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(dest)  # a file that exists is always a complete download
            return dest
        if r.status_code == 404:
            return None
        time.sleep(3 * (attempt + 1))
    return None


def hourly_temperature_f(ba: str, year: int, tidy_dir: Path, isd_dir: Path, years_back: int = 0) -> pd.Series:
    """Temperature in Fahrenheit at the BA's station, indexed like ``load_series_local`` for ``year``
    (and, with ``years_back``, the years before it, concatenated)."""
    st = station_table(isd_dir).set_index("icao").loc[BA_STATION[ba]]
    parts = []
    for y in range(year - years_back, year + 1):
        p = fetch_isd_lite(st["usaf"], st["wban"], y, isd_dir)
        if p is not None:
            parts.append(read_isd_lite(p))
    if parts:
        obs = pd.concat(parts, ignore_index=True).drop_duplicates("utc").set_index("utc")["temp_c"].sort_index()
    else:
        obs = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    out = []
    for y in range(year - years_back, year + 1):
        s = load_series_local(tidy_dir, ba, y)
        if s.empty:
            continue
        off = utc_offset(tidy_dir, ba, y).reindex(s.index)
        off = off.fillna(off.median())
        utc = pd.DatetimeIndex(s.index + pd.to_timedelta(off.to_numpy(), unit="h")).tz_localize("UTC")
        vals = obs.reindex(utc).to_numpy(dtype=float)
        out.append(pd.Series(vals * 9.0 / 5.0 + 32.0, index=s.index))
    return pd.concat(out) if out else pd.Series(dtype=float)


def daily_mean_f(temp_hourly: pd.Series, min_hours: int = 18) -> pd.Series:
    """Daily mean of an hourly series, only for days with at least ``min_hours`` readings."""
    g = temp_hourly.groupby(pd.DatetimeIndex(temp_hourly.index).normalize())
    m = g.mean()
    n = g.count()
    return m[n >= min_hours]


def temperature_tail(
    temp_hourly: pd.Series, half_weeks: int = 3, min_years: int = 2, fallback_weeks: int = 8
) -> pd.DataFrame:
    """Per timestamp, the 5th and 95th percentiles of the hourly temperature in the same calendar weeks
    (+/- ``half_weeks``) of the previous years of the series; when fewer than ``min_years`` previous years
    exist, the percentiles of the previous ``fallback_weeks`` weeks. NaN when fewer than a week of readings
    is available for the estimate."""
    t = temp_hourly.astype(float)
    idx = pd.DatetimeIndex(t.index)
    days = idx.normalize().unique()
    ords = np.array([d.toordinal() for d in days.date])
    day_pos = pd.Series(np.arange(len(days)), index=days).reindex(idx.normalize()).to_numpy()
    vals = t.to_numpy()
    p05 = np.full(len(days), np.nan)
    p95 = np.full(len(days), np.nan)
    half = 7 * half_weeks
    first_year = int(idx.year.min())
    for i, d in enumerate(days):
        years_avail = d.year - first_year
        if years_avail >= min_years:
            m = np.zeros(len(days), dtype=bool)
            for y in range(1, years_avail + 1):
                c = ords[i] - round(365.25 * y)
                m |= (ords >= c - half) & (ords <= c + half)
        else:
            m = (ords >= ords[i] - 7 * fallback_weeks) & (ords < ords[i])
        v = vals[np.isin(day_pos, np.flatnonzero(m))]
        v = v[np.isfinite(v)]
        if len(v) >= 24 * 7:
            p05[i], p95[i] = np.percentile(v, [5, 95])
    per_day = pd.DataFrame({"p05": p05, "p95": p95}, index=days)
    out = per_day.reindex(idx.normalize())
    out.index = idx
    return out
