"""Access to the tidy EIA-930 parquet files produced by ba-forecast-scorecard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def tidy_frame(tidy_dir: Path, ba: str, year: int, columns: tuple[str, ...]) -> pd.DataFrame:
    """Rows of one BA and one year from the half-year parquet files, sorted and de-duplicated on local hour."""
    frames = []
    for half in ("H1", "H2"):
        f = Path(tidy_dir) / f"{year}{half}.parquet"
        if f.exists():
            d = pd.read_parquet(f, columns=["ba", *columns])
            frames.append(d[d["ba"] == ba])
    if not frames:
        return pd.DataFrame(columns=["ba", *columns])
    return pd.concat(frames).sort_values("local_end").drop_duplicates("local_end")


def load_series_local(tidy_dir: Path, ba: str, year: int) -> pd.Series:
    """Hourly demand indexed by local hour beginning (naive), as reported in EIA-930; zeros are kept."""
    d = tidy_frame(tidy_dir, ba, year, ("local_end", "demand"))
    if d.empty:
        return pd.Series(dtype=float)
    s = d.set_index("local_end")["demand"].astype(float)
    s.index = pd.DatetimeIndex(s.index) - pd.Timedelta(hours=1)  # hour ending -> hour beginning
    return s


def utc_offset(tidy_dir: Path, ba: str, year: int) -> pd.Series:
    """Hours behind UTC for every local hour of the series (5 for Eastern standard time, 4 in summer)."""
    d = tidy_frame(tidy_dir, ba, year, ("local_end", "utc_end"))
    if d.empty:
        return pd.Series(dtype=float)
    utc = pd.to_datetime(d["utc_end"], utc=True).dt.tz_localize(None)
    off = (utc - pd.to_datetime(d["local_end"])).dt.total_seconds() / 3600.0
    off.index = pd.DatetimeIndex(d["local_end"]) - pd.Timedelta(hours=1)
    return off.astype(float)
