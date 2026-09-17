"""Representative NOAA station for each balancing authority.

One airport per BA (the main load centre, or the largest city in the service area), the same
assignment used in Cardinal Grid Note 2. Station identifiers and coordinates come from the public
ISD history file (https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv).
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import requests

HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"

# BA -> ICAO of the representative station.
BA_STATION = {
    "AEC": "KMGM", "AECI": "KSGF", "AVA": "KGEG", "AZPS": "KPHX", "BANC": "KSMF", "BPAT": "KPDX",
    "CISO": "KLAX", "CPLE": "KRDU", "CPLW": "KAVL", "DUK": "KCLT", "EPE": "KELP", "ERCO": "KDFW",
    "FMPP": "KMCO", "FPC": "KMCO", "FPL": "KMIA", "GCPD": "KMWH", "IPCO": "KBOI", "ISNE": "KBOS",
    "JEA": "KJAX", "LDWP": "KLAX", "LGEE": "KSDF", "MISO": "KIND", "NEVP": "KLAS", "NWMT": "KBIL",
    "NYIS": "KJFK", "PACE": "KSLC", "PACW": "KPDX", "PGE": "KPDX", "PJM": "KPHL", "PNM": "KABQ",
    "PSCO": "KDEN", "PSEI": "KSEA", "SC": "KCHS", "SCEG": "KCAE", "SCL": "KSEA", "SOCO": "KATL",
    "SRP": "KPHX", "SWPP": "KOKC", "TEC": "KTPA", "TEPC": "KTUS", "TPWR": "KSEA", "TVA": "KBNA",
    "WACM": "KDEN", "WALC": "KPHX",
}


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (haversine), in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def station_table(isd_dir: Path) -> pd.DataFrame:
    """One row per station in ``BA_STATION``: icao, usaf, wban, name, state, lat, lon.

    Downloads the ISD history file into ``isd_dir`` when it is not there.
    """
    isd_dir = Path(isd_dir)
    hist = isd_dir / "isd-history.csv"
    if not hist.exists():
        isd_dir.mkdir(parents=True, exist_ok=True)
        r = requests.get(HISTORY_URL, timeout=120)
        r.raise_for_status()
        hist.write_bytes(r.content)
    h = pd.read_csv(hist, dtype=str)
    h = h[(h["CTRY"] == "US") & h["ICAO"].isin(set(BA_STATION.values())) & (h["USAF"] != "999999")].copy()
    h["BEGIN"] = h["BEGIN"].astype(int)
    h["END"] = h["END"].astype(int)
    h = h[(h["BEGIN"] <= 20150701) & (h["END"] >= 20250101)].sort_values("BEGIN").groupby("ICAO").head(1)
    out = h.rename(
        columns={"ICAO": "icao", "USAF": "usaf", "WBAN": "wban", "STATION NAME": "name", "STATE": "state",
                 "LAT": "lat", "LON": "lon"}
    )[["icao", "usaf", "wban", "name", "state", "lat", "lon"]].copy()
    out["lat"] = out["lat"].astype(float)
    out["lon"] = out["lon"].astype(float)
    return out.reset_index(drop=True)


def ba_distances(isd_dir: Path) -> pd.DataFrame:
    """Distance in km between the representative stations of every pair of BAs with a resolved station."""
    t = station_table(isd_dir).set_index("icao")
    bas = [b for b, icao in BA_STATION.items() if icao in t.index]
    m = pd.DataFrame(0.0, index=bas, columns=bas)
    for a in bas:
        sa = t.loc[BA_STATION[a]]
        for b in bas:
            if a != b:
                sb = t.loc[BA_STATION[b]]
                m.loc[a, b] = distance_km(sa["lat"], sa["lon"], sb["lat"], sb["lon"])
    return m
