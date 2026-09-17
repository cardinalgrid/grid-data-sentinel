"""Neighbours of a balancing authority: the BAs it exchanges power with, from the public EIA-930
interchange record, with a distance-based fallback for BAs that report fewer than two partners.

The weight of a pair is the median absolute hourly interchange over the files given (one year in the
published table). Adjacency is a property of the network and is treated as fixed across years.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

INTERCHANGE_URL = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_INTERCHANGE_{year}_{half}.csv"


def download_interchange(year: int, half: str, dest_dir: Path) -> Path:
    """``half`` is ``Jan_Jun`` or ``Jul_Dec``. Files are about 100 MB each; cached in ``dest_dir``."""
    dest = Path(dest_dir) / f"EIA930_INTERCHANGE_{year}_{half}.csv"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(INTERCHANGE_URL.format(year=year, half=half), stream=True, timeout=600) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(".part")
        with open(tmp, "wb") as f:
            f.writelines(r.iter_content(chunk_size=1 << 20))
        tmp.replace(dest)
    return dest


def interchange_pairs(csv_paths: list[Path]) -> pd.DataFrame:
    """One row per (ba, neighbor): ``weight_mw`` (median absolute hourly interchange) and ``hours``."""
    frames = []
    for p in csv_paths:
        d = pd.read_csv(
            p, usecols=["Balancing Authority", "Directly Interconnected Balancing Authority", "Interchange (MW)"],
            dtype=str,
        )
        d.columns = ["ba", "neighbor", "mw"]
        d["mw"] = pd.to_numeric(d["mw"].str.replace(",", "", regex=False), errors="coerce")
        frames.append(d.dropna(subset=["mw"]))
    d = pd.concat(frames, ignore_index=True)
    d["abs_mw"] = d["mw"].abs()
    return d.groupby(["ba", "neighbor"])["abs_mw"].agg(weight_mw="median", hours="size").reset_index()


def neighbor_table(
    pairs: pd.DataFrame, distances: pd.DataFrame, has_load: set[str], min_neighbors: int = 2,
    max_km: float = 400.0, max_distance_neighbors: int = 5,
) -> pd.DataFrame:
    """Neighbours per BA: interchange partners with load data (heaviest first) when there are at least
    ``min_neighbors``; otherwise the BAs whose stations lie within ``max_km`` (nearest first, at most
    ``max_distance_neighbors``). ``source`` records which rule produced each row; ``weight`` is MW for
    interchange rows and km for distance rows."""
    rows = []
    bas = sorted(set(pairs["ba"]) | set(distances.index))
    for ba in bas:
        p = pairs[(pairs["ba"] == ba) & pairs["neighbor"].isin(has_load) & (pairs["neighbor"] != ba)
                  & (pairs["weight_mw"] > 0)]  # a tie that is idle half the time is not a neighbour
        p = p.sort_values("weight_mw", ascending=False)
        if len(p) >= min_neighbors:
            rows += [{"ba": ba, "neighbor": r.neighbor, "source": "interchange", "weight": float(r.weight_mw)}
                     for r in p.itertuples()]
            continue
        if ba in distances.index:
            d = distances.loc[ba].drop(ba)
            d = d[(d <= max_km) & d.index.isin(has_load)].sort_values().head(max_distance_neighbors)
            rows += [{"ba": ba, "neighbor": n, "source": "distance", "weight": float(km)} for n, km in d.items()]
    return pd.DataFrame(rows, columns=["ba", "neighbor", "source", "weight"])


def neighbors_of(table: pd.DataFrame, ba: str) -> list[str]:
    return list(table[table["ba"] == ba]["neighbor"])


def load_neighbor_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
