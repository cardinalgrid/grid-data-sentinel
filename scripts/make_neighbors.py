"""Build docs/neighbors.csv (interchange partners with a distance fallback) and docs/neighbors_distance.csv
(distance only) from the EIA-930 interchange files of one year (default 2024)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from grid_sentinel.neighbors import download_interchange, interchange_pairs, neighbor_table
from grid_sentinel.stations import BA_STATION, ba_distances


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--tidy", type=Path, default=Path("../ba-forecast-scorecard/data/tidy"))
    ap.add_argument("--isd", type=Path, default=Path("../notes/03-daily-load-profiles/data/isd"))
    ap.add_argument("--raw", type=Path, default=Path("data/interchange"))
    ap.add_argument("--out", type=Path, default=Path("docs/neighbors.csv"))
    a = ap.parse_args()
    files = [download_interchange(a.year, h, a.raw) for h in ("Jan_Jun", "Jul_Dec")]
    pairs = interchange_pairs(files)
    has_load = set(pd.read_parquet(a.tidy / f"{a.year}H1.parquet", columns=["ba"])["ba"].unique()) & set(BA_STATION)
    dist = ba_distances(a.isd)
    table = neighbor_table(pairs, dist, has_load=has_load)
    table = table[table["ba"].isin(BA_STATION)].reset_index(drop=True)
    table.to_csv(a.out, index=False)
    by_distance = neighbor_table(pairs.iloc[0:0], dist, has_load=has_load)
    by_distance = by_distance[by_distance["ba"].isin(BA_STATION)].reset_index(drop=True)
    by_distance.to_csv(a.out.with_name("neighbors_distance.csv"), index=False)
    print(table.groupby("source").size().to_string())
    print(f"{table['ba'].nunique()} BAs with neighbours -> {a.out}; {by_distance['ba'].nunique()} by distance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
