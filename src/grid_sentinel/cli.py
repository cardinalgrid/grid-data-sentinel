"""Command line: ``grid-sentinel detect`` and ``grid-sentinel benchmark``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from grid_sentinel import __version__
from grid_sentinel.autoencoder import SparseAutoencoder
from grid_sentinel.baselines import hampel, iqr, rolling_zscore
from grid_sentinel.benchmark import DEFAULT_BAS, run, summarise, write_results
from grid_sentinel.repair import repair
from grid_sentinel.rules import sentinel, stuck_values
from grid_sentinel.teda import RecursiveTEDA

METHODS = ("sentinel", "teda", "teda-level", "autoencoder", "stuck", "zscore", "hampel", "iqr")


def _detect(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.csv)
    if args.column not in df.columns:
        print(f"column {args.column!r} not in {list(df.columns)}", file=sys.stderr)
        return 2
    s = df[args.column].astype(float)
    if args.method == "sentinel":
        res = sentinel(s)
    elif args.method == "stuck":
        res = stuck_values(s)
    elif args.method == "teda":
        res = RecursiveTEDA(m=args.m, diff=True).detect(s)
    elif args.method == "teda-level":
        res = RecursiveTEDA(m=args.m, diff=False).detect(s)
    elif args.method == "autoencoder":
        res = SparseAutoencoder().detect(s)
    elif args.method == "zscore":
        res = rolling_zscore(s)
    elif args.method == "hampel":
        res = hampel(s)
    else:
        res = iqr(s)
    out = df.copy()
    out["score"] = res["score"].to_numpy()
    out["is_anomaly"] = res["is_anomaly"].astype(int).to_numpy()
    if args.repair:
        fixed, log = repair(s, res["is_anomaly"].to_numpy(), reason=args.method)
        out[f"{args.column}_repaired"] = fixed.to_numpy()
        log.to_csv(Path(args.out).with_suffix(".audit.csv"))
    out.to_csv(args.out, index=False)
    print(f"{int(res['is_anomaly'].sum())} of {len(s)} readings flagged by {args.method}; wrote {args.out}")
    return 0


def _benchmark(args: argparse.Namespace) -> int:
    results = run(
        Path(args.tidy),
        bas=tuple(args.bas),
        years=tuple(args.years),
        rate=args.rate,
        seed=args.seed,
    )
    if results.empty:
        print("no series found; check --tidy, --bas and --years", file=sys.stderr)
        return 1
    write_results(results, Path(args.out), Path(args.tidy), args.bas, args.years, args.rate, args.seed)
    print(summarise(results).round(3).to_string())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="grid-sentinel", description="Anomaly detection for load telemetry.")
    p.add_argument("--version", action="version", version=f"grid-data-sentinel {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="flag anomalies in one column of a CSV file")
    d.add_argument("csv")
    d.add_argument("--column", default="demand")
    d.add_argument("--method", choices=METHODS, default="sentinel")
    d.add_argument("--m", type=float, default=3.0, help="TEDA sensitivity")
    d.add_argument("--repair", action="store_true", help="also write a repaired column and an audit log")
    d.add_argument("--out", default="detections.csv")
    d.set_defaults(func=_detect)

    b = sub.add_parser("benchmark", help="run all detectors on EIA-930 series with injected anomalies")
    b.add_argument("--tidy", required=True, help="directory with the scorecard's tidy parquet files")
    b.add_argument("--bas", nargs="+", default=list(DEFAULT_BAS))
    b.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    b.add_argument("--rate", type=float, default=0.005)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default="results")
    b.set_defaults(func=_benchmark)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
