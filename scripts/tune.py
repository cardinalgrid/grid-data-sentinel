"""Sensitivity grid for the two main detectors on a subset of series; writes results/tuning.csv.

The defaults in ``grid_sentinel.benchmark.default_detectors`` were chosen from this table.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from grid_sentinel.autoencoder import SparseAutoencoder
from grid_sentinel.benchmark import run, summarise
from grid_sentinel.teda import RecursiveTEDA

ROOT = Path(__file__).resolve().parents[1]


def teda(m: float, diff: bool):
    return lambda s: RecursiveTEDA(m=m, diff=diff).detect(s)


def ae(factor: float, window: int = 4):
    return lambda s: SparseAutoencoder(window=window, encoding_dim=max(2, window // 4), epochs=20, factor=factor).detect(s)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tidy", required=True)
    p.add_argument("--bas", nargs="+", default=["PJM", "CISO", "TVA", "BPAT"])
    p.add_argument("--years", nargs="+", type=int, default=[2023])
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    dets = {}
    for m in (2.5, 3.0, 4.0, 5.0, 6.0):
        dets[f"teda_level_m{m}"] = teda(m, False)
        dets[f"teda_diff_m{m}"] = teda(m, True)
    for f in (3, 5, 8, 12, 20, 30):
        dets[f"autoencoder_w4_f{f}"] = ae(f)
    for w in (8, 24):
        dets[f"autoencoder_w{w}_f20"] = ae(20, w)
    res = run(Path(args.tidy), bas=tuple(args.bas), years=tuple(args.years), detectors=dets)
    table = summarise(res)
    out = ROOT / "results" / "tuning.csv"
    out.parent.mkdir(exist_ok=True)
    table.to_csv(out)
    print(table.round(3).to_string())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
