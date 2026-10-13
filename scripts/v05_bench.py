"""Phase v0.5 benchmark: the composite, the feeder-loss detector and their union, on the standard injection (five
fault types) and on the injection with the feeder-loss fault, for the given seeds and years.

    python scripts/v05_bench.py --tidy <tidy> --isd <isd> --seeds 0 --years 2023 2024 --out results/v05/dev
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grid_sentinel.benchmark import DEFAULT_BAS, make_context, on_contract, run, summarise
from grid_sentinel.detectors import FeederLoss, Sentinel
from grid_sentinel.synthetic import ALL_TYPES, TYPES


def union(*fns):
    """Flags of several detectors together; the first detector's reason wins where both flag."""

    def fn(s, ctx):
        outs = [f(s, ctx) for f in fns]
        res = outs[0].copy()
        for o in outs[1:]:
            add = o["is_anomaly"].to_numpy(bool) & ~res["is_anomaly"].to_numpy(bool)
            res.loc[add, "reason"] = o.loc[add, "reason"]
            res["is_anomaly"] = res["is_anomaly"].to_numpy(bool) | o["is_anomaly"].to_numpy(bool)
        return res

    return fn


def composite_variants(variants: dict) -> dict:
    """Front A: the composite with different keyword arguments."""
    out = {"sentinel_v3": on_contract(Sentinel)}
    for name, kw in variants.items():
        out[f"sentinel_v3_{name}"] = on_contract(lambda kw=kw: Sentinel(**kw))
    return out


def detectors(variants: dict, base_kw: dict | None = None) -> dict:
    base = on_contract(lambda: Sentinel(**(base_kw or {})))
    out = {"sentinel_v3": on_contract(Sentinel)}
    if base_kw:
        out["sentinel_base"] = base
    for name, kw in variants.items():
        out[f"feeder_{name}"] = on_contract(lambda kw=kw: FeederLoss(**kw))
        out[f"{'sentinel_base' if base_kw else 'sentinel_v3'}+feeder_{name}"] = union(base, on_contract(lambda kw=kw: FeederLoss(**kw)))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tidy", required=True)
    ap.add_argument("--isd", required=True)
    ap.add_argument("--neighbors", default="docs/neighbors.csv")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    ap.add_argument("--variants", default='{"default": {}}', help="JSON: name -> FeederLoss keyword arguments")
    ap.add_argument("--front", choices=["feeder", "composite"], default="feeder")
    ap.add_argument("--base", default="{}", help="JSON: Sentinel keyword arguments of the composite the feeder joins")
    ap.add_argument("--only", nargs="*", default=None, help="benchmarks to run: standard feeder feeder_dense")
    ap.add_argument("--per-series-seed", action="store_true", help="a different injection for each balancing authority")
    ap.add_argument("--dense-rate", type=float, default=0.02, help="rate of the feeder-loss-only benchmark")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = make_context(Path(a.tidy), Path(a.isd), Path(a.neighbors))
    if a.front == "feeder":
        dets = detectors(json.loads(a.variants), json.loads(a.base))
    else:
        dets = composite_variants(json.loads(a.variants))
    benches = (("standard", TYPES, 0.005), ("feeder", ALL_TYPES, 0.005), ("feeder_dense", ("feeder_loss",), a.dense_rate))
    for label, types, rate in benches:
        if a.only and label not in a.only:
            continue
        frames = []
        for seed in a.seeds:
            r = run(Path(a.tidy), bas=DEFAULT_BAS, years=tuple(a.years), seed=seed, context=ctx, detectors=dets, types=types,
                    rate=rate, per_series_seed=a.per_series_seed)
            r["seed"] = seed
            frames.append(r)
            print(label, "seed", seed, "done", flush=True)
        res = pd.concat(frames)
        res.to_csv(out / f"{label}_by_series.csv", index=False)
        summ = summarise(res)
        summ.to_csv(out / f"{label}_summary.csv")
        cols = [c for c in ["precision_adj", "recall", "f1_adj", "recall_spike", "recall_dip", "recall_zero", "recall_stuck",
                            "recall_scale", "recall_feeder_loss", "seconds"] if c in summ.columns]
        print(f"\n== {label}\n" + summ[cols].round(4).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    raise SystemExit(main())
