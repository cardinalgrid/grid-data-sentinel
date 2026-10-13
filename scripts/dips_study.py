"""Diagnosis of the faults the composite misses: single dips by size and partial feeder losses by depth and length.

    python scripts/dips_study.py --tidy <tidy> --isd <isd> --out results/dips_study

Injects the six fault types (seed 0) into the benchmark series, runs the composite and the profile residual,
and writes, per injected event, its kind, size, length, hour, the load level, whether the composite caught it,
and the profile-based statistics a new detector could use: the residual of the event in log against the
expected profile, the log jump at its edges against the expected shape, and the robust spread of both on the
clean hours of the same series.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from grid_sentinel.benchmark import DEFAULT_BAS, load_series_local, make_context
from grid_sentinel.detectors import ProfileResidual, Sentinel
from grid_sentinel.benchmark import on_contract
from grid_sentinel.synthetic import ALL_TYPES, TYPES, inject_anomalies


def events(inj: pd.DataFrame) -> list[tuple[int, int, str]]:
    lab = inj["label"].to_numpy()
    kind = inj["kind"].to_numpy()
    out, i = [], 0
    while i < len(lab):
        if lab[i]:
            j = i
            while j + 1 < len(lab) and lab[j + 1] and kind[j + 1] == kind[i]:
                j += 1
            out.append((i, j, kind[i]))
            i = j + 1
        else:
            i += 1
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tidy", required=True)
    ap.add_argument("--isd", required=True)
    ap.add_argument("--neighbors", default="docs/neighbors.csv")
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--types", choices=["all", "standard"], default="all")
    ap.add_argument("--out", default="results/dips_study")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx_of = make_context(Path(a.tidy), Path(a.isd), Path(a.neighbors))
    sentinel = on_contract(Sentinel)
    profile = on_contract(lambda: ProfileResidual(k=4.0, regime="temperature"))
    rows, noise = [], []
    types = ALL_TYPES if a.types == "all" else TYPES
    for bi, ba in enumerate(DEFAULT_BAS):
        for year in a.years:
          for seed in a.seeds:
            s = load_series_local(Path(a.tidy), ba, year)
            s = s[s > 0]
            if len(s) < 24 * 30:
                continue
            inj = inject_anomalies(s, rate=0.005, types=types, seed=1000 * seed + 37 * bi + year)
            ctx = ctx_of(ba, year)
            sres = sentinel(inj["value"], ctx)
            det = profile(inj["value"], ctx)
            # expected from a profile run on the corrupted values (what a detector would see)
            from grid_sentinel.detectors.profile import ProfileResidual as PR  # noqa: F401
            pr = ProfileResidual(k=4.0, regime="temperature")
            from grid_sentinel.pandas import regularize
            from grid_sentinel.pandas import from_series
            reg = regularize(inj["value"])
            v, t = from_series(reg)
            temp = ctx.get("temperature_f")
            from grid_sentinel.benchmark import context_from_dict
            res = pr.predict(v, t, context_from_dict(ctx) if ctx else None)
            pos = reg.index.get_indexer(pd.DatetimeIndex(inj.index))
            expected = res.expected[pos] if res.expected is not None else np.full(len(inj), np.nan)
            val = inj["value"].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.log(val / expected)
                lv = np.log(val)
                le = np.log(expected)
            u = np.r_[np.nan, np.diff(lv) - np.diff(le)]
            clean = inj["label"].to_numpy() == 0
            sig_r = 1.4826 * np.nanmedian(np.abs(r[clean] - np.nanmedian(r[clean])))
            sig_u = 1.4826 * np.nanmedian(np.abs(u[clean] - np.nanmedian(u[clean])))
            noise.append({"ba": ba, "year": year, "sigma_r": sig_r, "sigma_u": sig_u})
            caught = sres["is_anomaly"].to_numpy(bool)
            hours = pd.DatetimeIndex(inj.index).hour
            for i, j, k in events(inj):
                ratio = float(np.nanmedian(val[i:j + 1] / inj["clean"].to_numpy(float)[i:j + 1]))
                rows.append({
                    "ba": ba, "year": year, "kind": k, "start": i, "length": j - i + 1, "hour": int(hours[i]),
                    "depth": 1.0 - ratio, "level_mw": float(np.nanmedian(inj["clean"].to_numpy(float)[i:j + 1])),
                    "caught_any": bool(caught[max(0, i - 1):j + 2].any()),
                    "caught_share": float(caught[i:j + 1].mean()),
                    "r_mean": float(np.nanmean(r[i:j + 1])), "u_start": float(u[i]),
                    "u_end": float(u[j + 1]) if j + 1 < len(u) else np.nan,
                    "sigma_r": sig_r, "sigma_u": sig_u,
                })
            print(ba, year, "events", sum(1 for _ in events(inj)), "sigma_r %.4f sigma_u %.4f" % (sig_r, sig_u), flush=True)
    ev = pd.DataFrame(rows)
    ev.to_csv(out / "events.csv", index=False)
    pd.DataFrame(noise).to_csv(out / "noise.csv", index=False)
    for k in ("dip", "feeder_loss"):
        e = ev[ev.kind == k].copy()
        if e.empty:
            continue
        e["depth_bin"] = pd.cut(e["depth"], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.95])
        print(f"\n{k}: {len(e)} events, caught {e.caught_any.mean():.2f}")
        print(e.groupby("depth_bin", observed=True).agg(n=("caught_any", "size"), caught=("caught_any", "mean"),
                                                         u_start=("u_start", "median"), r_mean=("r_mean", "median")).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
