"""Tuning loop for FeederLoss: share of dense injected losses found (per-series seeds) and readings flagged on the
series as published, for several keyword sets, one process per set."""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grid_sentinel.benchmark import (
    DEFAULT_BAS,
    context_from_dict,
    load_series_local,
    make_context,
)
from grid_sentinel.detectors import FeederLoss
from grid_sentinel.pandas import from_series, regularize
from grid_sentinel.synthetic import inject_anomalies

# data folders from the environment: GRID_SENTINEL_TIDY (scorecard tidy files) and GRID_SENTINEL_ISD (ISD-Lite cache)
TIDY = Path(os.environ["GRID_SENTINEL_TIDY"])
ISD = Path(os.environ["GRID_SENTINEL_ISD"])
SEEDS = (0, 1)


def one(item):
    name, kw = item
    np.seterr(all="ignore")
    import warnings
    warnings.filterwarnings("ignore")
    ctx_of = make_context(TIDY, ISD, Path("docs/neighbors.csv"))
    hits = events = clean_flags = 0
    for bi, ba in enumerate(DEFAULT_BAS):
        for year in (2023, 2024):
            s = load_series_local(TIDY, ba, year)
            s = s[s > 0]
            ctx = context_from_dict(ctx_of(ba, year))
            reg = regularize(s)
            v, t = from_series(reg)
            clean_flags += int(FeederLoss(**kw).predict(v, t, ctx).is_anomaly.sum())
            for seed in SEEDS:
                inj = inject_anomalies(s, rate=0.02, types=("feeder_loss",), seed=1000 * seed + 37 * bi + year)
                reg = regularize(inj["value"])
                v, t = from_series(reg)
                f = FeederLoss(**kw).predict(v, t, ctx).is_anomaly
                pos = reg.index.get_indexer(pd.DatetimeIndex(inj.index))
                kind = inj["kind"].to_numpy()
                i = 0
                while i < len(kind):
                    if kind[i] == "feeder_loss":
                        j = i
                        while j + 1 < len(kind) and kind[j + 1] == "feeder_loss":
                            j += 1
                        events += 1
                        hits += f[pos[i]:pos[j] + 1].mean() > 0.5
                        i = j + 1
                    else:
                        i += 1
    return {"variant": name, "events": events, "found": round(hits / events, 3), "clean_flagged": clean_flags, "kw": json.dumps(kw)}


if __name__ == "__main__":
    variants = json.loads(sys.argv[1])
    with Pool(min(len(variants), int(sys.argv[2]) if len(sys.argv) > 2 else 4)) as p:
        rows = p.map(one, list(variants.items()))
    d = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(d.to_string(index=False))
    out = Path("results/v05/tune_feeder.csv")
    d.to_csv(out, mode="a", header=not out.exists(), index=False)
