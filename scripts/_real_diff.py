"""Readings flagged on the series as published by two composite configurations: counts by reason, and the
readings that only one of them flags."""
import json
import sys
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
from grid_sentinel.detectors import Sentinel
from grid_sentinel.pandas import from_series, regularize

TIDY = Path("C:/Users/Ricardo Guerra/cardinalgrid-data/tidy")
ISD = Path("C:/Users/Ricardo Guerra/cardinalgrid-data/isd")
kw = json.loads(sys.argv[1])
out = Path(sys.argv[2])
ctx_of = make_context(TIDY, ISD, Path("docs/neighbors.csv"))
rows, only = [], []
from grid_sentinel.stations import BA_STATION

BAS = sorted(BA_STATION) if len(sys.argv) > 3 and sys.argv[3] == "all" else DEFAULT_BAS
YEARS = (2023, 2024) if len(sys.argv) > 3 and sys.argv[3] == "all" else (2021, 2022, 2023, 2024)
for ba in BAS:
    for year in YEARS:
        try:
            s = load_series_local(TIDY, ba, year)
        except Exception as e:  # noqa: BLE001
            print(ba, year, "skipped", e, flush=True)
            continue
        s = s[s > 0]
        if len(s) < 24 * 60:
            continue
        reg = regularize(s)
        v, t = from_series(reg)
        ctx = context_from_dict(ctx_of(ba, year))
        a = Sentinel().predict(v, t, ctx)
        b = Sentinel(**kw).predict(v, t, ctx)
        rows.append({"ba": ba, "year": year, "hours": int(np.isfinite(v).sum()), "v04": int(a.is_anomaly.sum()), "new": int(b.is_anomaly.sum()),
                     "only_v04": int((a.is_anomaly & ~b.is_anomaly).sum()), "only_new": int((b.is_anomaly & ~a.is_anomaly).sum())})
        for i in np.flatnonzero(a.is_anomaly != b.is_anomaly):
            only.append({"ba": ba, "time": str(t[i])[:13], "value": v[i], "expected": a.expected[i], "prev": v[i - 1] if i else np.nan,
                         "next": v[i + 1] if i + 1 < len(v) else np.nan, "v04": a.reason[i], "new": b.reason[i],
                         "flag_v04": bool(a.is_anomaly[i]), "flag_new": bool(b.is_anomaly[i])})
        print(ba, year, rows[-1], flush=True)
d = pd.DataFrame(rows)
d.to_csv(out.with_suffix(".counts.csv"), index=False)
pd.DataFrame(only).to_csv(out.with_suffix(".diff.csv"), index=False)
print(d.sum(numeric_only=True))
