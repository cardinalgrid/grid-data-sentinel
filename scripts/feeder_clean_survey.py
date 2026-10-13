"""What FeederLoss flags on the series as published (nothing injected): one row per flagged segment.

    python scripts/feeder_clean_survey.py --tidy <tidy> --isd <isd> --out results/v05/clean_segments.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from grid_sentinel.benchmark import DEFAULT_BAS, context_from_dict, load_series_local, make_context
from grid_sentinel.detectors import FeederLoss
from grid_sentinel.pandas import from_series, regularize


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tidy", required=True)
    ap.add_argument("--isd", required=True)
    ap.add_argument("--neighbors", default="docs/neighbors.csv")
    ap.add_argument("--years", nargs="+", type=int, default=[2023, 2024])
    ap.add_argument("--kw", default="{}")
    ap.add_argument("--all-bas", action="store_true", help="every BA with a weather station, not only the benchmark ten")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    ctx_of = make_context(Path(a.tidy), Path(a.isd), Path(a.neighbors))
    rows = []
    from grid_sentinel.stations import BA_STATION

    for ba in (sorted(BA_STATION) if a.all_bas else DEFAULT_BAS):
        for year in a.years:
            try:
                s = load_series_local(Path(a.tidy), ba, year)
            except Exception as e:  # noqa: BLE001
                print(ba, year, "skipped", e, flush=True)
                continue
            s = s[s > 0]
            if len(s) < 24 * 60:
                continue
            reg = regularize(s)
            v, t = from_series(reg)
            det = FeederLoss(**json.loads(a.kw))
            res = det.predict(v, t, context_from_dict(ctx_of(ba, year)))
            u, scale, expected = det.steps(v, t, context_from_dict(ctx_of(ba, year)))
            f = res.is_anomaly
            i = 0
            while i < len(f):
                if f[i]:
                    j = i
                    while j + 1 < len(f) and f[j + 1]:
                        j += 1
                    rows.append({"ba": ba, "start": str(t[i]), "hours": j - i + 1, "u_start": float(u[i]),
                                 "u_end": float(u[j + 1]) if j + 1 < len(u) else np.nan, "scale": float(scale[i]),
                                 "weekday": pd.Timestamp(t[i]).day_name(), "hour": pd.Timestamp(t[i]).hour,
                                 "mw_before": float(v[i - 1]), "mw_first": float(v[i])})
                    i = j + 1
                else:
                    i += 1
            print(ba, year, int(f.sum()), "flagged", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    print(d.round(4).to_string())
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    raise SystemExit(main())
