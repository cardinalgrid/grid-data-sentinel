"""Benchmark of all detectors on EIA-930 demand series with injected, labelled anomalies.

Input is the tidy parquet produced by ``ba-forecast-scorecard`` (one file per half-year with
columns ``ba``, ``utc_end``, ``demand``). For each balancing authority and year the hourly demand
series is corrupted with ``inject_anomalies`` and every detector is run on the corrupted series.
Metrics are computed against the injected labels with a tolerance of one reading.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from grid_sentinel import __version__
from grid_sentinel.autoencoder import SparseAutoencoder
from grid_sentinel.baselines import hampel, iqr, rolling_zscore
from grid_sentinel.metrics import score_labels
from grid_sentinel.rules import sentinel, stuck_values
from grid_sentinel.synthetic import inject_anomalies
from grid_sentinel.teda import RecursiveTEDA

DEFAULT_BAS = ("PJM", "MISO", "ERCO", "CISO", "SWPP", "NYIS", "ISNE", "TVA", "DUK", "BPAT")


def default_detectors() -> dict[str, Callable[[pd.Series], pd.DataFrame]]:
    return {
        "sentinel": lambda s: sentinel(s),
        "teda_level": lambda s: RecursiveTEDA(m=4.0, diff=False).detect(s),
        "teda_diff": lambda s: RecursiveTEDA(m=3.0, diff=True).detect(s),
        "autoencoder": lambda s: SparseAutoencoder(window=4, encoding_dim=2, epochs=20, factor=20.0).detect(s),
        "stuck_rule": lambda s: stuck_values(s, min_run=3),
        "rolling_zscore": lambda s: rolling_zscore(s, window=168, k=3.0),
        "hampel": lambda s: hampel(s, window=24, k=3.0),
        "iqr": lambda s: iqr(s, k=1.5),
    }


def load_series(tidy_dir: Path, ba: str, year: int) -> pd.Series:
    frames = []
    for half in ("H1", "H2"):
        f = tidy_dir / f"{year}{half}.parquet"
        if f.exists():
            d = pd.read_parquet(f, columns=["ba", "utc_end", "demand"])
            frames.append(d[d["ba"] == ba])
    if not frames:
        return pd.Series(dtype=float)
    d = pd.concat(frames).sort_values("utc_end")
    s = d.set_index("utc_end")["demand"].astype(float)
    return s[s > 0]


def run(
    tidy_dir: Path,
    bas: tuple[str, ...] = DEFAULT_BAS,
    years: tuple[int, ...] = (2023, 2024),
    rate: float = 0.005,
    seed: int = 0,
    detectors: dict[str, Callable[[pd.Series], pd.DataFrame]] | None = None,
    tolerance: int = 1,
) -> pd.DataFrame:
    detectors = detectors or default_detectors()
    rows = []
    for ba in bas:
        for year in years:
            s = load_series(tidy_dir, ba, year)
            if len(s) < 24 * 30:
                continue
            inj = inject_anomalies(s, rate=rate, seed=seed + year)
            for name, fn in detectors.items():
                t0 = time.perf_counter()
                res = fn(inj["value"])
                dt = time.perf_counter() - t0
                m = score_labels(inj["label"].to_numpy(), res["is_anomaly"].to_numpy(), tolerance=tolerance)
                by_kind = {}
                for kind in ("spike", "dip", "zero", "stuck", "scale"):
                    sel = (inj["kind"] == kind).to_numpy()
                    if sel.any():
                        hit = np.asarray(res["is_anomaly"], dtype=bool)
                        wide = hit.copy()
                        for d in range(1, tolerance + 1):
                            wide[d:] |= hit[:-d]
                            wide[:-d] |= hit[d:]
                        by_kind[f"recall_{kind}"] = float((wide & sel).sum() / sel.sum())
                rows.append({"ba": ba, "year": year, "detector": name, "hours": len(s), "seconds": dt, **m, **by_kind})
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    cols = ["precision", "recall", "f1", "mcc", "seconds"] + [c for c in results.columns if c.startswith("recall_")]
    agg = results.groupby("detector")[cols].mean().sort_values("f1", ascending=False)
    agg["series"] = results.groupby("detector").size()
    return agg


def write_results(results: pd.DataFrame, out_dir: Path, tidy_dir: Path, bas, years, rate, seed) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "benchmark_by_series.csv", index=False)
    summary = summarise(results)
    summary.to_csv(out_dir / "benchmark_summary.csv")
    meta = {
        "version": __version__,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bas": list(bas),
        "years": list(years),
        "injection_rate": rate,
        "seed": seed,
        "tolerance": 1,
        "n_series": int(results.groupby(["ba", "year"]).ngroups),
        "detectors": {k: {c: float(v) for c, v in row.items()} for k, row in summary.iterrows()},
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
