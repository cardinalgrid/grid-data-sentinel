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
from grid_sentinel.baselines import hampel, iqr, modified_zscore, relative_deviation, rolling_zscore
from grid_sentinel.data import load_series_local
from grid_sentinel.metrics import score_labels
from grid_sentinel.profile import profile_residual
from grid_sentinel.rules import sentinel, sentinel_v2, sentinel_v3, stuck_values
from grid_sentinel.synthetic import inject_anomalies
from grid_sentinel.teda import RecursiveTEDA

DEFAULT_BAS = ("PJM", "MISO", "ERCO", "CISO", "SWPP", "NYIS", "ISNE", "TVA", "DUK", "BPAT")


def default_detectors() -> dict[str, Callable[[pd.Series, dict], pd.DataFrame]]:
    """Every detector takes the series and a context dict (``temperature_f``, ``neighbors``,
    ``neighbors_distance``); detectors that need no context ignore it."""
    return {
        "sentinel_v3": lambda s, c: sentinel_v3(s, temperature_f=c.get("temperature_f"), neighbors=c.get("neighbors")),
        "sentinel_v3_distance_neighbors": lambda s, c: sentinel_v3(
            s, temperature_f=c.get("temperature_f"), neighbors=c.get("neighbors_distance")
        ),
        "sentinel_v3_profile_neighbors": lambda s, c: sentinel_v3(
            s, neighbors=c.get("neighbors"), checks=("profile", "neighbors")
        ),
        "sentinel_v3_profile_weather": lambda s, c: sentinel_v3(
            s, temperature_f=c.get("temperature_f"), checks=("profile", "weather")
        ),
        "sentinel_v3_regime_only": lambda s, c: sentinel_v3(s, temperature_f=c.get("temperature_f"), extremes="off"),
        "sentinel_v2": lambda s, c: sentinel_v2(s),
        "profile_residual": lambda s, c: profile_residual(s, regime=None),
        "modified_zscore": lambda s, c: modified_zscore(s),
        "relative_deviation": lambda s, c: relative_deviation(s),
        "sentinel": lambda s, c: sentinel(s),
        "teda_level": lambda s, c: RecursiveTEDA(m=4.0, diff=False).detect(s),
        "teda_level_robust": lambda s, c: RecursiveTEDA(m=4.0, diff=False, robust=True).detect(s),
        "teda_diff": lambda s, c: RecursiveTEDA(m=3.0, diff=True).detect(s),
        "autoencoder": lambda s, c: SparseAutoencoder(window=4, encoding_dim=2, epochs=20, factor=20.0).detect(s),
        "stuck_rule": lambda s, c: stuck_values(s, min_run=3),
        "rolling_zscore": lambda s, c: rolling_zscore(s, window=168, k=3.0),
        "hampel": lambda s, c: hampel(s, window=24, k=3.0),
        "iqr": lambda s, c: iqr(s, k=1.5),
    }


def make_context(tidy_dir: Path, isd_dir: Path, neighbor_table_path: Path) -> Callable[[str, int], dict]:
    """Per (ba, year): the BA's hourly temperature (two previous years included, for the seasonal tails)
    and the local-hour load of its neighbours, from ``docs/neighbors.csv`` and, when present next to it,
    ``docs/neighbors_distance.csv``."""
    from grid_sentinel.neighbors import load_neighbor_table, neighbors_of
    from grid_sentinel.weather import hourly_temperature_f

    table = load_neighbor_table(neighbor_table_path)
    dist_path = Path(neighbor_table_path).with_name("neighbors_distance.csv")
    table_distance = load_neighbor_table(dist_path) if dist_path.exists() else table.iloc[0:0]

    def series_of(names: list[str], year: int) -> dict[str, pd.Series]:
        out = {}
        for n in names:
            s = load_series_local(tidy_dir, n, year)
            s = s[s > 0]
            if len(s) > 24 * 30:
                out[n] = s
        return out

    def ctx(ba: str, year: int) -> dict:
        try:
            temp = hourly_temperature_f(ba, year, tidy_dir, isd_dir, years_back=2)
        except KeyError:
            temp = None
        return {
            "temperature_f": temp,
            "neighbors": series_of(neighbors_of(table, ba), year),
            "neighbors_distance": series_of(neighbors_of(table_distance, ba), year),
        }

    return ctx


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


def raw_fault_mask(clean: np.ndarray) -> np.ndarray:
    """Readings of the source series that carry one of the survey signatures (frozen run of 3+, non-positive,
    or outside 1/5..5x of the 14-day rolling median), widened by one reading. Used to compute a precision
    that does not count a detector's hit on a real fault of the public data as a false alarm."""
    x = np.asarray(clean, dtype=float)
    n = len(x)
    same = np.zeros(n, dtype=bool)
    same[1:] = (x[1:] == x[:-1]) & (x[1:] > 0)
    run = np.zeros(n, dtype=int)
    for i in range(1, n):
        run[i] = run[i - 1] + 1 if same[i] else 0
    frozen = run >= 2
    for i in range(n - 1, 0, -1):
        if frozen[i] and same[i]:
            frozen[i - 1] = True
    med = pd.Series(x).rolling(24 * 14, min_periods=48, center=True).median().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = x / med
    m = frozen | ~(x > 0) | (ratio > 5) | (ratio < 0.2)
    wide = m.copy()
    wide[1:] |= m[:-1]
    wide[:-1] |= m[1:]
    return wide


def run(
    tidy_dir: Path,
    bas: tuple[str, ...] = DEFAULT_BAS,
    years: tuple[int, ...] = (2023, 2024),
    rate: float = 0.005,
    seed: int = 0,
    detectors: dict[str, Callable[[pd.Series, dict], pd.DataFrame]] | None = None,
    tolerance: int = 1,
    context: Callable[[str, int], dict] | None = None,
) -> pd.DataFrame:
    detectors = detectors or default_detectors()
    rows = []
    for ba in bas:
        for year in years:
            s = load_series_local(tidy_dir, ba, year)
            s = s[s > 0]
            if len(s) < 24 * 30:
                continue
            inj = inject_anomalies(s, rate=rate, seed=seed + year)
            raw_bad = raw_fault_mask(inj["clean"].to_numpy())
            ctx = context(ba, year) if context else {}
            for name, fn in detectors.items():
                t0 = time.perf_counter()
                res = fn(inj["value"], ctx)
                dt = time.perf_counter() - t0
                m = score_labels(inj["label"].to_numpy(), res["is_anomaly"].to_numpy(), tolerance=tolerance)
                keep = ~raw_bad | (inj["label"].to_numpy() == 1)
                m_adj = score_labels(inj["label"].to_numpy()[keep], res["is_anomaly"].to_numpy()[keep], tolerance=tolerance)
                m["precision_adj"] = m_adj["precision"]
                m["f1_adj"] = m_adj["f1"]
                m["raw_fault_readings"] = int(raw_bad.sum())
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
    cols = ["precision", "precision_adj", "recall", "f1", "f1_adj", "mcc", "seconds"] + [c for c in results.columns if c.startswith("recall_")]
    agg = results.groupby("detector")[cols].mean().sort_values("f1_adj", ascending=False)
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
