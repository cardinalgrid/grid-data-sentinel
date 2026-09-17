"""Evaluation on the four winter events: what each detector would have flagged in the 72 hours around
each BA's peak, and whether it still finds faults injected into the same windows.

Usage: python scripts/storms.py [--tidy DIR] [--isd DIR] [--neighbors CSV] [--out DIR] [--rate R] [--seed N]
Writes results/storms/{by_ba.csv,summary.csv,summary.json,fig*.pdf,fig*.png} and docs/storms.md.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from grid_sentinel import __version__
from grid_sentinel.benchmark import default_detectors, make_context, raw_fault_mask
from grid_sentinel.data import load_series_local
from grid_sentinel.events import EVENTS, event_window
from grid_sentinel.metrics import score_labels
from grid_sentinel.stations import BA_STATION
from grid_sentinel.synthetic import inject_anomalies

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DETECTORS = (
    "teda_level", "autoencoder", "modified_zscore", "relative_deviation", "sentinel_v2",
    "sentinel_v3_regime_only", "sentinel_v3_profile_neighbors", "sentinel_v3_profile_weather",
    "sentinel_v3_distance_neighbors", "sentinel_v3",
)
SHORT = {
    "teda_level": "TEDA", "autoencoder": "AE", "modified_zscore": "mod. z", "relative_deviation": "rel. dev.",
    "sentinel_v2": "v0.2", "sentinel_v3_regime_only": "v0.3 regime", "sentinel_v3_profile_neighbors": "v0.3 P+N",
    "sentinel_v3_profile_weather": "v0.3 P+W", "sentinel_v3_distance_neighbors": "v0.3 dist.", "sentinel_v3": "v0.3",
}
LIMITATIONS = (
    "One NOAA station per BA; neighbours from the aggregate 2024 interchange record, not per tie line; "
    "a 72-hour window around each BA's peak, chosen here; events are the four with an official under-forecast "
    "figure, not a random sample; injected faults follow the benchmark protocol; nothing here describes any "
    "operator's internal data."
)


def count_window(res: pd.DataFrame, win: pd.DatetimeIndex, peak: pd.Timestamp) -> dict:
    """On the series as reported: how much of the window a detector marks, and whether the peak survives."""
    pos = res.index.get_indexer(win)
    pos = pos[pos >= 0]
    flags = np.asarray(res["is_anomaly"], dtype=bool)[pos]
    return {
        "window_hours": len(pos),
        "flagged": int(flags.sum()),
        "flagged_share": float(flags.sum() / max(len(pos), 1)),
        "peak_flagged": bool(res.loc[peak, "is_anomaly"]) if peak in res.index else None,
    }


def recall_window(res: pd.DataFrame, win: pd.DatetimeIndex, label: np.ndarray, raw_bad: np.ndarray) -> dict:
    """On the series with faults injected inside the window: recall and adjusted precision there."""
    pos = res.index.get_indexer(win)
    pos = pos[pos >= 0]
    flags = np.asarray(res["is_anomaly"], dtype=bool)[pos]
    lab = np.asarray(label)[pos]
    keep = ~np.asarray(raw_bad, dtype=bool)[pos] | (lab == 1)
    if lab.sum():
        m = score_labels(lab[keep], flags[keep], tolerance=1)
    else:
        m = {"recall": np.nan, "precision": np.nan}
    return {
        "injected_in_window": int(lab.sum()),
        "recall_in_window": float(m["recall"]),
        "precision_adj_in_window": float(m["precision"]),
    }


def evaluate_window(res_clean: pd.DataFrame, res_injected: pd.DataFrame, win: pd.DatetimeIndex, peak: pd.Timestamp,
                    label: np.ndarray, raw_bad: np.ndarray) -> dict:
    return {**count_window(res_clean, win, peak), **recall_window(res_injected, win, label, raw_bad)}


def inject_in_window(s: pd.Series, win: pd.DatetimeIndex, rate: float, seed: int) -> pd.DataFrame:
    """The benchmark's injection restricted to the event window; the rest of the year stays clean."""
    inj = pd.DataFrame({"clean": s.to_numpy(), "value": s.to_numpy(), "label": np.zeros(len(s), dtype=np.int8),
                        "kind": np.array([""] * len(s), dtype=object)}, index=s.index)
    in_win = s.index.isin(win)
    sub = inject_anomalies(s[in_win], rate=rate, seed=seed)
    inj.loc[in_win, "value"] = sub["value"].to_numpy()
    inj.loc[in_win, "label"] = sub["label"].to_numpy()
    inj.loc[in_win, "kind"] = sub["kind"].to_numpy()
    return inj


def make_figures(summ: pd.DataFrame, out: Path) -> None:
    plt.style.use(Path(__file__).resolve().parents[1].parent / "notes" / "_template-ieee" / "ieee.mplstyle")
    events = [e["name"] for e in EVENTS]
    labels = {e["name"]: e["label"].replace("Winter Storm", "").replace("s Gerri", " Gerri").strip() for e in EVENTS}
    dets = [d for d in DETECTORS if d in set(summ["detector"])]
    x = np.arange(len(events))
    w = 0.8 / len(dets)
    for fig_name, col, ylabel in (
        ("fig1_flagged_share", "flagged_share", "Share of the 72-hour window flagged as fault"),
        ("fig2_peak_kept", "peak_kept", "Share of BAs whose peak reading is kept"),
    ):
        fig, ax = plt.subplots(figsize=(7.16, 2.8))
        for i, d in enumerate(dets):
            vals = [summ[(summ["event"] == ev) & (summ["detector"] == d)][col].mean() for ev in events]
            ax.bar(x + (i - len(dets) / 2 + 0.5) * w, vals, w, label=SHORT[d])
        ax.set_xticks(x, [labels[e] for e in events])
        ax.set_ylabel(ylabel)
        ax.legend(ncol=5, fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        fig.savefig(out / f"{fig_name}.pdf", bbox_inches="tight")
        fig.savefig(out / f"{fig_name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for d in dets:
        g = summ[summ["detector"] == d]
        ax.scatter(g["flagged_share"], g["recall_in_window"], s=12, label=SHORT[d])
    ax.set_xlabel("Share of window flagged")
    ax.set_ylabel("Recall on injected faults")
    ax.legend(fontsize=5, frameon=False, ncol=2)
    fig.savefig(out / "fig3_recall_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(out / "fig3_recall_tradeoff.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_docs(summ: pd.DataFrame, meta: dict, path: Path) -> None:
    lines = [
        "# Detectors on the four winter events",
        "",
        (
            f"Generated {meta['generated'][:10]} by `scripts/storms.py`, grid-data-sentinel {meta['version']}. "
            "For each event and each BA with a station, the 72 hours around the BA's peak. `flagged_share` and "
            "`peak_kept` come from the series as reported: the share of the window a detector marks as fault and the "
            "share of BAs whose peak reading survives. `recall_in_window` and `precision_adj_in_window` come from a "
            f"second run with faults injected into the same window at a rate of {meta['injection_rate_in_window']:.0%} "
            "(one random draw per BA); the adjusted precision ignores readings the source already had wrong."
        ),
        "",
    ]
    for ev in EVENTS:
        g = summ[summ["event"] == ev["name"]].sort_values("flagged_share")
        if g.empty:
            continue
        lines += [f"## {ev['label']} ({ev['start']} to {ev['end']}, {int(g['bas'].max())} BAs)", "",
                  "| Detector | flagged_share | peak_kept | recall_in_window | precision_adj_in_window |", "|---|---|---|---|---|"]
        for r in g.itertuples():
            lines.append(f"| {SHORT[r.detector]} | {r.flagged_share:.3f} | {r.peak_kept:.2f} | {r.recall_in_window:.2f} | {r.precision_adj_in_window:.2f} |")
        lines += ["", "BAs: " + ", ".join(meta["events"][ev["name"]]["bas"]), ""]
    lines += ["## Limitations", "", LIMITATIONS, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tidy", type=Path, default=Path("../ba-forecast-scorecard/data/tidy"))
    ap.add_argument("--isd", type=Path, default=Path("../notes/03-daily-load-profiles/data/isd"))
    ap.add_argument("--neighbors", type=Path, default=Path("docs/neighbors.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/storms"))
    ap.add_argument("--rate", type=float, default=0.02, help="injection rate inside the event windows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bas", nargs="*", default=None, help="subset of BAs (default: all with a station)")
    ap.add_argument("--events", nargs="*", default=None, help="subset of events (default: all four)")
    ap.add_argument("--resume", action="store_true", help="skip (event, BA) pairs already in by_ba.csv")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore")
    ctx_fn = make_context(a.tidy, a.isd, a.neighbors)
    dets = {k: v for k, v in default_detectors().items() if k in DETECTORS}
    by_ba = a.out / "by_ba.csv"
    done: set[tuple[str, str]] = set()
    if a.resume and by_ba.exists():
        prev = pd.read_csv(by_ba)
        done = set(zip(prev["event"], prev["ba"]))
    elif by_ba.exists():
        by_ba.unlink()
    for ev in EVENTS:
        if a.events and ev["name"] not in a.events:
            continue
        year = int(ev["start"][:4])
        for ba in (a.bas or sorted(BA_STATION)):
            if (ev["name"], ba) in done:
                continue
            s = load_series_local(a.tidy, ba, year)
            s = s[s > 0]
            if len(s) < 24 * 60:
                continue
            try:
                peak, win = event_window(s, ev["start"], ev["end"])
            except ValueError:
                continue
            inj = inject_in_window(s, win, a.rate, a.seed + year + sum(map(ord, ba)))  # one draw per BA
            raw_bad = raw_fault_mask(inj["clean"].to_numpy())
            ctx = ctx_fn(ba, year)
            rows = []
            for name, fn in dets.items():
                res_clean = fn(s, ctx)
                res_injected = fn(inj["value"], ctx)
                rows.append({"event": ev["name"], "ba": ba, "detector": name, "peak_time": peak.isoformat(),
                             **evaluate_window(res_clean, res_injected, win, peak, inj["label"].to_numpy(), raw_bad)})
            pd.DataFrame(rows).to_csv(by_ba, mode="a", header=not by_ba.exists(), index=False)
            del ctx, rows
            print(f"{ev['name']} {ba} peak {peak} done", flush=True)
    df = pd.read_csv(by_ba)
    df["peak_kept"] = 1.0 - df["peak_flagged"].astype(float)
    summ = df.groupby(["event", "detector"]).agg(
        flagged_share=("flagged_share", "mean"), peak_kept=("peak_kept", "mean"),
        recall_in_window=("recall_in_window", "mean"), precision_adj_in_window=("precision_adj_in_window", "mean"),
        bas=("ba", "nunique"),
    ).reset_index()
    summ.to_csv(a.out / "summary.csv", index=False)
    meta = {
        "version": __version__, "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "injection_rate_in_window": a.rate, "seed": a.seed,
        "events": {ev["name"]: {"label": ev["label"], "start": ev["start"], "end": ev["end"],
                                "bas": sorted(df[df["event"] == ev["name"]]["ba"].unique())} for ev in EVENTS},
        "results": {f"{r.event}/{r.detector}": {c: (None if pd.isna(v) else float(v)) for c, v in r._asdict().items()
                                                 if c not in ("event", "detector", "Index")} for r in summ.itertuples()},
    }
    (a.out / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    make_figures(summ, a.out)
    write_docs(summ, meta, Path("docs/storms.md"))
    print(summ.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
