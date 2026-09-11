"""Figures for the README: the detectors acting on a real EIA-930 series with injected faults.

Usage: python scripts/make_figures.py --tidy path/to/ba-forecast-scorecard/data/tidy [--ba PJM --year 2024]
Writes docs/figures/*.png.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grid_sentinel.autoencoder import SparseAutoencoder
from grid_sentinel.baselines import hampel
from grid_sentinel.benchmark import load_series
from grid_sentinel.repair import repair
from grid_sentinel.rules import sentinel, stuck_values
from grid_sentinel.teda import RecursiveTEDA

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
COLORS = {"spike": "#c62828", "dip": "#ef6c00", "zero": "#6a1b9a", "stuck": "#00838f", "scale": "#2e7d32"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "figure.dpi": 150})


def demo_faults(series: pd.Series, start: str) -> pd.DataFrame:
    """Hand-placed faults of every kind inside a two-week window, for illustration.

    The benchmark uses random injection (``inject_anomalies``); this fixed layout exists only so
    that one figure shows every fault type at once.
    """
    clean = series.astype(float).to_numpy()
    values = clean.copy()
    label = np.zeros(len(clean), dtype=np.int8)
    kind = np.array([""] * len(clean), dtype=object)
    i0 = int(series.index.get_indexer([pd.Timestamp(start, tz="UTC")])[0])
    layout = [
        ("spike", 30, 1, 1.45),
        ("dip", 78, 1, 0.55),
        ("zero", 120, 2, 0.0),
        ("stuck", 160, 8, None),
        ("spike", 205, 1, 1.30),
        ("scale", 240, 12, 0.1),
        ("dip", 300, 1, 0.70),
    ]
    for k, off, n, f in layout:
        seg = slice(i0 + off, i0 + off + n)
        values[seg] = clean[i0 + off - 1] if k == "stuck" else clean[seg] * f
        label[seg] = 1
        kind[seg] = k
    return pd.DataFrame({"clean": clean, "value": values, "label": label, "kind": kind}, index=series.index)


def gw(ax):
    ax.set_ylabel("Demand (GW)")
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_detections(inj: pd.DataFrame, res: pd.DataFrame, ba: str, start: str, days: int, name: str, title: str):
    """Two panels: the corrupted series with the injected faults, and the same series with the detector's flags."""
    t0 = pd.Timestamp(start, tz="UTC")
    w = inj.loc[t0 : t0 + pd.Timedelta(days=days)]
    r = res.loc[w.index]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    ax = axes[0]
    ax.plot(w.index, w["clean"] / 1e3, color="#9e9e9e", lw=1, label="clean series")
    ax.plot(w.index, w["value"] / 1e3, color="#212121", lw=0.8, label="series with injected faults")
    for kind, c in COLORS.items():
        sel = w["kind"] == kind
        if sel.any():
            ax.scatter(w.index[sel], w.loc[sel, "value"] / 1e3, s=14, color=c, zorder=3, label=f"injected {kind}")
    ax.set_title(f"{ba}, {days} days from {start}: faults placed on the real series")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    ax.legend(ncol=4, loc="upper left")
    gw(ax)
    ax = axes[1]
    ax.plot(w.index, w["value"] / 1e3, color="#212121", lw=0.8)
    hit = r["is_anomaly"].to_numpy()
    lab = w["label"].to_numpy() == 1
    ax.scatter(w.index[hit & lab], w["value"][hit & lab] / 1e3, s=18, color="#2e7d32", zorder=3, label="flagged, injected")
    ax.scatter(w.index[hit & ~lab], w["value"][hit & ~lab] / 1e3, s=18, color="#1565c0", marker="x", zorder=3,
               label="flagged, not injected")
    ax.scatter(w.index[~hit & lab], w["value"][~hit & lab] / 1e3, s=30, facecolors="none", edgecolors="#c62828",
               zorder=3, label="injected, missed")
    ax.set_title(title)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    ax.legend(ncol=3, loc="upper left")
    gw(ax)
    ax.set_xlabel("UTC")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)


def fig_teda_scores(inj: pd.DataFrame, ba: str, start: str, days: int):
    """The recursive TEDA statistic against its threshold, streaming through the corrupted series."""
    t0 = pd.Timestamp(start, tz="UTC")
    res = RecursiveTEDA(m=4.0, diff=False).detect(inj["value"])
    w = res.loc[t0 : t0 + pd.Timedelta(days=days)]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    axes[0].plot(w.index, w["value"] / 1e3, color="#212121", lw=0.8)
    lab = inj.loc[w.index, "label"].to_numpy() == 1
    axes[0].scatter(w.index[lab], w["value"][lab] / 1e3, s=14, color="#c62828", zorder=3, label="injected faults")
    axes[0].legend(loc="upper left")
    axes[0].set_title(f"{ba}: series seen by the streaming detector")
    gw(axes[0])
    ax = axes[1]
    ax.plot(w.index, w["score"], color="#1565c0", lw=0.8, label="normalised eccentricity / threshold")
    ax.axhline(1.0, color="#c62828", lw=1, ls="--", label="threshold (m = 4)")
    ax.set_yscale("log")
    ax.set_ylabel("score (log)")
    ax.set_title("Recursive TEDA on levels: a reading is flagged when the ratio exceeds 1")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("UTC")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "fig2_teda_scores.png")
    plt.close(fig)


def fig_repair(inj: pd.DataFrame, res: pd.DataFrame, ba: str, start: str, days: int):
    """Repair with the audit trail: corrupted, repaired and clean series on a short window."""
    fixed, log = repair(inj["value"], res["is_anomaly"].to_numpy(), reason="sentinel v0.1")
    t0 = pd.Timestamp(start, tz="UTC")
    w = slice(t0, t0 + pd.Timedelta(days=days))
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(inj.loc[w].index, inj.loc[w, "value"] / 1e3, color="#bdbdbd", lw=1.2, label="as received (with faults)")
    ax.plot(fixed.loc[w].index, fixed.loc[w] / 1e3, color="#1565c0", lw=1.0, label="repaired")
    ax.plot(inj.loc[w].index, inj.loc[w, "clean"] / 1e3, color="#212121", lw=0.6, ls=":", label="clean (truth)")
    lg = log.loc[w]
    ax.scatter(lg.index, lg["repaired"] / 1e3, s=14, color="#1565c0", zorder=3, label=f"{len(lg)} audited repairs")
    ax.set_title(f"{ba}: repair of the flagged readings, each change logged with its reason")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)
    ax.legend(ncol=2, loc="upper left")
    gw(ax)
    ax.set_xlabel("UTC")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "fig3_repair.png")
    plt.close(fig)
    return log


def fig_stuck_and_scale(inj: pd.DataFrame, ba: str):
    """A frozen run and a unit-error run: what the rule and the level detector see, versus a Hampel filter."""
    kinds = inj["kind"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, kind, det, dname in (
        (axes[0], "stuck", stuck_values(inj["value"]), "stuck-value rule"),
        (axes[1], "scale", RecursiveTEDA(m=4.0).detect(inj["value"]), "recursive TEDA (levels)"),
    ):
        pos = np.flatnonzero(kinds == kind)
        if len(pos) == 0:
            continue
        # pick the first run and show 3 days around it
        i0 = pos[0]
        w = slice(max(0, i0 - 30), min(len(inj), i0 + len(pos) + 30))
        idx = inj.index[w]
        ax.plot(idx, inj["value"].iloc[w] / 1e3, color="#212121", lw=0.8, label="series")
        h = hampel(inj["value"])["is_anomaly"].to_numpy()[w]
        d = det["is_anomaly"].to_numpy()[w]
        lab = (kinds == kind)[w]
        ax.scatter(idx[lab], inj["value"].iloc[w][lab] / 1e3, s=40, facecolors="none", edgecolors=COLORS[kind],
                   label=f"injected {kind}")
        ax.scatter(idx[d], inj["value"].iloc[w][d] / 1e3, s=10, color="#2e7d32", zorder=3, label=f"flagged by {dname}")
        ax.scatter(idx[h], inj["value"].iloc[w][h] / 1e3, s=10, color="#ef6c00", marker="x", zorder=4,
                   label="flagged by Hampel (24 h)")
        ax.set_title(f"{ba}: {kind} run")
        ax.legend(loc="best")
        gw(ax)
        ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_stuck_and_scale.png")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tidy", required=True)
    p.add_argument("--ba", default="PJM")
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--start", default="2024-01-08")
    p.add_argument("--days", type=int, default=14)
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    OUT.mkdir(parents=True, exist_ok=True)
    s = load_series(Path(args.tidy), args.ba, args.year)
    inj = demo_faults(s, args.start)
    comp = sentinel(inj["value"])
    ae = SparseAutoencoder(window=4, encoding_dim=2, epochs=20, factor=20.0).detect(inj["value"])
    fig_detections(inj, comp, args.ba, args.start, args.days, "fig1_sentinel.png",
                   "Composite sentinel: TEDA on levels + TEDA on differences + stuck-value rule")
    fig_detections(inj, ae, args.ba, args.start, args.days, "fig1b_autoencoder.png",
                   "Sparse autoencoder (window 4, code 2, threshold 20x trimmed-mean error)")
    fig_teda_scores(inj, args.ba, args.start, args.days)
    log = fig_repair(inj, comp, args.ba, args.start, args.days)
    fig_stuck_and_scale(inj, args.ba)
    print(f"{args.ba} {args.year}: {int(inj['label'].sum())} injected readings, "
          f"{int(comp['is_anomaly'].sum())} flagged by sentinel, {len(log)} repairs logged; figures in {OUT}")
    print(log.head(8).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
