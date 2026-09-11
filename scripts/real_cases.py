"""Real anomalies in the public EIA-930 demand record, and what the detectors do with them.

Two products. (1) A survey of every BA: zero hours, frozen runs, hour-to-hour jumps and unit-error
hours in the raw demand series, written to results/real_anomaly_survey.csv. (2) A gallery of named
cases (docs/real_cases.md with figures in docs/figures/real/), each showing the raw series, the
flags of the v0.1 and v0.2 composites, and the repaired series. The last cases are not data faults
but genuine regime days, where the right answer is no alarm.

Usage: python scripts/real_cases.py --tidy ../ba-forecast-scorecard/data/tidy
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from grid_sentinel import intervals_from_mask, repair, sentinel, sentinel_v2, summarize_intervals  # noqa: E402
from grid_sentinel.benchmark import load_series_local  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIG = DOCS / "figures" / "real"
NAVY, RED, GRAY, TEAL, ORANGE = "#1F3A5F", "#CC2000", "#9CA3AF", "#46B2B4", "#E08A1E"

# (ba, year, window start, window end, title, what it is)
CASES = [
    ("BANC", 2019, "2019-03-27", "2019-04-22", "BANC, April 2019: 204 hours frozen at 1,670 MW, then two weeks of nonsense",
     "A stale value repeated for eight and a half days, followed by readings around 53,000 MW (roughly 30x the normal level) for three days."),
    ("AZPS", 2018, "2018-03-24", "2018-03-29", "AZPS, March 2018: a single hour at 1,676,497 MW",
     "A unit error of a factor of about six hundred on an otherwise normal week."),
    ("FMPP", 2018, "2018-03-20", "2018-03-27", "FMPP, March 2018: a single hour at 50,320 MW",
     "About 28 times the normal level for one hour, in the middle of an ordinary afternoon."),
    ("CISO", 2019, "2019-12-17", "2019-12-23", "CISO, December 2019: California at 14 MW",
     "The largest BA in the West reported a few megawatts around midnight; a near-zero that is not exactly zero, which a zero rule would miss."),
    ("FPL", 2015, "2015-12-07", "2015-12-13", "FPL, December 2015: readings around 1,000 MW",
     "A partial report: about one twelfth of the normal level, most likely one zone of the system instead of the whole."),
    ("SWPP", 2026, "2026-01-03", "2026-01-18", "SWPP, January 2026: 202 hours frozen at 31,226 MW",
     "Southwest Power Pool has 82 frozen runs totalling 2,837 hours in the record; this one lasts more than eight days at the level of a whole RTO."),
    ("LDWP", 2018, "2018-04-15", "2018-04-28", "LDWP, April 2018: 147 hours frozen at 2,866 MW",
     "Los Angeles reported the same megawatt value for six days."),
    ("TVA", 2021, "2021-02-12", "2021-02-22", "TVA, Winter Storm Uri, February 2021: genuine extremes",
     "A real event. The right answer is no alarm on the load, and the v0.1 composite is tested on it."),
    ("CPLE", 2023, "2023-10-28", "2023-11-05", "CPLE, 1-2 November 2023: the first cold morning",
     "A regime transition, not a fault. Demand peaks at 7 a.m. for the first time in the season; a calendar profile alone would raise its loudest alarm of the year here."),
]


def survey(tidy: Path) -> pd.DataFrame:
    files = sorted(tidy.glob("*.parquet"))
    d = pd.concat([pd.read_parquet(f, columns=["ba", "utc_end", "demand"]) for f in files]).sort_values(["ba", "utc_end"])
    rows = []
    for ba, g in d.groupby("ba"):
        x = g["demand"].to_numpy()
        n = len(x)
        valid = np.isfinite(x)
        zero = (x <= 0) & valid
        same = np.zeros(n, bool)
        same[1:] = (x[1:] == x[:-1]) & (x[1:] > 0)
        run = np.zeros(n, int)
        for i in range(1, n):
            run[i] = run[i - 1] + 1 if same[i] else 0
        stuck_ends = np.flatnonzero((run >= 2) & (np.append(run[1:], 0) == 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = x[1:] / x[:-1]
        jump = np.flatnonzero((r > 2) | (r < 0.5)) + 1
        jump = jump[(x[jump] > 0) & (x[jump - 1] > 0)]
        med = pd.Series(x).rolling(24 * 14, min_periods=48, center=True).median().to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            rr = x / med
        unit = ((rr > 5) | (rr < 0.2)) & valid & (x > 0)
        rows.append({"ba": ba, "hours": n, "mean_mw": float(np.nanmean(x[x > 0])) if (x > 0).any() else 0.0,
                     "zero_hours": int(zero.sum()), "frozen_runs": len(stuck_ends), "frozen_hours": int((run >= 2).sum() + len(stuck_ends)),
                     "longest_frozen_run_h": int(run.max() + 1) if n else 0, "hour_to_hour_jumps": len(jump), "unit_error_hours": int(unit.sum())})
    return pd.DataFrame(rows).sort_values("mean_mw", ascending=False)


def plot_case(ba: str, s: pd.Series, v1: pd.DataFrame, v2: pd.DataFrame, fixed: pd.Series, start: str, end: str, title: str, name: str) -> None:
    w = slice(start, end)
    x = s.loc[w]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    ax = axes[0]
    ax.plot(x.index, x / 1e3, color="#212121", lw=0.8, label="as reported")
    f1 = v1.loc[w, "is_anomaly"].to_numpy()
    f2 = v2.loc[w, "is_anomaly"].to_numpy()
    ax.scatter(x.index[f1 & ~f2], x[f1 & ~f2] / 1e3, s=22, facecolors="none", edgecolors=ORANGE, label="flagged by v0.1 only")
    ax.scatter(x.index[f2], x[f2] / 1e3, s=14, color=RED, zorder=3, label="flagged by v0.2")
    exp = v2.loc[w, "expected"]
    ax.plot(exp.index, exp / 1e3, color=TEAL, lw=0.8, ls="--", label="expected (profile)")
    ax.set_title(title, loc="left", fontsize=11, color=NAVY, fontweight="bold")
    ax.set_ylabel("demand, GW")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    hi = float(np.nanpercentile(x, 98))
    if np.nanmax(np.abs(x)) > 4 * hi:
        top = 2.2 * hi / 1e3
        for a in axes:
            a.set_ylim(-0.15 * top, top)
        off = x[(x > top * 1e3) | (x < -0.15 * top * 1e3)]
        for a in axes:
            a.scatter(off.index, np.clip(off / 1e3, -0.15 * top, top * 0.98), marker="^", s=18, color=RED, zorder=4)
        worst = off.abs().idxmax()
        axes[0].annotate(f"{int(off.loc[worst]):,} MW ({len(off)} reading{'s' if len(off) != 1 else ''} off scale)", xy=(worst, top * 0.98), xytext=(0, -14),
                         textcoords="offset points", fontsize=8, color=RED, ha="center")
    ax = axes[1]
    ax.plot(x.index, x / 1e3, color=GRAY, lw=1.2, label="as reported")
    ax.plot(fixed.loc[w].index, fixed.loc[w] / 1e3, color=NAVY, lw=0.9, label="repaired (v0.2, equivalent days)")
    ax.set_ylabel("demand, GW")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    ax.set_xlabel("local time")
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tidy", required=True)
    args = p.parse_args()
    warnings.filterwarnings("ignore")
    tidy = Path(args.tidy)
    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    sv = survey(tidy)
    sv.to_csv(ROOT / "results" / "real_anomaly_survey.csv", index=False)
    big = sv[sv["mean_mw"] >= 500]
    lines = [
        "# Real anomalies in the EIA-930 demand record",
        "",
        "Everything on this page comes from the public hourly demand that U.S. balancing authorities report to EIA, as published. "
        "Nothing was injected. The survey counts four simple signatures in the raw series of every BA; the gallery then shows named cases "
        "with the flags of the two composite detectors and the repaired series. The last two cases are genuine events, where the right answer is no alarm.",
        "",
        "## Survey",
        "",
        f"BAs with at least 500 MW of mean demand ({len(big)} of {len(sv)}), July 2015 to the latest file. A frozen run is three or more identical consecutive hourly values; "
        "a jump is an hour-to-hour ratio above 2 or below 1/2 between positive readings; a unit-error hour is a reading more than 5x or less than 1/5 of the two-week rolling median.",
        "",
        "| BA | mean MW | zero hours | frozen runs | frozen hours | longest frozen run (h) | jumps | unit-error hours |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in big.iterrows():
        lines.append(f"| {r['ba']} | {r['mean_mw']:,.0f} | {r['zero_hours']} | {r['frozen_runs']} | {r['frozen_hours']} | {r['longest_frozen_run_h']} | {r['hour_to_hour_jumps']} | {r['unit_error_hours']} |")
    tot = sv[["zero_hours", "frozen_hours", "hour_to_hour_jumps", "unit_error_hours"]].sum()
    lines += ["", f"All BAs together: {tot['zero_hours']:,} zero hours, {tot['frozen_hours']:,} frozen hours, {tot['hour_to_hour_jumps']:,} jumps, {tot['unit_error_hours']:,} unit-error hours "
              "in a record of about 6.5 million BA-hours. Every one of them reached the public dataset as reported.", "", "## Gallery", ""]

    summary_rows = []
    for ba, year, start, end, title, what in CASES:
        years = sorted({year, int(start[:4]), int(end[:4])})
        s = pd.concat([load_series_local(tidy, ba, y) for y in range(min(years) - 1, max(years) + 1)])
        s = s[~s.index.duplicated()]
        if s.empty:
            continue
        v1 = sentinel(s)
        v2 = sentinel_v2(s)
        fixed, log = repair(s, v2["is_anomaly"].to_numpy(), method="equivalent_days", reason="sentinel v0.2")
        name = f"{ba.lower()}_{start}.png"
        plot_case(ba, s, v1, v2, fixed, start, end, title, name)
        w = slice(start, end)
        n1, n2 = int(v1.loc[w, "is_anomaly"].sum()), int(v2.loc[w, "is_anomaly"].sum())
        iv = intervals_from_mask(v2.loc[w, "is_anomaly"].to_numpy(), index=s.loc[w].index)
        cls = summarize_intervals(iv)
        cls_txt = ", ".join(f"{int(r['intervals'])} {r['duration_class']}" for _, r in cls.iterrows() if r["intervals"] > 0) or "none"
        meth = log.loc[start:end, "method"].value_counts().to_dict() if not log.empty else {}
        summary_rows.append((ba, start, end, n1, n2))
        lines += [f"### {title}", "", what, "",
                  f"Flags in the window: v0.1 composite {n1}, v0.2 composite {n2}. Intervals found by v0.2: {cls_txt}. Repairs: {meth or 'none'}.", "",
                  f"![{title}](figures/real/{name})", ""]
    (DOCS / "real_cases.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(pd.DataFrame(summary_rows, columns=["ba", "start", "end", "flags_v01", "flags_v02"]).to_string(index=False))
    print(f"wrote {DOCS / 'real_cases.md'} and {len(summary_rows)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
