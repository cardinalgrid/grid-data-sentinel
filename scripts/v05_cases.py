"""What the v0.5 composite changes on the public record: named windows where v0.4 and v0.5 disagree.

    python scripts/v05_cases.py --tidy <tidy> --isd <isd>

Writes docs/real_cases_v05.md and its figures in docs/figures/real/. Each window shows the series as published,
the expectation of the profile, and the readings flagged by the v0.3/v0.4 composite (``Sentinel.v04()``) and by
the v0.5 composite (``Sentinel()``), both run on the whole year with temperature and neighbours.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid_sentinel.benchmark import context_from_dict, load_series_local, make_context
from grid_sentinel.detectors import Sentinel
from grid_sentinel.pandas import from_series, regularize

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FIG = DOCS / "figures" / "real"
NAVY, RED, TEAL, ORANGE = "#1F3A5F", "#CC2000", "#46B2B4", "#E08A1E"

# (ba, year, window start, window end, title, what it is)
CASES = [
    ("CISO", 2021, "2021-06-14", "2021-06-21", "CISO, June 2021: a heat wave that v0.4 flagged",
     ("The western heat wave of mid-June 2021. Demand runs 30% to 60% above what the calendar profile expects for "
     "days, and neither the temperature check (one airport, its own seasonal tail) nor the neighbour check confirms "
     "every hour. The v0.4 composite condemns the afternoons; v0.5 keeps them because each hour builds on an "
     "hour that was already high.")),
    ("NYIS", 2021, "2021-06-27", "2021-07-01", "NYIS, late June 2021: the same in New York",
     "Four hot days in a row; the readings v0.4 flagged are the peaks of the hottest afternoons."),
    ("SRP", 2023, "2023-05-12", "2023-05-19", "SRP, May 2023: the 23:00 reading returns to the afternoon level",
     ("From May to early July 2023 the 23:00 reading of Salt River Project goes back up to about the level of "
     "17:00 to 18:00 and falls again at midnight, almost every day. This is not demand. The v0.4 composite let "
     "it pass because the repeated jump had widened its own spread of hourly changes; v0.5 keeps condemned "
     "readings out of that spread.")),
]


def run_year(tidy: Path, ctx_of, ba: str, year: int):
    s = load_series_local(tidy, ba, year)
    s = s[s > 0]
    reg = regularize(s)
    v, t = from_series(reg)
    ctx = context_from_dict(ctx_of(ba, year))
    old = Sentinel.v04().predict(v, t, ctx)
    new = Sentinel().predict(v, t, ctx)
    idx = pd.DatetimeIndex(t)
    return (pd.Series(v, idx), pd.Series(old.is_anomaly, idx), pd.Series(new.is_anomaly, idx),
            pd.Series(new.expected, idx), pd.Series(np.asarray(new.reason), idx))


def plot(x, f_old, f_new, exp, title, name):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(x.index, x / 1e3, color="#212121", lw=0.8, label="as reported")
    ax.plot(exp.index, exp / 1e3, color=TEAL, lw=0.8, ls="--", label="expected (profile)")
    only_old = f_old & ~f_new
    only_new = f_new & ~f_old
    both = f_old & f_new
    ax.scatter(x.index[only_old], x[only_old] / 1e3, s=26, facecolors="none", edgecolors=ORANGE, label="flagged by v0.4 only")
    ax.scatter(x.index[only_new], x[only_new] / 1e3, s=16, color=RED, zorder=3, label="flagged by v0.5 only")
    if both.any():
        ax.scatter(x.index[both], x[both] / 1e3, s=16, color=NAVY, zorder=3, label="flagged by both")
    ax.set_title(title, loc="left", fontsize=11, color=NAVY, fontweight="bold")
    ax.set_ylabel("demand, GW")
    ax.set_xlabel("local time")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tidy", required=True)
    ap.add_argument("--isd", required=True)
    ap.add_argument("--neighbors", default="docs/neighbors.csv")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    FIG.mkdir(parents=True, exist_ok=True)
    tidy = Path(a.tidy)
    ctx_of = make_context(tidy, Path(a.isd), Path(a.neighbors))
    lines = [
        "# What v0.5 changes on the public record",
        "",
        ("The same public EIA-930 demand, as published, run through the v0.3/v0.4 composite (`Sentinel.v04()`) and the v0.5 "
        "composite (`Sentinel()`), each on the whole year with the BA's temperature and neighbours. Over the ten benchmark "
        "BAs from 2021 to 2024, v0.4 flags 856 readings and v0.5 flags 627, none of them new: the readings that leave are "
        "morning ramps flagged in the first weeks of each year, before the profile has history, and genuine heat-wave hours. "
        "Over every BA with a weather station in 2023 and 2024, v0.4 flags 2,300 and v0.5 1,680; 86 readings are flagged "
        "only by v0.5, and on inspection they are faults of the published series, such as the third case below. "
        "Produced by `scripts/v05_cases.py`; the counts by `scripts/real_diff.py`."),
        "",
    ]
    for ba, year, start, end, title, what in CASES:
        x, f_old, f_new, exp, _ = run_year(tidy, ctx_of, ba, year)
        w = slice(start, end)
        name = f"v05_{ba.lower()}_{start}.png"
        plot(x.loc[w], f_old.loc[w], f_new.loc[w], exp.loc[w], title, name)
        n_old, n_new = int(f_old.loc[w].sum()), int(f_new.loc[w].sum())
        lines += [f"## {title}", "", what, "", f"Flags in the window: v0.4 composite {n_old}, v0.5 composite {n_new}.", "",
                  f"![{title}](figures/real/{name})", ""]
        print(ba, start, "v0.4", n_old, "v0.5", n_new, flush=True)
    (DOCS / "real_cases_v05.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("wrote docs/real_cases_v05.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
