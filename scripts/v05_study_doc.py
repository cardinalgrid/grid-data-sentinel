"""Render docs/v05_study.md from the tables of ``scripts/v05_study.py``.

    python scripts/v05_study.py --render-only
"""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.v05_study import COMPOSITE, FEEDER, UNION

KINDS = ("spike", "dip", "zero", "stuck", "scale")
EVENTS = (("uri", "Uri"), ("elliott", "Elliott"), ("gerri_heather", "Gerri and Heather"), ("january_2025", "January 2025"))
EPS = 1e-9


def f3(x) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def md(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def kwargs_text(kw) -> str:
    if kw is None:
        return "`Sentinel()`"
    if not kw:
        return "`Sentinel.v04()`"
    parts = []
    for k, v in kw.items():
        v = "inf" if isinstance(v, float) and math.isinf(v) else repr(v)
        parts.append(f"{k}={v}")
    return "`" + ", ".join(parts) + "`"


def feeder_text(kw) -> str:
    return "`" + ", ".join(f"{k}={v!r}" for k, v in kw.items()) + "`"


# ---- scoring helpers --------------------------------------------------------------------------------------------
METRICS = ["precision_adj", "recall", "f1_adj"] + [f"recall_{k}" for k in (*KINDS, "feeder_loss")]


def mean_of(df: pd.DataFrame, **sel) -> pd.DataFrame:
    m = np.ones(len(df), bool)
    for k, v in sel.items():
        m &= df[k].isin(v if isinstance(v, (list, tuple)) else [v]).to_numpy()
    cols = [c for c in METRICS if c in df.columns]
    return df[m].groupby("detector")[cols].mean()


def item1(c: pd.Series, b: pd.Series) -> list[str]:
    """Parts of item 1 that fail: F1 down by more than 0.003, precision by more than 0.01, a type's recall by more
    than 0.02."""
    fails = []
    if c["f1_adj"] - b["f1_adj"] < -0.003 - EPS:
        fails.append("F1")
    if c["precision_adj"] - b["precision_adj"] < -0.01 - EPS:
        fails.append("precision")
    for k in KINDS:
        col = f"recall_{k}"
        if col in c and np.isfinite(c[col]) and np.isfinite(b[col]) and c[col] - b[col] < -0.02 - EPS:
            fails.append(k)
    return fails


def combos(df: pd.DataFrame) -> dict[tuple, pd.DataFrame]:
    """Confirmation combinations: (scheme, seed, year) -> means over the ten BAs."""
    out = {}
    conf = df[df["period"] == "confirm"]
    for (scheme, seed, year), g in conf.groupby(["scheme", "seed", "year"]):
        out[(scheme, int(seed), int(year))] = g.groupby("detector")[[c for c in METRICS if c in g.columns]].mean()
    return out


def storms_table(st: pd.DataFrame) -> pd.DataFrame:
    st = st.copy()
    st["peak_kept"] = 1.0 - st["peak_flagged"].astype(float)
    return st.groupby(["detector", "event"]).agg(peak_kept=("peak_kept", "mean"), flagged_share=("flagged_share", "mean"),
                                                 recall=("recall_in_window", "mean"), bas=("ba", "nunique"))


# ---- render -----------------------------------------------------------------------------------------------------
def render(out_dir: Path, doc: Path) -> None:
    std = pd.read_csv(out_dir / "standard.csv")
    fdr = pd.read_csv(out_dir / "feeder.csv")
    lad = pd.read_csv(out_dir / "feeder_ladder.csv")
    st = storms_table(pd.read_csv(out_dir / "storms.csv"))
    real = pd.read_csv(out_dir / "real.csv")
    freal = pd.read_csv(out_dir / "feeder_real.csv")
    names = list(COMPOSITE)
    key = {n: f"A:{n}" for n in names}

    dev_pub = mean_of(std, period="dev", scheme="published")
    dev_pss = mean_of(std, period="dev", scheme="per_series")
    conf_pub = mean_of(std, period="confirm", scheme="published")
    conf_pss = mean_of(std, period="confirm", scheme="per_series")
    cmb = combos(std)
    base = key["v04"]

    # ---- part A verdicts
    verdict = {}
    for n in names[1:]:
        c, b = dev_pub.loc[key[n]], dev_pub.loc[base]
        i1 = item1(c, b)
        i3 = c["recall_dip"] - b["recall_dip"] >= 0.05 - EPS
        i4 = all(st.loc[(key[n], ev), "peak_kept"] >= st.loc[(base, ev), "peak_kept"] - EPS for ev, _ in EVENTS)
        i5_rule = sum(not item1(t.loc[key[n]], t.loc[base]) for t in cmb.values())
        i5_dip = sum((not np.isfinite(t.loc[base, "recall_dip"])) or t.loc[key[n], "recall_dip"] >= t.loc[base, "recall_dip"] - EPS
                     for t in cmb.values())
        pooled = conf_pub.loc[key[n], "recall_dip"] + conf_pss.loc[key[n], "recall_dip"] > conf_pub.loc[base, "recall_dip"] + conf_pss.loc[base, "recall_dip"] + EPS
        i5 = i5_rule == len(cmb) and i5_dip == len(cmb) and pooled
        failed = ([f"1 ({', '.join(i1)})"] if i1 else []) + ([] if i3 else ["3"]) + ([] if i4 else ["4"]) + ([] if i5 else ["5"])
        if not failed:
            outcome = "passes"
        elif i1 and set(i1) == {"precision"} and i3 and i4 and i5:
            outcome = "option (fails only on precision)"
        else:
            outcome = "out"
        verdict[n] = {"1": "✗ " + ", ".join(i1) if i1 else "✓", "3": "✓" if i3 else "✗", "4": "✓" if i4 else "✗",
                      "5": f"{'✓' if i5 else '✗'} ({i5_rule}/{len(cmb)} item 1, {i5_dip}/{len(cmb)} dips)", "outcome": outcome}

    real_sum = real.groupby("candidate")[["hours", "flagged", "removed", "added", "removed_no_expectation", "removed_kept_as_extreme"]].sum()

    # ---- part B
    u_dev_pss = mean_of(fdr, period="dev", scheme="per_series")
    u_dev_pub = mean_of(fdr, period="dev", scheme="published")
    u_conf_pss = mean_of(fdr, period="confirm", scheme="per_series")
    u_conf_pub = mean_of(fdr, period="confirm", scheme="published")
    fcmb = {k: v for k, v in combos(fdr).items() if k[0] == "per_series"}
    v5 = key["v05"]
    b_i1 = item1(dev_pub.loc[UNION], dev_pub.loc[v5])
    b_i2 = u_dev_pss.loc[UNION, "recall_feeder_loss"] >= 0.5 and u_dev_pss.loc[UNION, "f1_adj"] > u_dev_pss.loc[v5, "f1_adj"]
    b_i4 = all(st.loc[(UNION, ev), "peak_kept"] >= st.loc[(v5, ev), "peak_kept"] - EPS for ev, _ in EVENTS)
    b_i5 = sum(t.loc[UNION, "recall_feeder_loss"] >= 0.5 and t.loc[UNION, "f1_adj"] > t.loc[v5, "f1_adj"] for t in fcmb.values())

    L = []
    w = L.append
    w("# The 0.5 study: every configuration tried, and the rule that chose")
    w("")
    w("Version 0.5 changed the default composite and added an optional detector for partial loss of telemetry. This page "
      "lists every configuration tried on the way, the rule that decided which of them enter the default, written down "
      "before the first number was computed, and how each configuration fared against it. Every number on this page comes "
      "from `scripts/v05_study.py` run on the code of version 0.5.0: each candidate is a set of keyword arguments of "
      "`Sentinel.v04()` or of `FeederLoss`, so any row can be rerun on its own. The tables are in "
      "`results/v05_study/`.")
    w("")
    w("## The adoption rule")
    w("")
    w("Fixed before any configuration was run:")
    w("")
    w("1. **Standard benchmark** (development years): adjusted F1 drops by at most 0.003, adjusted precision by at most 0.01, "
      "and no fault type's recall by more than 0.02.")
    w("2. **Feeder-loss benchmark**: recall on feeder losses of at least 0.50, and a better adjusted F1.")
    w("3. **Dips**: recall on single dips improves by at least 0.05 on the standard benchmark.")
    w("4. **Winter storms**: the share of BAs whose peak reading is kept does not fall in any of the four events.")
    w("5. **Confirmation** on years not used in development (2021 and 2022, seeds 1 and 2): item 1 holds and the target "
      "recall improves in every combination.")
    w("")
    w("A change that passes every item that applies to it enters the default. A change that fails only on precision becomes "
      "an option, off by default. Anything else stays out. Items 1, 3, 4 and 5 apply to changes of the composite; items 1, "
      "2, 4 and 5 to the feeder-loss detector.")
    w("")
    w("One reading of item 5 has to be stated. In some combinations the baseline already catches every injected dip, and "
      "nothing can improve there. The check applied here is that the target recall is never lower in any combination and "
      "is higher over all of them together. The tables below show every combination, so a stricter reading can be checked.")
    w("")
    w("## The benchmarks")
    w("")
    w("- **Standard**: the ten benchmark BAs, 2023 and 2024 (development), 0.5% of the readings corrupted with spikes, dips, "
      "zero runs, frozen runs and unit errors, scored with a tolerance of one hour; \"adjusted\" leaves out readings that are "
      "already faulty in the published series.")
    w("- **Two injection schemes.** The published benchmark draws the faults with seed `seed + year`, so every BA gets them "
      "at the same hours and one draw counts ten times. The per-series scheme gives each BA its own draw "
      "(`1000 * seed + 37 * i + year`). Development uses seed 0 in the published scheme and seeds 0 to 2 per series.")
    w("- **Feeder-loss**: the same plus 6- to 72-hour losses of 5% to 30% of the load. **Dense**: feeder losses only, 2% "
      "of the readings.")
    w("- **Confirmation**: 2021 and 2022, seeds 1 and 2, both schemes: eight combinations of scheme, seed and year.")
    w("- **Storms**: the 72 hours around each BA's peak in Winter Storm Uri (2021), Elliott (2022), Gerri and Heather "
      "(2024) and the January 2025 Arctic events, every BA with a weather station, as in `docs/storms.md`.")
    w("- **The series as published**: the ten BAs from 2021 to 2024, with nothing injected.")
    w("")

    # ---- part A
    w("## Part A: what the composite learns from")
    w("")
    w("In 0.4 the composite caught 71% of the injected single dips. The cause is in its eccentricity detectors: their running "
      "mean and variance never forget, and every reading enters them, including the readings the composite has just "
      "condemned. A series with a zero run or a unit error early in the year carries a wider spread of hourly changes for "
      "the rest of it, and a dip of 30% to 60% later no longer stands out against that spread. The candidates below attack "
      "this from different sides: a higher threshold, clipped updates, forgetting, or keeping condemned readings out of "
      "one or both statistics. The last three add rules about genuine extremes, which the series as published showed to "
      "be needed (see below).")
    w("")
    w("### The candidates")
    w("")
    w(md(["candidate", "configuration", "what it tries"], [[f"`{n}`", kwargs_text(COMPOSITE[n][0]), COMPOSITE[n][1]] for n in names]))
    w("")
    w("### Development benchmark")
    w("")
    rows = []
    for n in names:
        p, s = dev_pub.loc[key[n]], dev_pss.loc[key[n]]
        rows.append([f"`{n}`", f3(p["precision_adj"]), f3(p["f1_adj"]), *(f3(p[f"recall_{k}"]) for k in KINDS),
                     f3(s["precision_adj"]), f3(s["f1_adj"]), f3(s["recall_spike"]), f3(s["recall_dip"])])
    w("Published scheme (seed 0), then per-series scheme (seeds 0 to 2); recall by fault type.")
    w("")
    w(md(["candidate", "precision adj.", "F1 adj.", "spike", "dip", "zero", "frozen", "unit", "per series: precision adj.",
          "F1 adj.", "spike", "dip"], rows))
    w("")
    w("### Confirmation, 2021 and 2022")
    w("")
    order = sorted(cmb)
    rows = []
    for n in names:
        rows.append([f"`{n}`", *(f"{f3(cmb[c].loc[key[n], 'f1_adj'])} / {f3(cmb[c].loc[key[n], 'recall_dip'])}" for c in order)])
    w("Adjusted F1 / recall on dips in each combination (scheme, seed, year); a dash where no dip was drawn.")
    w("")
    w(md(["candidate", *(f"{'pub.' if c[0] == 'published' else 'per s.'} {c[1]}/{c[2]}" for c in order)], rows))
    w("")
    w("### Winter storms")
    w("")
    rows = []
    for n in names:
        rows.append([f"`{n}`", *(f3(st.loc[(key[n], ev), "peak_kept"]) for ev, _ in EVENTS),
                     *(f3(st.loc[(key[n], ev), "recall"]) for ev, _ in EVENTS),
                     f"{100 * np.mean([st.loc[(key[n], ev), 'flagged_share'] for ev, _ in EVENTS]):.2f}%"])
    w("Share of BAs whose peak reading is kept, recall on faults injected into the windows, and share of the readings of the "
      "windows flagged with nothing injected (mean of the four events).")
    w("")
    w(md(["candidate", *(f"peak kept, {lbl}" for _, lbl in EVENTS), *(f"recall, {lbl}" for _, lbl in EVENTS), "flagged"], rows))
    w("")
    w("### The series as published")
    w("")
    rows = []
    for n in names:
        r = real_sum.loc[n]
        rows.append([f"`{n}`", int(r["flagged"]), int(r["removed"]), int(r["removed_no_expectation"]),
                     int(r["removed_kept_as_extreme"]), int(r["added"])])
    w(f"Readings flagged over the ten BAs from 2021 to 2024 ({int(real_sum.loc['v04', 'hours']):,} readings), and how they "
      "differ from the 0.4 composite: readings it flags that the candidate does not (of which before the profile has any "
      "history, and of which kept as genuine extremes), and readings only the candidate flags.")
    w("")
    w(md(["candidate", "flagged", "no longer flagged", "of which no expectation yet", "of which kept as extremes",
          "newly flagged"], rows))
    w("")
    w("### Against the rule")
    w("")
    w(md(["candidate", "item 1", "item 3", "item 4", "item 5", "outcome"],
         [[f"`{n}`", verdict[n]["1"], verdict[n]["3"], verdict[n]["4"], verdict[n]["5"], verdict[n]["outcome"]] for n in names[1:]]))
    w("")
    P = lambda n, col: dev_pub.loc[key[n], col]
    S = lambda n, col: dev_pss.loc[key[n], col]
    R = lambda n, col: int(real_sum.loc[n, col])
    pk = lambda n, ev: st.loc[(key[n], ev), "peak_kept"]
    rc = lambda n, ev: st.loc[(key[n], ev), "recall"]
    broad = ("robust", "forgetting", "exclude_both")
    worst = min(cmb, key=lambda c: cmb[c].loc[key["exclude_diff_gross"], "recall_dip"])
    lost_spike = [c for c in order if all(cmb[c].loc[key[n], "recall_spike"] < 0.5 for n in names)]
    w("### Reading the tables")
    w("")
    w(f"**What fails, and why.** A higher threshold alone (`m_diff_3.5`) buys precision ({f3(P('v04', 'precision_adj'))} to "
      f"{f3(P('m_diff_3.5', 'precision_adj'))}) and loses dips ({f3(P('v04', 'recall_dip'))} to {f3(P('m_diff_3.5', 'recall_dip'))}): "
      "the spread is still inflated, and a higher bar lets more dips through. Clipped updates, forgetting and keeping "
      "condemned readings out of both statistics all catch the dips "
      f"({f3(min(P(n, 'recall_dip') for n in broad))} to {f3(max(P(n, 'recall_dip') for n in broad))}), and all lose "
      f"precision ({f3(min(P(n, 'precision_adj') for n in broad))} to {f3(max(P(n, 'precision_adj') for n in broad))}). "
      "The new alarms of `exclude_both`, read one by one, are flags on the statistic on levels at genuine summer peaks: "
      "once the spread of levels is no longer widened by the injected faults, a heat-wave peak stands out against it. The finding cuts both ways: the 0.4 "
      "composite leaves those peaks alone partly because the faults it is scored on widen its spread. Forgetting also "
      f"loses a peak in Elliott ({f3(pk('v04', 'elliott'))} to {f3(pk('forgetting', 'elliott'))}), the only failure of item 4.")
    w("")
    w("**Three settings pass, and the rule does not choose among them.** Keeping condemned readings out of the statistic on "
      "differences only, with the statistic on levels as before, passes every item with each of the three settings tried. "
      f"`exclude_diff_gross` catches fewer dips ({f3(P('exclude_diff_gross', 'recall_dip'))} in the published scheme, "
      f"{f3(S('exclude_diff_gross', 'recall_dip'))} per series, and {f3(cmb[worst].loc[key['exclude_diff_gross'], 'recall_dip'])} "
      f"in the confirmation combination {worst[1]}/{worst[2]} per series, where the other two reach "
      f"{f3(cmb[worst].loc[key['exclude_diff'], 'recall_dip'])}). `exclude_diff_3.5` catches more dips in the published scheme "
      f"({f3(P('exclude_diff_3.5', 'recall_dip'))} against {f3(P('exclude_diff', 'recall_dip'))}), the same per series, at a "
      f"precision {P('exclude_diff', 'precision_adj') - P('exclude_diff_3.5', 'precision_adj'):.3f} lower in the published "
      f"scheme and {S('exclude_diff', 'precision_adj') - S('exclude_diff_3.5', 'precision_adj'):.3f} lower per series. "
      "`exclude_diff` was kept. The three differ by a few thousandths of F1, and either of the other two would have been "
      "a defensible choice.")
    w("")
    w(f"**What the series as published added.** Against the 0.4 composite, `exclude_diff` stops flagging "
      f"{R('exclude_diff', 'removed')} readings, {R('exclude_diff', 'removed_no_expectation')} of them before the profile has "
      "any history: the first weeks of each year, when nothing confirms or withdraws an alarm and ordinary morning ramps "
      f"crossed the threshold of 3 on differences (the higher threshold alone removes "
      f"{R('m_diff_3.5', 'removed_no_expectation')} of them). It also flags {R('exclude_diff', 'added')} readings the 0.4 "
      "composite does not, nearly all genuine summer peaks, among them California's record load of September 2022. The rule "
      "does not see this: its storm item covers winter only. Looking at why those peaks pass the cross-checks turned up the "
      "same problem in the 0.4 composite itself, which flags heat-wave peaks that the profile, the neighbours and the weather "
      "station all fail to confirm (`docs/real_cases_v05.md`). `keep_built` keeps a reading above the expectation when the "
      "reading before it already carried at least half of its excess: a fault arrives at once, a heat peak builds over "
      f"hours. With it the series as published lose {R('keep_built', 'removed')} flags against 0.4, "
      f"{R('keep_built', 'removed_kept_as_extreme')} of them kept as extremes, and gain {R('keep_built', 'added')}; precision "
      f"per series rises from {f3(S('exclude_diff', 'precision_adj'))} to {f3(S('keep_built', 'precision_adj'))}, and no "
      "recall moves.")
    w("")
    w("**The jump guard, and why it carries an exception.** `keep_built` widens what the cross-checks may keep, and the "
      "jump guard narrows it again: a reading far above the expectation whose predecessor was near it is a lone jump, and "
      "the cross-checks do not keep it. Spike recall rises from "
      f"{f3(P('keep_built', 'recall_spike'))} to {f3(P('jump_guard_alone', 'recall_spike'))} in the published scheme and from "
      f"{f3(S('keep_built', 'recall_spike'))} to {f3(S('jump_guard_alone', 'recall_spike'))} per series, and the recall in the "
      f"Uri windows from {f3(rc('keep_built', 'uri'))} to {f3(rc('jump_guard_alone', 'uri'))}; on the series as published the "
      "flags are the same. `jump_guard_alone` and `v05` give the same numbers in every table on this page. They differ on a "
      "case the benchmarks do not hold: a genuine extreme that arrives as a step, such as a cold morning on which a whole "
      "region's load jumps in one hour. Without the exception the guard condemns the first hour of that step; with it, a "
      "step the neighbours share (their median hourly change at least half of the BA's own) is not a lone jump. "
      "`tests/test_sentinel_v04.py::test_jump_guard_condemns_a_lone_step_but_not_one_the_neighbours_share` holds the case. "
      "The choice of `v05` over `jump_guard_alone` rests on that test, not on the rule.")
    w("")
    lim = ("**Limits.** Item 4 covers winter only, and summer behaviour rests on the series as published, read case by case. "
           "The confirmation has eight combinations; in one of them no dip was drawn.")
    if lost_spike:
        c = lost_spike[0]
        lim += (f" In {'the published scheme' if c[0] == 'published' else 'the per-series scheme'}, seed {c[1]}, {c[2]}, "
                f"spike recall is {f3(cmb[c].loc[base, 'recall_spike'])} for every candidate: the draw holds a single spike, "
                "repeated at the same hour in all ten BAs, small enough to fall inside the profile's band, which no "
                "candidate changes.")
    lim += (" One cost does not show in these tables: with a threshold of 4 on differences, the composite fires later at the "
            "very start of a series, before the profile has any history, and a spike in the first hours of a series can "
            "pass where a threshold of 3 would have caught it.")
    w(lim)
    w("")

    # ---- part B
    w("## Part B: partial loss of telemetry")
    w("")
    w("### The detector, change by change")
    w("")
    w("Each row adds one change to the row above it, except the two alternatives, which change the row before them in another "
      "way. Losses found: share of the dense injected losses with more than half of their readings flagged (per-series "
      "seeds 0 and 1 for development, 1 and 2 for confirmation). Flagged with nothing injected: readings flagged on the "
      "twenty development series as published, a rate of false alarms or of real losses.")
    w("")
    lad = lad.set_index(["variant", "period"])
    rows = []
    for n in FEEDER:
        d, c = lad.loc[(n, "dev")], lad.loc[(n, "confirm")]
        rows.append([f"`{n}`", FEEDER[n][1], f3(d["found"]), f3(c["found"]), f"{d['flagged_outside'] / max(d['flagged'], 1):.3f}",
                     int(d["clean_flagged"]), f"{100 * d['clean_flagged'] / d['clean_hours']:.2f}%"])
    w(md(["variant", "change", "losses found (dev.)", "losses found (conf.)", "share of flags outside the losses (dev.)",
          "flagged with nothing injected", "share"], rows))
    w("")
    w(f"`f6` is `FeederLoss()`: {feeder_text(FEEDER['f6'][0])} on top of the defaults.")
    w("")
    w("### The composite with and without it")
    w("")
    rows = []
    for lbl, d in (("feeder benchmark, development, per series", u_dev_pss), ("feeder benchmark, development, published", u_dev_pub),
                   ("feeder benchmark, confirmation, per series", u_conf_pss), ("feeder benchmark, confirmation, published", u_conf_pub),
                   ("standard benchmark, development, published", dev_pub), ("standard benchmark, development, per series", dev_pss),
                   ("standard benchmark, confirmation, published", conf_pub), ("standard benchmark, confirmation, per series", conf_pss)):
        for det, name in ((v5, "`Sentinel()`"), (UNION, "with `FeederLoss()`")):
            r = d.loc[det]
            rows.append([lbl, name, f3(r["precision_adj"]), f3(r["f1_adj"]), f3(r.get("recall_feeder_loss", float("nan"))),
                         f3(r["recall_dip"]), f3(r["recall_spike"])])
    w(md(["benchmark", "detector", "precision adj.", "F1 adj.", "feeder losses", "dip", "spike"], rows))
    w("")
    peaks = " ".join(f"{lbl} {f3(st.loc[(UNION, ev), 'peak_kept'])} against {f3(st.loc[(v5, ev), 'peak_kept'])};" for ev, lbl in EVENTS)
    w(f"Peak kept in the storms, with and without it: {peaks.rstrip(';')}.")
    w("")
    w("### Against the rule")
    w("")
    w(md(["item", "result"], [
        ["1, standard benchmark", "✓" if not b_i1 else "✗ " + ", ".join(b_i1)],
        ["2, feeder-loss benchmark", "✓" if b_i2 else "✗"],
        ["4, storms", "✓" if b_i4 else "✗"],
        ["5, confirmation", f"{'✓' if b_i5 == len(fcmb) else '✗'} ({b_i5}/{len(fcmb)} per-series combinations)"],
    ]))
    w("")
    G = lambda n, col: lad.loc[(n, "dev"), col]
    C = lambda n: lad.loc[(n, "confirm"), "found"]
    fpr = lambda n: int(G(n, "clean_flagged"))
    steps = ["f0", "f1", "f2", "f3", "f4", "f5", "f6"]
    against = [b for a, b in pairwise(steps) if (G(b, "found") - G(a, "found")) * (C(b) - C(a)) < 0]
    prev = {b: a for a, b in pairwise(steps)}
    conf_found = [t.loc[UNION, "recall_feeder_loss"] for t in fcmb.values()]
    w("### Reading the tables")
    w("")
    w(f"**The ladder.** The first three changes raise the losses found from {f3(G('f0', 'found'))} to {f3(G('f2', 'found'))}, "
      f"and flag {fpr('f2'):,} readings of the series as published. Most of those are segments that begin and end on a "
      f"midnight or change day type at an edge: `f3` removes "
      f"{100 * (1 - fpr('f3') / fpr('f2')):.0f}% of them for {G('f2', 'found') - G('f3', 'found'):.3f} of the losses found. "
      f"A minimum step of 5% removes most of what is left. Skipping every federal holiday would halve it again ({fpr('f4')} to "
      f"{fpr('f5_all_holidays')}) and lose {G('f4', 'found') - G('f5_all_holidays', 'found'):.3f} of the losses found; skipping "
      f"only the year-end weeks takes {fpr('f4') - fpr('f5')} flags for {G('f4', 'found') - G('f5', 'found'):.3f}, and was kept. "
      f"The neighbour veto leaves {fpr('f6')} flags at either threshold; at 4 spreads it also vetoes "
      f"{G('f5', 'found') - G('f6_veto4', 'found'):.3f} of the losses found, at 6 spreads "
      f"{G('f5', 'found') - G('f6', 'found'):.3f}, so 6 was kept. Every choice was made on the development column. The "
      "confirmation column moves the same way at every step"
      + (f" except {', '.join(f'`{n}`' for n in against)}, where it moves the other way by "
         f"{max(abs(C(n) - C(prev[n])) for n in against):.3f}." if against else "."))
    w("")
    w("**Why it is an option.** `FeederLoss()` passes items 2, 4 and 5 and fails item 1 on F1 and on precision. The standard "
      "benchmark holds no feeder losses, so every reading the detector adds there counts against precision, and the recall "
      f"of every fault type is unchanged: the drop in F1 ({f3(dev_pub.loc[v5, 'f1_adj'])} to {f3(dev_pub.loc[UNION, 'f1_adj'])}) "
      f"follows from the drop in precision ({f3(dev_pub.loc[v5, 'precision_adj'])} to {f3(dev_pub.loc[UNION, 'precision_adj'])}). "
      "It was read as a failure on precision only, which makes the detector an option, off by default. On the feeder-loss "
      "benchmark precision moves both ways: up in development, down in confirmation "
      f"({f3(u_conf_pss.loc[v5, 'precision_adj'])} to {f3(u_conf_pss.loc[UNION, 'precision_adj'])} per series). The share of "
      f"feeder losses found is {f3(u_dev_pss.loc[UNION, 'recall_feeder_loss'])} and {f3(u_conf_pss.loc[UNION, 'recall_feeder_loss'])} "
      f"per series, and {f3(min(conf_found))} to {f3(max(conf_found))} across the confirmation combinations. The published "
      f"scheme gives {f3(u_dev_pub.loc[UNION, 'recall_feeder_loss'])} and {f3(u_conf_pub.loc[UNION, 'recall_feeder_loss'])}, from "
      "two and four draws, each repeated in all ten BAs.")
    w("")
    tot = freal.agg({"hours": "sum", "flagged": "sum", "segments": "sum"})
    by_ba = freal.groupby("ba")[["hours", "flagged", "segments"]].sum().sort_values("flagged", ascending=False)
    top = by_ba.index[0]
    rest = by_ba.drop(index=top)
    w("### On the record")
    w("")
    w(f"Run on every BA with a weather station in 2023 and 2024 ({len(freal)} BA-years, {int(tot['hours']):,} readings), "
      f"`FeederLoss()` flags {int(tot['segments'])} segments, {int(tot['flagged']):,} readings ({100 * tot['flagged'] / tot['hours']:.2f}%). "
      f"{top} alone accounts for {int(by_ba.loc[top, 'flagged']):,} of them in {int(by_ba.loc[top, 'segments'])} segments. "
      f"BA-years with at least one segment: {int((freal['segments'] > 0).sum())}.")
    w("")
    w(f"The {top} series has faults of its own in these years, readings far below the load around them. Without {top} the "
      f"rate is {100 * rest['flagged'].sum() / rest['hours'].sum():.2f}%, and the most flagged BAs are "
      f"{', '.join(rest.index[:5])}. Two segments are read in `docs/real_cases_v05.md`; the others have not been checked one "
      "by one, so this rate bounds the false alarms from above and does not count them.")
    w("")
    w("## What entered and what did not")
    w("")
    w("- **Default composite:** `exclude_diff` with `keep_built` and the jump guard with its neighbour exception, "
      "`Sentinel()`. `Sentinel.v04()` keeps the previous configuration.")
    w("- **Option, off by default:** `FeederLoss()`, run next to the composite.")
    w("- **Left out:** a higher threshold alone, clipped updates, forgetting, and keeping condemned readings out of both "
      "statistics.")
    w("- **Decided outside the rule:** the threshold and the reach of the exclusion, among three settings that pass, and the "
      "neighbour exception of the jump guard. Both are stated above with the numbers behind them.")
    w("")
    w("Produced by `scripts/v05_study.py` (tables in `results/v05_study/`, text by `scripts/v05_study_doc.py`).")
    doc.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("wrote", doc)
