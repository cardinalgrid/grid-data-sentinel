"""The 0.5 study, regenerated with the released code: every configuration tried, scored against the adoption rule.

    python scripts/v05_study.py --tidy <tidy> --isd <isd> [--workers 4]
    python scripts/v05_study.py --render-only

The candidates are keyword sets of ``Sentinel.v04()`` and of ``FeederLoss``; ``v05`` is ``Sentinel()`` itself. Each is
run on the standard benchmark (published and per-series injection), on the feeder-loss benchmark, on the
confirmation years, on the four winter storms and on the series as published. Results go to
``results/v05_study/*.csv``; ``docs/v05_study.md`` is rendered from them by ``render`` (see ``v05_study_doc.py``).
Each job writes its own file under ``results/v05_study/parts/``, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from grid_sentinel.benchmark import (
    DEFAULT_BAS,
    context_from_dict,
    load_series_local,
    make_context,
    on_contract,
    raw_fault_mask,
    run,
    union,
)
from grid_sentinel.detectors import FeederLoss, Sentinel
from grid_sentinel.events import EVENTS, event_window
from grid_sentinel.pandas import from_series, regularize
from grid_sentinel.stations import BA_STATION
from grid_sentinel.synthetic import ALL_TYPES, TYPES, inject_anomalies

OUT = ROOT / "results" / "v05_study"
PARTS = OUT / "parts"

# ---- candidates -------------------------------------------------------------------------------------------------
EXCL = {"exclude_faults": "all", "exclude_from": "diff", "m_diff": 4.0}
KB = {**EXCL, "keep_built": 0.5}
# name: (keyword arguments on top of Sentinel.v04(), None for Sentinel() itself; what the candidate tries)
COMPOSITE = {
    "v04": ({}, "The 0.4 composite, the baseline."),
    "m_diff_3.5": ({"m_diff": 3.5}, "A higher threshold on first differences, nothing else."),
    "robust": ({"robust": True}, "Winsorised updates: a flagged reading enters both statistics clipped to the acceptance boundary."),
    "forgetting": ({"half_life_hours": 336}, "Statistics that forget, with a half-life of two weeks."),
    "exclude_both": ({"exclude_faults": "all", "exclude_from": "both"},
                     "Condemned readings kept out of the statistics on levels and on differences."),
    "exclude_diff_gross": ({"exclude_faults": "gross", "exclude_from": "diff", "m_diff": 4.0},
                           "Only gross faults (non-positive, gross ratio, frozen) kept out of the statistics on differences; threshold 4."),
    "exclude_diff_3.5": ({**EXCL, "m_diff": 3.5}, "Every condemned reading kept out of the statistics on differences; threshold 3.5."),
    "exclude_diff": (EXCL, "Every condemned reading kept out of the statistics on differences; threshold 4."),
    "keep_built": (KB, "As above, and a reading above the expectation whose predecessor carried at least half of its excess is kept."),
    "jump_guard_alone": ({**KB, "jump_guard": 0.5, "jump_guard_share": math.inf},
                         "As above, and a lone jump is never kept by the cross-checks."),
    "v05": (None, "As above, but a jump the neighbours share (median step at least half of the BA's) is not a lone jump: `Sentinel()`."),
}

F0 = {"regime": "temperature", "confirm": "drift", "skip_day_aligned": False, "min_drop": 0.03, "skip_holidays": False,
      "neighbor_veto": None}
F1 = {**F0, "regime": None}
F2 = {**F1, "confirm": "plateau"}
F3 = {**F2, "skip_day_aligned": True}
F4 = {**F3, "min_drop": 0.05}
F5 = {**F4, "skip_holidays": "christmas"}
# name: (FeederLoss keyword arguments; what changes)
FEEDER = {
    "f0": (F0, "Steps matched within 50%, drift of the plateau bounded; profile with the temperature regime; 3% minimum step."),
    "f1": (F1, "Calendar-only profile (the temperature profile has no expectation where the temperature is missing)."),
    "f2": (F2, "Plateau test instead of the summed drift."),
    "f3": (F3, "Skip losses with both edges on midnights or an edge on a midnight where the day type changes."),
    "f4": (F4, "Minimum step 5%."),
    "f5_all_holidays": ({**F4, "skip_holidays": "all"}, "Skip the day before, of and after every federal holiday, and 24 Dec to 2 Jan."),
    "f5": (F5, "Skip only 24 December to 2 January."),
    "f6_veto4": ({**F5, "neighbor_veto": 4.0}, "Veto a step a neighbour makes in the same hour, at 4 spreads."),
    "f6": ({**F5, "neighbor_veto": 6.0}, "The same veto at 6 spreads: `FeederLoss()`."),
}

STD_CONFIGS = [("dev", "published", 0), ("dev", "per_series", 0), ("dev", "per_series", 1), ("dev", "per_series", 2),
               ("confirm", "published", 1), ("confirm", "published", 2), ("confirm", "per_series", 1), ("confirm", "per_series", 2)]
FEEDER_CONFIGS = [("dev", "published", 0), ("dev", "per_series", 0), ("dev", "per_series", 1),
                  ("confirm", "published", 1), ("confirm", "published", 2), ("confirm", "per_series", 1), ("confirm", "per_series", 2)]
YEARS = {"dev": (2023, 2024), "confirm": (2021, 2022)}
UNION = "U:v05+feeder_loss"
STORM_DETECTORS = [f"A:{n}" for n in COMPOSITE] + [UNION]


def make_sentinel(name: str) -> Sentinel:
    kw = COMPOSITE[name][0]
    return Sentinel() if kw is None else Sentinel.v04(**kw)


def det_fn(key: str):
    kind, name = key.split(":", 1)
    if kind == "A":
        return on_contract(lambda: make_sentinel(name))
    if kind == "U":
        return union(on_contract(Sentinel), on_contract(FeederLoss))
    raise ValueError(key)


# ---- jobs ---------------------------------------------------------------------------------------------------------
_TIDY: Path | None = None
_CTX = None


def _init(tidy: str, isd: str) -> None:
    global _TIDY, _CTX
    warnings.filterwarnings("ignore")
    np.seterr(all="ignore")
    _TIDY = Path(tidy)
    _CTX = make_context(Path(tidy), Path(isd), ROOT / "docs" / "neighbors.csv")


def job_bench(bench: str, key: str, period: str, scheme: str, seed: int) -> pd.DataFrame:
    types = TYPES if bench == "standard" else ALL_TYPES
    r = run(_TIDY, bas=DEFAULT_BAS, years=YEARS[period], rate=0.005, seed=seed, detectors={key: det_fn(key)},
            context=_CTX, types=types, per_series_seed=scheme == "per_series")
    r["bench"], r["period"], r["scheme"], r["seed"] = bench, period, scheme, seed
    return r


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def job_ladder(variant: str, period: str) -> pd.DataFrame:
    """Dense feeder losses only (2%, per-series seeds): share of losses found (more than half of their readings
    flagged) and readings flagged outside them; readings flagged on the same series as published."""
    kw = FEEDER[variant][0]
    seeds = (0, 1) if period == "dev" else (1, 2)
    events = found = outside = flagged = clean = clean_hours = 0
    for bi, ba in enumerate(DEFAULT_BAS):
        for year in YEARS[period]:
            s = load_series_local(_TIDY, ba, year)
            s = s[s > 0]
            ctx = context_from_dict(_CTX(ba, year))
            v, t = from_series(regularize(s))
            clean += int(FeederLoss(**kw).predict(v, t, ctx).is_anomaly.sum())
            clean_hours += int(np.isfinite(v).sum())
            for seed in seeds:
                inj = inject_anomalies(s, rate=0.02, types=("feeder_loss",), seed=1000 * seed + 37 * bi + year)
                reg = regularize(inj["value"])
                v2, t2 = from_series(reg)
                f = FeederLoss(**kw).predict(v2, t2, ctx).is_anomaly
                pos = reg.index.get_indexer(pd.DatetimeIndex(inj.index))
                lab = inj["kind"].to_numpy() == "feeder_loss"
                fl = f[pos]
                for i, j in _runs(lab):
                    events += 1
                    found += fl[i:j + 1].mean() > 0.5
                outside += int((fl & ~lab).sum())
                flagged += int(fl.sum())
    return pd.DataFrame([{"variant": variant, "period": period, "losses": events, "found": found / max(events, 1),
                          "flagged": flagged, "flagged_outside": outside, "clean_flagged": clean, "clean_hours": clean_hours}])


def job_storm(event: str, ba: str) -> pd.DataFrame:
    from scripts.storms import evaluate_window, inject_in_window

    ev = next(e for e in EVENTS if e["name"] == event)
    year = int(ev["start"][:4])
    s = load_series_local(_TIDY, ba, year)
    s = s[s > 0]
    if len(s) < 24 * 60:
        return pd.DataFrame()
    try:
        peak, win = event_window(s, ev["start"], ev["end"])
    except ValueError:
        return pd.DataFrame()
    inj = inject_in_window(s, win, 0.02, year + sum(map(ord, ba)))  # the draw of scripts/storms.py, seed 0
    raw_bad = raw_fault_mask(inj["clean"].to_numpy())
    ctx = _CTX(ba, year)
    rows = []
    for key in STORM_DETECTORS:
        fn = det_fn(key)
        rows.append({"event": event, "ba": ba, "detector": key,
                     **evaluate_window(fn(s, ctx), fn(inj["value"], ctx), win, peak, inj["label"].to_numpy(), raw_bad)})
    return pd.DataFrame(rows)


def job_real(ba: str, year: int) -> pd.DataFrame:
    """Each candidate against the 0.4 composite on the series as published."""
    s = load_series_local(_TIDY, ba, year)
    s = s[s > 0]
    if len(s) < 24 * 60:
        return pd.DataFrame()
    v, t = from_series(regularize(s))
    ctx = context_from_dict(_CTX(ba, year))
    base = Sentinel.v04().predict(v, t, ctx)
    a = np.asarray(base.is_anomaly, bool)
    no_exp = ~np.isfinite(base.expected)
    rows = []
    for name in COMPOSITE:
        r = base if name == "v04" else make_sentinel(name).predict(v, t, ctx)
        b = np.asarray(r.is_anomaly, bool)
        reason = np.asarray(r.reason).astype(str)
        rows.append({"ba": ba, "year": year, "candidate": name, "hours": int(np.isfinite(v).sum()), "flagged": int(b.sum()),
                     "removed": int((a & ~b).sum()), "added": int((b & ~a).sum()),
                     "removed_no_expectation": int((a & ~b & no_exp).sum()),
                     "removed_kept_as_extreme": int((a & ~b & (reason == "extreme_kept")).sum())})
    return pd.DataFrame(rows)


def job_feeder_real(ba: str, year: int) -> pd.DataFrame:
    try:
        s = load_series_local(_TIDY, ba, year)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    s = s[s > 0]
    if len(s) < 24 * 60:
        return pd.DataFrame()
    v, t = from_series(regularize(s))
    f = np.asarray(FeederLoss().predict(v, t, context_from_dict(_CTX(ba, year))).is_anomaly, bool)
    return pd.DataFrame([{"ba": ba, "year": year, "hours": int(np.isfinite(v).sum()), "flagged": int(f.sum()),
                          "segments": len(_runs(f))}])


JOBS = {"bench": job_bench, "ladder": job_ladder, "storm": job_storm, "real": job_real, "feeder_real": job_feeder_real}


def job_list() -> list[tuple]:
    jobs: list[tuple] = []
    for ev in EVENTS:
        for ba in sorted(BA_STATION):
            jobs.append(("storm", ev["name"], ba))
    for key in [f"A:{n}" for n in COMPOSITE] + [UNION]:
        for period, scheme, seed in STD_CONFIGS:
            jobs.append(("bench", "standard", key, period, scheme, seed))
    for key in ("A:v04", "A:v05", UNION):
        for period, scheme, seed in FEEDER_CONFIGS:
            jobs.append(("bench", "feeder", key, period, scheme, seed))
    for ba in DEFAULT_BAS:
        for year in (2021, 2022, 2023, 2024):
            jobs.append(("real", ba, year))
    for variant in FEEDER:
        for period in ("dev", "confirm"):
            jobs.append(("ladder", variant, period))
    for ba in sorted(BA_STATION):
        for year in (2023, 2024):
            jobs.append(("feeder_real", ba, year))
    return jobs


def part_path(job: tuple) -> Path:
    name = "__".join(str(x) for x in job).replace(":", "-").replace("+", "_")
    return PARTS / f"{name}.csv"


def do(job: tuple) -> tuple[tuple, float]:
    t0 = time.perf_counter()
    df = JOBS[job[0]](*job[1:])
    tmp = part_path(job).with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(part_path(job))
    return job, time.perf_counter() - t0


def collect() -> None:
    groups: dict[str, list[pd.DataFrame]] = {}
    for job in job_list():
        p = part_path(job)
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        label = f"{job[0]}_{job[1]}" if job[0] == "bench" else job[0]
        groups.setdefault(label, []).append(df)
    names = {"bench_standard": "standard", "bench_feeder": "feeder", "ladder": "feeder_ladder", "storm": "storms",
             "real": "real", "feeder_real": "feeder_real"}
    for label, frames in groups.items():
        pd.concat(frames, ignore_index=True).to_csv(OUT / f"{names[label]}.csv", index=False)


def check_candidates() -> None:
    """``v05`` is the default and ``f6`` is ``FeederLoss()``: the study scores the released configuration."""
    full = Sentinel.v04(**{**KB, "jump_guard": 0.5, "jump_guard_share": 0.5})
    assert vars(full).keys() == vars(Sentinel()).keys()
    for k in ("m_diff", "exclude_faults", "exclude_from", "keep_built", "jump_guard", "jump_guard_share"):
        assert getattr(full, k) == getattr(Sentinel(), k), k
    f6, default = FeederLoss(**FEEDER["f6"][0]), FeederLoss()
    for k, v in vars(default).items():
        assert getattr(f6, k) == v, k


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tidy")
    ap.add_argument("--isd")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args(argv)
    check_candidates()
    if not a.render_only:
        PARTS.mkdir(parents=True, exist_ok=True)
        pending = [j for j in job_list() if not part_path(j).exists()]
        print(f"{len(pending)} jobs to run of {len(job_list())}", flush=True)
        t0 = time.perf_counter()
        with Pool(a.workers, initializer=_init, initargs=(a.tidy, a.isd), maxtasksperchild=8) as pool:
            for k, (job, secs) in enumerate(pool.imap_unordered(do, pending), 1):
                print(f"[{k}/{len(pending)}] {job} {secs:.0f}s elapsed {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        collect()
    from scripts.v05_study_doc import render

    render(OUT, ROOT / "docs" / "v05_study.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
