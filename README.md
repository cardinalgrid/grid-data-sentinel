# Grid Data Sentinel

**Anomaly detection and repair for load telemetry, benchmarked on public data.**

The FERC and NERC reviews of recent winter storms record the largest day-ahead load under-forecast by a balancing authority at 14% during Winter Storms Gerri and Heather (January 2024), with errors peaking early in the storms as the weather worsened ([FERC/NERC 2025](https://www.ferc.gov/news-events/news/ferc-nerc-issue-report-system-performance-during-january-2025-arctic-weather)). Those first cold hours are exactly the readings a statistical filter is most likely to mistake for bad data. NERC lists data validation, anomaly detection and provenance as preconditions for trustworthy operator-facing AI ([NERC 2024](https://www.nerc.com/pa/rrm/bpsa/Documents/Whitepaper-AI%20and%20ML%20in%20Real-Time%20System%20Operations.pdf)). This package is a small, dependency-light toolkit for that problem: detect corrupted readings in a load series, repair them with an audit trail, and measure how well each method does on public EIA-930 data, both with known injected faults and on the faults the public record actually contains.

> Status: **v0.4.0**. One estimator contract for every detector (`fit` / `predict` / `score` in batch, `update` one reading at a time), a core that depends on numpy only, and the whole composite, cross-checks included, in streaming with a serialisable state. The 0.3 functions remain in `grid_sentinel.legacy` for one minor version. See `docs/api.md` and `docs/migration.md`.

## Methods

Detection and repair are separate steps. Every detector returns the series unchanged, with a score, a flag and, for the composites, the reason for each flag (`confirmed_by`) and for each withdrawn alarm (`extreme kept`). Nothing is modified until `repair` is called explicitly with those flags, and `repair` returns the corrected series together with an audit log that records, reading by reading, the original value, the new value, the method and the reason. Gaps longer than 48 hours are left open.

| Detector | What it does | Origin |
|---|---|---|
| `TEDA` | Streaming detector based on typicality and eccentricity data analytics: keeps a recursive mean and variance and flags a reading when its normalised eccentricity exceeds (m²+1)/2k. No window, no training, no distributional assumption. Runs on levels or on first differences; optional winsorised update (`robust=True`) so that a run of bad readings does not inflate the variance. | Angelov (2014); the maintainer's M.Sc. work on outlier detection in demand curves |
| `SparseAutoencoder` | Window autoencoder (default 4 readings, code of size 2, L1 activity penalty) trained to reconstruct the series; the reconstruction error is the anomaly score. Implemented in NumPy, so there is no deep-learning dependency. | Guerra Filho et al., [*Energies* 2024, 17(24), 6403](https://doi.org/10.3390/en17246403) |
| `StuckValues` | Rule (causal since 0.4: a reading is flagged from the third identical one on): a run of identical consecutive readings is a frozen telemetry value, not a measurement. | Operational practice |
| `ProfileResidual` | Calendar-aware expectation, computed one day at a time: day types (workday, Saturday, Sunday, with the special days that behave like each) and an optional regime label; references are the last two weeks plus the same three weeks of the previous year; the residual is scaled by its recent MAD. The design follows [Cardinal Grid Note 2](https://github.com/cardinalgrid/notes), which tested the alternatives on 44 balancing authorities. | This package |
| `Sentinel(cross_checks=(), regime="load", extremes="off")` | Composite v0.2: the base detectors, each flag kept only if the reading (or the one before it) is also implausible against the profile; frozen runs, non-positive readings and gross ratios are faults regardless. | This package |
| `Sentinel()` | Composite v0.3, now streaming: v0.2 with the profile regime read from the daily mean temperature, plus a rule for readings flagged *above* the expected value: the flag stands only if at least two of the available cross-checks fail to confirm the reading as genuine. Cross-checks: the profile (ratio to the expectation within 35%), the neighbours (the median normalised load of the BA's interchange partners is above its own recent 95th percentile at that hour) and the weather (the hour's temperature is in the BA's own seasonal tail for the heating or cooling regime). With two checks available the reading must fail both; with fewer, the v0.2 decision stands. Readings below the expectation, frozen runs and non-positive values are unchanged. | This package |
| `neighbors`, `weather`, `stations` | Neighbour table from the public EIA-930 interchange record (`docs/neighbors.csv`, one year of hourly interchange, median absolute MW as weight, station distance as fallback), hourly ISD-Lite temperature per BA on the load's local index, and the BA-to-airport map. | EIA-930, NOAA ISD-Lite |
| `legacy.sentinel_v2` | The v0.2 function, unchanged (deprecated): the v0.1 detectors, each flag kept only if the reading (or a neighbour) is also implausible against the profile; frozen runs, non-positive readings and gross ratios to the expected value (outside 1/3 to 3×) are kept without the check. | This package |
| `RollingZScore`, `Hampel`, `IQR`, `ModifiedZScore`, `RelativeDeviation` | Reference detectors, including the two rules most common in utility practice: the Iglewicz-Hoaglin modified z-score and a 15% deviation from a centred mean. | Standard |

Plus `inject_anomalies` (labelled spikes, dips, zeros, stuck runs and unit errors for benchmarking), `intervals_from_mask` (flags grouped into intervals with a duration class: up to 1 h, 1 day, 1 week, 1 month, more), `repair` (a straight line for gaps up to an hour; for longer gaps the mean of the same interval one or more weeks before and after, shifted to meet the neighbouring good readings; gaps longer than 48 readings are left open rather than invented; every change logged with its reason and method), `day_types` (U.S. calendar with special days) and `score_labels` (precision, recall, F1, MCC, with an optional ±n tolerance).

## In pictures

Two weeks of real PJM hourly demand from EIA-930 (January 2024) with seven faults placed on it: two spikes, two dips, a two-hour zero, an eight-hour frozen value and a twelve-hour unit error (×0.1). Generated by `scripts/make_figures.py`.

![Composite sentinel on PJM](docs/figures/fig1_sentinel.png)

The v0.1 composite finds every fault except the milder dip (30% below the clean value at 12:00 UTC on 20 January, a level PJM has seen at that hour) and raises four false alarms on genuine steep ramps; the v0.2 confirmation against the profile is what removes those. The sparse autoencoder on the same window:

![Sparse autoencoder on PJM](docs/figures/fig1b_autoencoder.png)

It catches the mild dip but not the frozen value, which reconstructs perfectly, and it flags the readings next to a fault as often as the fault itself. This is why the detectors are combined.

What the streaming detector computes, reading by reading, is one number against one threshold:

![Recursive TEDA score](docs/figures/fig2_teda_scores.png)

The repair step fills the flagged readings and logs every change with the original value, the new value, the reason and the method. Long gaps are left open rather than invented.

![Repair with audit trail](docs/figures/fig3_repair.png)

Two failure modes that window-based filters get wrong. On the left, a frozen value: the Hampel filter says nothing about the run and instead flags the ramp after it. On the right, a unit error lasting twelve hours: the Hampel filter, with its 24-hour window, sees the corrupted run as the new normal and flags the recovery.

![Stuck and scale runs](docs/figures/fig4_stuck_and_scale.png)

## Real anomalies in the public record

Nothing needs to be injected to test the detectors: the EIA-930 demand series, as published, contain about 11,500 zero hours, 9,900 frozen hours, 5,800 hour-to-hour jumps of more than 2× and 2,600 unit-error hours across 6.5 million BA-hours. [docs/real_cases.md](docs/real_cases.md) has the survey by BA and a gallery of named cases with the flags of both composites and the repaired series, produced by `scripts/real_cases.py`. Two of them:

![BANC, April 2019](docs/figures/real/banc_2019-03-27.png)

BANC reported the same 1,670 MW for 204 hours, then two weeks in which 80 hourly readings ranged from −2.1 million to +1.8 million MW on a system of about 1.7 GW. The v0.2 composite flags the frozen run through the stuck rule and the nonsense through the non-positive and gross-ratio rules, and the repair leaves the eight-day gap open, as it should.

![CPLE, November 2023](docs/figures/real/cple_2023-10-28.png)

The first cold morning of the season in the Carolinas: demand peaks at 7 a.m. for the first time since March, 45% above what a calendar profile expects. This is not a fault, and neither composite raises an alarm. A detector built on the profile alone would raise its loudest alarm of the year here, which is why the profile is used to confirm, never to accuse.

## Benchmark on EIA-930 with injected faults

Ten large balancing authorities, hourly demand for 2023 and 2024 from the EIA-930 balance files (via [ba-forecast-scorecard](https://github.com/cardinalgrid/ba-forecast-scorecard)), 0.5% of readings corrupted with labelled faults, every detector run on the corrupted series with its default settings.

<!-- results:start -->
Benchmark v0.3.0 run 2026-09-17: 20 BA-years (PJM, MISO, ERCO, CISO, SWPP, NYIS, ISNE, TVA, DUK, BPAT; 2023, 2024), 0.5% of readings corrupted, tolerance ±1 reading. Means over series.

| Detector | Precision | Precision (adj.) | Recall | F1 (adj.) | MCC | Spike | Dip | Zero | Stuck | Scale | s/series |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Composite v0.3, profile and neighbours only | 0.91 | 0.97 | 0.98 | 0.97 | 0.93 | 0.85 | 0.68 | 1.00 | 1.00 | 1.00 | 0.2 |
| Composite v0.3 (v0.2 + temperature regime + extremes kept by two of three cross-checks) | 0.92 | 0.98 | 0.97 | 0.97 | 0.93 | 0.95 | 0.71 | 1.00 | 1.00 | 0.99 | 0.3 |
| Composite v0.3 with neighbours by station distance instead of interchange | 0.92 | 0.98 | 0.97 | 0.97 | 0.93 | 0.90 | 0.70 | 1.00 | 1.00 | 0.99 | 0.3 |
| Composite v0.3, profile and weather only | 0.92 | 0.98 | 0.97 | 0.97 | 0.93 | 0.85 | 0.68 | 1.00 | 1.00 | 0.99 | 0.2 |
| Composite v0.2 (v0.1 detectors confirmed by the calendar profile) | 0.90 | 0.96 | 0.98 | 0.97 | 0.93 | 0.95 | 0.70 | 1.00 | 1.00 | 1.00 | 0.2 |
| Composite v0.2 with the temperature regime, no extremes rule | 0.91 | 0.97 | 0.97 | 0.97 | 0.93 | 1.00 | 0.72 | 1.00 | 1.00 | 0.99 | 0.2 |
| Composite v0.1 (TEDA levels + TEDA differences + stuck rule) | 0.87 | 0.93 | 0.92 | 0.92 | 0.88 | 1.00 | 0.61 | 1.00 | 1.00 | 0.94 | 0.1 |
| Sparse autoencoder (window 4, code 2, factor 20) | 0.89 | 0.89 | 0.93 | 0.91 | 0.91 | 0.90 | 0.90 | 0.90 | 0.04 | 0.98 | 0.3 |
| Recursive TEDA (levels, m=4) | 0.97 | 0.98 | 0.82 | 0.89 | 0.89 | 0.60 | 0.21 | 1.00 | 0.00 | 0.93 | 0.0 |
| Recursive TEDA (levels, m=4, robust update) | 0.81 | 0.82 | 0.91 | 0.84 | 0.85 | 0.65 | 0.46 | 1.00 | 0.00 | 1.00 | 0.0 |
| Modified z-score (30 d, k=3.5) | 0.67 | 0.68 | 0.92 | 0.76 | 0.78 | 0.85 | 0.60 | 1.00 | 0.00 | 1.00 | 0.0 |
| Relative deviation from 5-h centred mean (15%) | 0.69 | 0.70 | 0.30 | 0.41 | 0.45 | 1.00 | 1.00 | 1.00 | 0.00 | 0.15 | 0.0 |
| Profile residual alone (day type, 2 weeks + last-year analogs, k=4) | 0.23 | 0.23 | 0.95 | 0.37 | 0.46 | 0.90 | 0.94 | 1.00 | 0.16 | 1.00 | 0.1 |
| Recursive TEDA (differenced, m=3) | 0.72 | 0.72 | 0.19 | 0.29 | 0.35 | 1.00 | 0.61 | 1.00 | 0.00 | 0.20 | 0.0 |
| Global IQR (k=1.5) | 0.17 | 0.17 | 0.93 | 0.27 | 0.38 | 0.60 | 0.78 | 1.00 | 0.10 | 1.00 | 0.0 |
| Rolling z-score (168 h, k=3) | 0.70 | 0.70 | 0.11 | 0.18 | 0.26 | 0.95 | 0.85 | 1.00 | 0.00 | 0.00 | 0.0 |
| stuck_rule | 0.46 | 0.55 | 0.06 | 0.11 | 0.16 | 0.00 | 0.00 | 0.00 | 1.00 | 0.01 | 0.0 |
| Hampel filter (24 h, k=3) | 0.07 | 0.07 | 0.20 | 0.10 | 0.11 | 0.95 | 0.72 | 1.00 | 0.08 | 0.15 | 0.0 |

Columns Spike to Scale are recall by anomaly type. Precision (adj.) and F1 (adj.) do not count as false alarms the flags on readings that the public series already had wrong (frozen runs, non-positive values, unit errors; see `docs/real_cases.md`); the plain precision does. Full table: `results/benchmark_by_series.csv`.
<!-- results:end -->

What the table says:

- **The v0.2 composite is the best detector on every count that matters**: adjusted precision 0.96 and recall 0.98, against 0.93 and 0.92 for v0.1; every fault type is caught, and dips, the weak spot of v0.1, go from 61% to 70%.
- **Plain precision understates both composites.** The source series already contain frozen runs and unit errors (see the survey above); a detector that flags them is right, but the injected-fault benchmark counts it as a false alarm. The adjusted columns exclude those readings from the denominator.
- **The autoencoder has the best recall on dips but misses frozen values**, which no reconstruction method sees: a frozen value is a perfectly reconstructable reading. The rule catches them all.
- **Reference detectors fail in the expected ways.** The global IQR rule flags every legitimate winter and summer peak; the Hampel filter and rolling z-score are blind to unit-error runs longer than their window; the modified z-score and the 15% rule, common in practice, sit in the middle.
- **The profile alone is not a detector.** With precision 0.23 it fires on every genuine departure from the calendar, first cold mornings included. As a confirmation of the level and difference detectors it is what lifts v0.2 above v0.1.

Defaults were chosen from sensitivity grids (`scripts/tune.py`, `results/tuning.csv`, and the v0.2 grid described in the changelog). The autoencoder threshold factor in particular is not portable across data: 3× the trimmed-mean error was right for 15-minute substation data in the 2024 paper; 20× is right for hourly BA data.

## The four winter events

For each of Winter Storm Uri (2021), Winter Storm Elliott (2022), Winter Storms Gerri and Heather (2024) and the January 2025 Arctic events, the 72 hours around each balancing authority's peak, on the series as reported: the share of BAs whose peak reading survives each detector, and, from a second run with faults injected into the same window, the recall on those faults. The FERC/NERC review of the January 2025 events records the largest under-forecasts of these storms early on, as the weather worsened; those are the readings a statistical filter is most likely to reject. Full tables, per-BA results and figures: `docs/storms.md`, `results/storms/`.

| Detector | Uri: peak kept / recall | Elliott | Gerri and Heather | January 2025 |
|---|---|---|---|---|
| Recursive TEDA (levels) | 0.95 / 0.68 | 0.88 / 0.47 | 0.88 / 0.57 | 1.00 / 0.67 |
| Sparse autoencoder | 0.91 / 0.72 | 0.65 / 0.75 | 0.74 / 0.76 | 0.72 / 0.66 |
| Modified z-score | 0.86 / 0.71 | 0.58 / 0.74 | 0.77 / 0.34 | 0.88 / 0.74 |
| Composite v0.2 | 0.95 / 0.99 | 0.88 / 0.96 | 0.91 / 0.90 | 1.00 / 0.99 |
| Composite v0.3 | 0.95 / 0.97 | 0.95 / 0.96 | 1.00 / 0.90 | 1.00 / 1.00 |

![Share of BAs whose peak reading survives, per event](results/storms/fig2_peak_kept.png)

## Install and use

```bash
pip install git+https://github.com/cardinalgrid/grid-data-sentinel            # core: numpy only
pip install "grid-data-sentinel[pandas] @ git+https://github.com/cardinalgrid/grid-data-sentinel"   # + pandas adapter, public data, benchmark
```

Batch, on numpy arrays (hourly readings and local-time timestamps on a regular grid):

```python
import numpy as np
from grid_sentinel import Sentinel, Repairer, Context

det = Sentinel()                                             # the v0.3 composite; Sentinel(cross_checks=(), regime="load", extremes="off") is v0.2
res = det.predict(values, timestamps, Context(temperature_f=(temp_f, temp_t), neighbors={"MISO": (miso, miso_t)}))
res.is_anomaly, res.reason, res.expected, res.checks       # reason: teda_level, stuck, gross_ratio, extreme_kept, ...
res.summary(); res.explain(i); res.intervals()
fixed, audit = Repairer(method="equivalent_days").transform(values, timestamps, res.is_anomaly)   # detection never modifies the series
```

The same detector, one reading at a time, with a state that can be saved and restored:

```python
from grid_sentinel import state_to_json, state_from_json

det = Sentinel().fit(history_values, history_timestamps, history_context)   # warm start from the history
for t, x, temp, nb in live_readings:                                          # nb = {"MISO": 71234.0, ...}
    d = det.update(t, x, temperature_f=temp, neighbors=nb)                  # Decision: is_anomaly, reason, expected, checks
saved = state_to_json(det.get_state())
det2 = Sentinel().set_state(state_from_json(saved))                          # continues exactly where det stopped
```

Every detector follows the same contract (`TEDA`, `StuckValues`, `ProfileResidual`, `SparseAutoencoder`, the baselines): `fit`, `predict`, `score`, `update`, `get_state`, `set_state`, `get_params`. For a streaming detector, `predict` is the sequence of `update` calls replayed, so batch and stream agree exactly. From pandas: `from grid_sentinel.pandas import predict_series, regularize, context_from`.

Command line:

```bash
grid-sentinel detect my_series.csv --column demand --method sentinel --repair --out detections.csv
grid-sentinel benchmark --tidy path/to/ba-forecast-scorecard/data/tidy --isd path/to/isd --out results
python scripts/storms.py --tidy path/to/ba-forecast-scorecard/data/tidy --isd path/to/isd   # the four winter events
python scripts/render_readme.py                                       # refresh the table above
python scripts/make_figures.py --tidy path/to/ba-forecast-scorecard/data/tidy   # the figures above
python scripts/real_cases.py --tidy path/to/ba-forecast-scorecard/data/tidy     # the survey and gallery
```

Streaming use of the TEDA detector, one reading at a time:

```python
from grid_sentinel import RecursiveTEDA
det = RecursiveTEDA(m=4.0)
for x in readings:
    r = det.update(x)
    if r.is_anomaly:
        ...
```

## Limitations

- The benchmark measures sensitivity to *injected* faults on top of series that already have faults of their own; the adjusted precision is the honest one, and it is still a benchmark on simple fault types (single-reading spikes and dips, zero runs, frozen values, ×10 and ×0.1 runs). Slow drift and partial feeder loss are not modelled yet.
- `RecursiveTEDA` keeps all history. On a decade of data the running variance is dominated by the seasonal cycle. The winsorised update protects it from bad runs, not from drift; a forgetting factor is planned.
- Without a temperature series the profile's regime is read from yesterday's load (morning peak or not). With one, the regime is the daily mean temperature (heating below 59 °F, cooling above 72 °F); one airport represents each BA.
- Neighbours come from one year of aggregate interchange (2024) and are treated as fixed; BAs whose partners are outside the EIA-930 record (Canada, Mexico) fall back to station distance and may end with a single neighbour, in which case the neighbour check is unavailable and the rule requires both remaining checks to fail.
- The composite decides each reading when it arrives. Compared with the 0.3 batch functions this costs a reading per frozen run (flagged from the third identical value) and the look-ahead of the profile confirmation; `docs/migration.md` lists every such difference and its measured size.
- `predict` is a replay of `update` in Python: about 4 s per BA-year for `Sentinel`, against 0.3 s for the vectorised 0.3 function it reproduces.
- One series, one BA, one detector at a time. No claims are made about any operator's internal data quality; the public record is what it is.

## Roadmap

| Version | Target | Scope |
|---|---|---|
| v0.1 | September 2026 | TEDA + autoencoder + rules, batch mode, benchmark on EIA-930 |
| v0.2 | September 2026 | Calendar-aware profile and confirmation, interval table, equivalent-day repair, practice baselines, survey and gallery of real anomalies |
| v0.3 | September 2026 | Temperature regime, neighbour and weather cross-checks, preservation of genuine extremes, evaluation on four winter events |
| v0.4 | September 2026 | Estimator contract for every detector, numpy-only core, streaming composite with serialisable state, optional forgetting, feeder-loss fault (this release) |
| v1.0 | December 2026 | Removal of the legacy functions, PyPI, documentation and examples on public data, better recall on dips and feeder loss |
| v1.0 | December 2026 | Audit trail format, documentation, examples on public data, PyPI |

## Development

```bash
pip install -e ".[dev]"
ruff check src tests scripts && pytest -q
```

## Citing

Guerra Filho, R. W. C. (2026). *Grid Data Sentinel: anomaly detection and repair for load telemetry with recursive TEDA, sparse autoencoders and calendar-aware profiles* (v0.2.0). Zenodo. https://doi.org/10.5281/zenodo.22726014 (concept DOI for all versions: 10.5281/zenodo.22726013). See `CITATION.cff`.

## License

Apache-2.0 for code; text and figures CC BY 4.0. Analyses rely on public data only and represent the maintainer's own views.
