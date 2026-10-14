# Grid Data Sentinel

**Anomaly detection and repair for load telemetry, benchmarked on public data.**

The FERC and NERC reviews of recent winter storms record the largest day-ahead load under-forecast by a balancing authority at 14% during Winter Storms Gerri and Heather (January 2024), with errors peaking early in the storms as the weather worsened ([FERC/NERC 2025](https://www.ferc.gov/news-events/news/ferc-nerc-issue-report-system-performance-during-january-2025-arctic-weather)). Those first cold hours are exactly the readings a statistical filter is most likely to mistake for bad data. NERC lists data validation, anomaly detection and provenance as preconditions for trustworthy operator-facing AI ([NERC 2024](https://www.nerc.com/pa/rrm/bpsa/Documents/Whitepaper-AI%20and%20ML%20in%20Real-Time%20System%20Operations.pdf)). This package is a small, dependency-light toolkit for that problem: detect corrupted readings in a load series, repair them with an audit trail, and measure how well each method does on public EIA-930 data, both with known injected faults and on the faults the public record actually contains.

> Status: **v0.5.0**. The composite no longer learns from the faults it condemns, which lifts recall on single dips from 0.71 to 0.95, and it keeps readings that build over hours above the expectation, so heat waves are no longer flagged; adjusted precision goes from 0.976 to 0.994. A detector for partial loss of load telemetry (`FeederLoss`) joins as an option. One estimator contract for every detector, a numpy-only core, the composite in streaming with a serialisable state; the 0.3 functions remain in `grid_sentinel.legacy` until 1.0. See `CHANGELOG.md`, `docs/api.md` and `docs/migration.md`; every configuration tried for 0.5, and the rule that chose among them, is in [docs/v05_study.md](docs/v05_study.md).

## Methods

Detection and repair are separate steps. Every detector returns the series unchanged, with a score, a flag and, for the composites, the reason for each flag (`confirmed_by`) and for each withdrawn alarm (`extreme kept`). Nothing is modified until `repair` is called explicitly with those flags, and `repair` returns the corrected series together with an audit log that records, reading by reading, the original value, the new value, the method and the reason. Gaps longer than 48 hours are left open.

| Detector | What it does | Origin |
|---|---|---|
| `TEDA` | Streaming detector based on typicality and eccentricity data analytics: keeps a recursive mean and variance and flags a reading when its normalised eccentricity exceeds (m²+1)/2k. No window, no training, no distributional assumption. Runs on levels or on first differences; optional winsorised update (`robust=True`) so that a run of bad readings does not inflate the variance. | Angelov (2014); the maintainer's M.Sc. work on outlier detection in demand curves |
| `SparseAutoencoder` | Window autoencoder (default 4 readings, code of size 2, L1 activity penalty) trained to reconstruct the series; the reconstruction error is the anomaly score. Implemented in NumPy, so there is no deep-learning dependency. | Guerra Filho et al., [*Energies* 2024, 17(24), 6403](https://doi.org/10.3390/en17246403) |
| `StuckValues` | Rule (causal since 0.4: a reading is flagged from the third identical one on): a run of identical consecutive readings is a frozen telemetry value, not a measurement. | Operational practice |
| `ProfileResidual` | Calendar-aware expectation, computed one day at a time: day types (workday, Saturday, Sunday, with the special days that behave like each) and an optional regime label; references are the last two weeks plus the same three weeks of the previous year; the residual is scaled by its recent MAD. The design follows [Cardinal Grid Note 2](https://github.com/cardinalgrid/notes), which tested the alternatives on 44 balancing authorities. | This package |
| `Sentinel(cross_checks=(), regime="load", extremes="off")` | Composite v0.2: the base detectors, each flag kept only if the reading (or the one before it) is also implausible against the profile; frozen runs, non-positive readings and gross ratios are faults regardless. | This package |
| `Sentinel.v04()` | Composite v0.3, streaming since v0.4: v0.2 with the profile regime read from the daily mean temperature, plus a rule for readings flagged *above* the expected value: the flag stands only if at least two of the available cross-checks fail to confirm the reading as genuine. Cross-checks: the profile (ratio to the expectation within 35%), the neighbours (the median normalised load of the BA's interchange partners is above its own recent 95th percentile at that hour) and the weather (the hour's temperature is in the BA's own seasonal tail for the heating or cooling regime). With two checks available the reading must fail both; with fewer, the v0.2 decision stands. Readings below the expectation, frozen runs and non-positive values are unchanged. | This package |
| `Sentinel()` (since v0.5) | The v0.3 composite with two guards. A condemned reading does not update the detector on first differences, whose next difference is taken from the expectation, and the threshold on differences is 4: earlier faults no longer widen the spread a later one is judged against. A reading above the expectation whose predecessor already carried at least half of its excess is kept, whatever the cross-checks say: a demand extreme builds over hours, a fault does not. A lone jump (a reading far above the expectation after one near it) is not kept by the cross-checks unless the neighbours stepped up in the same hour. `Sentinel.v04()` is the v0.3/v0.4 configuration. | This package |
| `FeederLoss` (optional, batch) | Partial loss of telemetry: a step down of at least 5 %, a plateau below the expectation and a matching step back up within 3 to 96 hours, in the hourly change of the log load net of the calendar profile's expected change. Steps on day-type midnights, in the Christmas week or shared with a neighbouring BA in the same hour are not confirmed. | This package |
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

[docs/real_cases_v05.md](docs/real_cases_v05.md) shows where the v0.5 composite disagrees with v0.4 on the same public series: heat waves in California and New York that v0.4 flagged and v0.5 keeps, and a repeated 23:00 artifact in Salt River Project's 2023 series that v0.5 flags and v0.4 let pass.

## Benchmark on EIA-930 with injected faults

Ten large balancing authorities, hourly demand for 2023 and 2024 from the EIA-930 balance files (via [ba-forecast-scorecard](https://github.com/cardinalgrid/ba-forecast-scorecard)), 0.5% of readings corrupted with labelled faults, every detector run on the corrupted series with its default settings.

<!-- results:start -->
Benchmark v0.5.0 run 2026-10-13: 20 BA-years (PJM, MISO, ERCO, CISO, SWPP, NYIS, ISNE, TVA, DUK, BPAT; 2023, 2024), 0.5% of readings corrupted, tolerance ±1 reading. Means over series.

| Detector | Precision | Precision (adj.) | Recall | F1 (adj.) | MCC | Spike | Dip | Zero | Stuck | Scale | s/series |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Composite v0.5, profile and weather only | 0.93 | 0.99 | 0.99 | 0.99 | 0.95 | 1.00 | 0.95 | 1.00 | 1.00 | 0.99 | 1.7 |
| Composite v0.5, profile and neighbours only | 0.93 | 0.99 | 0.99 | 0.99 | 0.95 | 1.00 | 0.95 | 1.00 | 1.00 | 0.99 | 3.3 |
| Composite v0.5 with neighbours by station distance instead of interchange | 0.93 | 0.99 | 0.99 | 0.99 | 0.95 | 1.00 | 0.95 | 1.00 | 1.00 | 0.99 | 2.7 |
| Composite v0.5 (v0.3 + faults kept out of the difference statistics + extremes that build over hours kept) | 0.93 | 0.99 | 0.99 | 0.99 | 0.95 | 1.00 | 0.95 | 1.00 | 1.00 | 0.99 | 3.5 |
| Composite v0.5 without the extremes rule | 0.92 | 0.98 | 0.99 | 0.98 | 0.94 | 1.00 | 0.95 | 1.00 | 1.00 | 0.99 | 3.5 |
| v0.3 function sentinel_v3 (legacy, for comparison) | 0.92 | 0.98 | 0.97 | 0.97 | 0.93 | 0.95 | 0.71 | 1.00 | 1.00 | 0.99 | 0.3 |
| Composite v0.3/v0.4 (v0.2 + temperature regime + extremes kept by two of three cross-checks) | 0.92 | 0.98 | 0.97 | 0.97 | 0.93 | 0.95 | 0.71 | 1.00 | 1.00 | 0.99 | 3.5 |
| v0.3 function sentinel_v2 (legacy, for comparison) | 0.90 | 0.96 | 0.98 | 0.97 | 0.93 | 0.95 | 0.70 | 1.00 | 1.00 | 1.00 | 0.2 |
| Composite v0.2 (v0.1 detectors confirmed by the calendar profile) | 0.90 | 0.96 | 0.98 | 0.97 | 0.93 | 0.95 | 0.70 | 1.00 | 1.00 | 1.00 | 1.6 |
| Composite v0.4 with exponential forgetting (half-life 14 days) | 0.88 | 0.94 | 0.96 | 0.95 | 0.91 | 0.95 | 0.99 | 1.00 | 1.00 | 0.95 | 3.5 |
| Composite v0.5 together with the optional feeder-loss detector | 0.84 | 0.90 | 0.99 | 0.93 | 0.89 | 1.00 | 0.95 | 1.00 | 1.00 | 0.99 | 5.0 |
| v0.3 function sentinel, the v0.1 composite (legacy) | 0.87 | 0.93 | 0.92 | 0.92 | 0.88 | 1.00 | 0.61 | 1.00 | 1.00 | 0.94 | 0.1 |
| Sparse autoencoder (window 4, code 2, factor 20) | 0.88 | 0.89 | 0.93 | 0.91 | 0.91 | 0.90 | 0.90 | 0.90 | 0.04 | 0.98 | 0.3 |
| Recursive TEDA (levels, m=4) | 0.97 | 0.98 | 0.82 | 0.89 | 0.89 | 0.60 | 0.21 | 1.00 | 0.00 | 0.93 | 0.1 |
| Recursive TEDA (levels, m=4, robust update) | 0.81 | 0.82 | 0.91 | 0.84 | 0.85 | 0.65 | 0.46 | 1.00 | 0.00 | 1.00 | 0.1 |
| Modified z-score (30 d, k=3.5) | 0.67 | 0.67 | 0.92 | 0.76 | 0.77 | 0.80 | 0.60 | 1.00 | 0.00 | 1.00 | 0.5 |
| Relative deviation from 5-h centred mean (15%) | 0.69 | 0.70 | 0.30 | 0.41 | 0.45 | 1.00 | 1.00 | 1.00 | 0.00 | 0.15 | 0.0 |
| Profile residual alone (day type, 2 weeks + last-year analogs, k=4) | 0.23 | 0.23 | 0.95 | 0.37 | 0.46 | 0.90 | 0.94 | 1.00 | 0.16 | 1.00 | 1.4 |
| Recursive TEDA (differenced, m=3) | 0.72 | 0.72 | 0.19 | 0.29 | 0.35 | 1.00 | 0.61 | 1.00 | 0.00 | 0.20 | 0.1 |
| Global IQR (k=1.5) | 0.17 | 0.17 | 0.93 | 0.27 | 0.38 | 0.60 | 0.78 | 1.00 | 0.10 | 1.00 | 0.0 |
| Rolling z-score (168 h, k=3) | 0.68 | 0.69 | 0.11 | 0.18 | 0.25 | 0.95 | 0.85 | 1.00 | 0.00 | 0.00 | 0.0 |
| Hampel filter (24 h, k=3) | 0.07 | 0.07 | 0.20 | 0.10 | 0.11 | 0.95 | 0.72 | 1.00 | 0.08 | 0.15 | 0.0 |
| stuck_rule | 0.46 | 0.55 | 0.05 | 0.10 | 0.14 | 0.00 | 0.00 | 0.00 | 1.00 | 0.01 | 0.1 |
| Feeder-loss detector alone (step down, plateau, matching step up) | 0.00 | 0.00 | 0.00 | 0.00 | -0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.6 |

Columns Spike to Scale are recall by anomaly type. Precision (adj.) and F1 (adj.) do not count as false alarms the flags on readings that the public series already had wrong (frozen runs, non-positive values, unit errors; see `docs/real_cases.md`); the plain precision does. Full table: `results/benchmark_by_series.csv`.
<!-- results:end -->

What the table says:

- **The v0.5 composite is the best detector on every count that matters**: adjusted precision 0.994 and recall 0.986, against 0.976 and 0.972 for v0.3/v0.4; dips, the weak spot of every earlier composite, go from 71% to 95%, spikes from 95% to 100%, and no other fault type loses recall.
- **Dips were missed because the detectors learned from the faults.** The eccentricity detectors keep no forgetting; every earlier spike, zero run or unit error widened the spread a later dip was judged against. Keeping condemned readings out of the difference statistics fixes that. Doing the same for the level statistics, or adding exponential forgetting (the `forgetting` row), also recovers the dips but flags genuine summer peaks.
- **The optional feeder-loss detector costs precision here** (0.994 to 0.899 in the union) because this benchmark injects no feeder losses: the readings it flags on the real series count as false alarms. Its own benchmark is below.
- **Plain precision understates the composites.** The source series already contain frozen runs and unit errors (see the survey above); a detector that flags them is right, but the injected-fault benchmark counts it as a false alarm. The adjusted columns exclude those readings from the denominator.
- **The autoencoder has the best recall on dips but misses frozen values**, which no reconstruction method sees: a frozen value is a perfectly reconstructable reading. The rule catches them all.
- **Reference detectors fail in the expected ways.** The global IQR rule flags every legitimate winter and summer peak; the Hampel filter and rolling z-score are blind to unit-error runs longer than their window; the modified z-score and the 15% rule, common in practice, sit in the middle.
- **The profile alone is not a detector.** With precision 0.23 it fires on every genuine departure from the calendar, first cold mornings included. As a confirmation of the level and difference detectors it is what lifts v0.2 above v0.1.
- **The injection scheme matters.** The table follows the published scheme, in which every BA gets its faults at the same hours, so one drawn fault counts ten times. With a different injection per BA (`--per-series-seed`, `results/per_series/`) the v0.3/v0.4 composite catches 75% of spikes rather than 95%; on that benchmark the v0.5 composite reaches adjusted precision 0.964 and F1 0.965 against 0.923 and 0.940, with spikes at 90% and dips at 100%.

### Partial loss of telemetry

A feeder or a metering point that drops out of a BA's total for hours and comes back is a fault no composite here sees: it looks like a plausible day. With 6- to 72-hour losses of 5% to 30% of the load added to the injection (`--feeder-loss`, `results/feeder_loss/`), every detector in the table above catches at most 4% of them. `FeederLoss` looks for the shape (a step down, a plateau below the expectation, a matching step back up) and finds about 60% of them with a different injection per BA (74% in 2021 and 2022), at a precision of 0.91 to 0.94 when only feeder losses are injected. Together with the composite, adjusted F1 on the feeder-loss benchmark goes from 0.54 to 0.75 (per-BA injections; 0.35 to 0.55 with the published scheme, which has only two distinct feeder-loss draws). On the series as published it flags 0.33% of readings across 43 BAs over 2023 and 2024, 40% of them in one BA with real faults (WALC). It stays optional: the default composite admits a new piece only if the standard benchmark's adjusted precision falls by no more than 0.01.

Defaults were chosen from sensitivity grids (`scripts/tune.py`, `results/tuning.csv`, and the v0.2 grid described in the changelog). The autoencoder threshold factor in particular is not portable across data: 3× the trimmed-mean error was right for 15-minute substation data in the 2024 paper; 20× is right for hourly BA data.

## The four winter events

For each of Winter Storm Uri (2021), Winter Storm Elliott (2022), Winter Storms Gerri and Heather (2024) and the January 2025 Arctic events, the 72 hours around each balancing authority's peak, on the series as reported: the share of BAs whose peak reading survives each detector, and, from a second run with faults injected into the same window, the recall on those faults. The FERC/NERC review of the January 2025 events records the largest under-forecasts of these storms early on, as the weather worsened; those are the readings a statistical filter is most likely to reject. Full tables, per-BA results and figures: `docs/storms.md`, `results/storms/`. The v0.5 composite keeps the same peaks as v0.3/v0.4 and flags half as many readings of the windows as reported.

| Detector | Uri: peak kept / recall | Elliott | Gerri and Heather | January 2025 |
|---|---|---|---|---|
| Recursive TEDA (levels) | 0.95 / 0.68 | 0.88 / 0.47 | 0.88 / 0.57 | 1.00 / 0.67 |
| Sparse autoencoder | 0.91 / 0.72 | 0.65 / 0.75 | 0.74 / 0.76 | 0.72 / 0.66 |
| Modified z-score | 0.86 / 0.71 | 0.58 / 0.74 | 0.77 / 0.34 | 0.88 / 0.74 |
| Composite v0.2 | 0.95 / 0.99 | 0.88 / 0.96 | 0.91 / 0.90 | 1.00 / 0.99 |
| Composite v0.3/v0.4 | 0.95 / 0.97 | 0.95 / 0.96 | 1.00 / 0.90 | 1.00 / 1.00 |
| Composite v0.5 | 0.95 / 0.99 | 0.95 / 0.98 | 1.00 / 0.92 | 1.00 / 1.00 |

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

det = Sentinel()                                             # the v0.5 composite; Sentinel.v04() is v0.3/v0.4
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

Partial loss of telemetry, in batch, alongside the composite:

```python
from grid_sentinel import FeederLoss

loss = FeederLoss().predict(values, timestamps, Context(neighbors={"MISO": (miso, miso_t)}))
flags = res.is_anomaly | loss.is_anomaly                    # loss.reason == "feeder_loss" where it fires
```

Streaming use of the TEDA detector, one reading at a time:

```python
from grid_sentinel import TEDA
det = TEDA(m=4.0)
for t, x in readings:
    d = det.update(t, x)
    if d.is_anomaly:
        ...
```

## Limitations

- The benchmark measures sensitivity to *injected* faults on top of series that already have faults of their own; the adjusted precision is the honest one, and it is still a benchmark on simple fault types (single-reading spikes and dips, zero runs, frozen values, ×10 and ×0.1 runs, and, optionally, partial feeder losses). Slow drift is not modelled.
- `TEDA` keeps all history unless `half_life_hours` is set. On a decade of data the running variance is dominated by the seasonal cycle; the composite keeps condemned readings out of the difference statistics, not the drift of the season.
- The rule that keeps extremes building over hours (`keep_built`) would also keep a fault that rises gradually and stays above the expectation; none of the injected types has that shape, and on the public series no such fault was found in the readings it keeps, but it is a known blind spot. Spikes smaller than the profile band (35%) are not confirmed. A genuine extreme that arrives as a one-hour step is kept only if the neighbours step up in the same hour; for a BA without neighbour readings its first hour is condemned. In the first hours of a series, before the profile has an expectation, the higher threshold on differences (4 instead of 3) misses some spikes that v0.4 caught.
- `FeederLoss` needs both edges of a loss: it confirms a loss only after the recovery, so it is batch only, and it misses losses smaller than about 5% or than five spreads of the hourly step (8% to 10% in the noisier BAs).
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
| v0.4 | October 2026 | Estimator contract for every detector, numpy-only core, streaming composite with serialisable state, optional forgetting, feeder-loss fault, hour-label alignment check between a forecast and its load |
| v0.5 | October 2026 | Faults kept out of the difference statistics, extremes that build over hours kept, optional feeder-loss detector, per-BA injection in the benchmark (this release) |
| v1.0 | December 2026 | Removal of the legacy functions, audit trail format, documentation and examples on public data, PyPI |

## Development

```bash
pip install -e ".[dev]"
ruff check src tests scripts && pytest -q
```

## Citing

Guerra Filho, R. W. C. (2026). *Grid Data Sentinel: anomaly detection and repair for load telemetry with recursive TEDA, sparse autoencoders and calendar-aware profiles* (v0.5.0). Zenodo. https://doi.org/10.5281/zenodo.23337413 (concept DOI for all versions: 10.5281/zenodo.22726013; v0.4.0: 10.5281/zenodo.23160968; v0.3.0: 10.5281/zenodo.22846631; v0.2.0: 10.5281/zenodo.22726014). See `CITATION.cff`.

## License

Apache-2.0 for code; text and figures CC BY 4.0. Analyses rely on public data only and represent the maintainer's own views.
