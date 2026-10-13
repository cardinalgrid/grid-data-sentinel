# Changelog

## 0.5.0 (unreleased)

The composite stops learning from the faults it condemns and stops condemning heat waves; a detector for partial loss of telemetry joins as an option.

Changed
- `Sentinel()` defaults. A condemned reading no longer updates the eccentricity detector on first differences: its state goes back to what it was, and the next difference is taken from the expectation (`exclude_faults="all"`, `exclude_from="diff"`); the threshold on differences is `m_diff=4` (was 3). The detectors keep no forgetting, so in 0.4 every earlier spike, zero run or unit error widened the spread a later dip was judged against; recall on single dips was 0.71. Excluding condemned readings from the level detector as well was tried and rejected: the level spread then stays narrow and genuine summer peaks get flagged (adjusted precision 0.87).
- `Sentinel()` keeps a reading above the expectation whose predecessor already carried at least half of its excess (`keep_built=0.5`), whatever the cross-checks say. A demand extreme builds over hours; a fault does not. On the series as published, the 0.4 composite flagged readings of genuine heat waves that the temperature and neighbour checks did not confirm (CISO in June 2021 and July 2023, NYISO and PJM in June 2021, among others: 126 readings in the ten benchmark BAs over 2021 to 2024).
- `Sentinel()` no longer lets the cross-checks keep a lone jump (`jump_guard=0.5`): a reading above the expectation whose predecessor carried less than half of its excess, unless the median load of the BA's neighbours stepped up in the same hour by at least half as much (`jump_guard_share=0.5`). The cross-checks look at the weather and at the neighbours' level, not at the jump, and kept some injected spikes on cold nights. Without the neighbour exception the rule would condemn the first hour of a genuine extreme that arrives as a step; without neighbour readings it still does, and a test documents both cases.
- `Sentinel.v04()` returns the 0.4 configuration, which also reproduces the 0.3 composite. The benchmark keeps it under its table name `sentinel_v3`; the new default is `sentinel_v5`.

Added
- `FeederLoss` (batch only, optional): a step down, a plateau and a matching step back up in the hourly change of the log load net of the calendar profile's expected change. The onset must be at least 5 % and 5 robust spreads (scaled MAD of the previous 28 days, updated daily); the recovery must come within 3 to 96 hours and match the onset within half its size; the readings in between must sit below the line joining the residual before and after by more than half the step in at least 80 % of the hours. Candidates are dropped when both edges fall on midnights or one falls on a midnight where the day type changes (the profile switches day group there), when an edge falls between 24 December and 2 January, and when a neighbouring BA makes a step of the same sign in the same hour (6 spreads). Exported from `grid_sentinel` and `grid_sentinel.detectors`.
- `Sentinel(exclude_faults="gross")` keeps only non-positive readings, gross ratios and frozen runs out of the statistics; `exclude_from` chooses `"level"`, `"diff"` or `"both"`.
- Benchmark: `--per-series-seed` draws a different injection for each BA (seed `1000 * seed + 37 * i + year`). The published scheme (`seed + year`) puts the faults at the same hours in every BA, so one drawn fault counts ten times; `summary.json` records which scheme was used. `benchmark.union` combines detectors. New benchmark entries: `sentinel_v5+feeder_loss`, `feeder_loss`, `sentinel_v5_no_extremes` and the cross-check ablations of the new composite.
- `scripts/v05_bench.py`, `scripts/real_diff.py`, `scripts/tune_feeder.py`, `scripts/feeder_clean_survey.py`, `scripts/dips_study.py`: the studies behind this release. `scripts/storms.py --variants` runs composite configurations on the four winter events without touching the docs.

Fixed
- `scripts/real_cases.py` ran with the 0.3 names removed from the top-level namespace in 0.4 and failed; it imports them from `grid_sentinel.legacy`. The SWPP case text quotes the survey's own count (82 frozen runs, 2,919 hours) instead of a hard-coded figure.

Results
- Standard benchmark (20 BA-years, 0.5 % faults, published injection scheme): adjusted precision 0.994 against 0.976 for 0.4, adjusted F1 0.990 against 0.974; recall on dips 0.95 against 0.71 and on spikes 1.00 against 0.95; zeros 1.00, frozen 1.00 and unit errors 0.99, unchanged.
- Same benchmark with a different injection per BA (`--per-series-seed`, seed 0, `results/per_series/`): adjusted precision 0.964 against 0.923, adjusted F1 0.965 against 0.940; spikes 0.90 against 0.76, dips 1.00 against 0.86, unit errors 0.97 against 0.97.
- Confirmation on years not used in development (2021 and 2022, seeds 1 and 2, both injection schemes): adjusted F1 0.996 against 0.963 (published scheme) and 0.991 against 0.963 (per-BA injections); adjusted precision is equal or higher in each of the eight seed-year combinations. A further check on 2019 and 2020 (seeds 3 and 4): adjusted F1 0.977 against 0.962 (published scheme) and 0.964 against 0.952 (per-BA), higher in all eight combinations; in one, a spike in the first hours of the series, before the profile has an expectation, is missed by the higher threshold on differences.
- Winter events (`docs/storms.md`): on the 72 hours around each BA's peak in Winter Storms Uri, Elliott, Gerri and Heather and the January 2025 Arctic events, the peak reading survives in the same share of BAs as with 0.4 (0.95, 0.95, 1.00, 1.00), the composite flags half as many readings of the uninjected windows, and recall on faults injected into the windows is equal or higher (0.99, 0.98, 0.92, 1.00 against 0.97, 0.95, 0.90, 1.00).
- On the series as published, ten benchmark BAs over 2021 to 2024: 627 readings flagged against 856 for 0.4, none of them new. The 229 that leave are 102 morning ramps flagged at the start of each year, before the profile has history, and 126 genuine heat-wave readings. On every BA with a weather station over 2023 and 2024: 1,680 against 2,300; 706 leave (ramps without expectation, heat-wave peaks in the West) and 86 enter, which on inspection are faults of the published series: in SRP from May to early July 2023 the 23:00 reading returns to the level of 17:00 to 18:00, and WALC has drops to a third of its load.
- Feeder loss (`--feeder-loss`): with the published scheme, the composite together with `FeederLoss` reaches adjusted F1 0.549 against 0.353 for the composite alone, recall on feeder losses 0.37 against 0.03; the published scheme has two distinct feeder-loss draws repeated over the ten BAs. With per-BA injections (seeds 0 and 1), recall 0.59 and F1 0.75 against 0.54; in 2021 and 2022 (seeds 1 and 2), recall 0.74 and F1 0.79 against 0.51. With only feeder losses injected at 2 %, it finds about two thirds of them at a precision of 0.91 to 0.94.
- Why `FeederLoss` stays optional: on the standard benchmark the union with the composite drops adjusted precision from 0.994 to 0.899. The detector flags 73 readings on the 20 uninjected benchmark series (three segments: two short drops at SWPP and a rainy Sunday at Duke), and they weigh heavily in series with few injected faults. On every BA with a station, 2023 and 2024, it flags 0.33 % of readings, 40 % of them at WALC. The adoption rule, fixed before the first number, admits a new piece into the default composite only if the standard benchmark's adjusted precision falls by no more than 0.01.

## 0.4.0 (2026-10-04)

The interface release: the methods do not change, the way of using them does.

Added
- One estimator contract for every detector (`grid_sentinel.base.Detector`): `fit`, `predict`, `score`, `fit_predict` in batch; `update` one reading at a time for detectors that declare `supports_streaming`; `reset`; `get_state` / `set_state` (plain types and numpy arrays, serialisable with `state_to_json` / `state_from_json`); `get_params` / `set_params` and a `__repr__` in the scikit-learn convention, without depending on scikit-learn.
- Classes: `TEDA` (with optional exponential forgetting, `half_life_hours`), `StuckValues`, `ProfileResidual` (the calendar-aware profile computed one day at a time), `SparseAutoencoder`, `ModifiedZScore` (streams), `RollingZScore`, `Hampel`, `IQR`, `RelativeDeviation`, and `Sentinel`, the composite as one streaming detector with the cross-checks (`detectors.crosscheck.NeighborCheck`, `WeatherCheck`, `preserve_one`) computed reading by reading. `Sentinel(cross_checks=(), regime="load", extremes="off")` is the v0.2 composite; the defaults are v0.3.
- Standard inputs and outputs: `as_series` validates readings and a regular hourly grid of local timestamps; `Context` carries temperature, neighbours and regime series; `DetectionResult` and `Decision` carry fixed fields and fixed reason codes (`teda_level`, `teda_diff`, `stuck`, `non_positive`, `gross_ratio`, `profile`, `baseline`, `dropped`, `extreme_kept`), with `intervals()`, `summary()`, `explain()` and `to_pandas()`.
- `Repairer` with an `Audit` (index, timestamp, original, repaired, method, reason); `intervals_from_mask` and `summarize_intervals` in numpy.
- Core without pandas: `numpy` is the only dependency. `pandas`, `pyarrow` and `requests` are the `[pandas]` extra, used by the adapter (`grid_sentinel.pandas`: `from_series`, `regularize`, `context_from`, `predict_series`), the public-data loaders, the benchmark, the scripts and `grid_sentinel.legacy`. A test imports the core with pandas blocked.
- Batch equals stream: for every streaming detector `predict` is the replay of `update`, and the tests assert equal flags and scores; a state saved mid-series and restored continues with the same decisions.
- Injection: opt-in `feeder_loss` fault (a run of 6 to 72 readings at 70% to 95% of the true load). Benchmark: `--feeder-loss`, `--with-legacy` (the v0.3 functions side by side), variant `sentinel_v3_forgetting` (half-life 14 days).
- `docs/api.md`, `docs/migration.md`.
- `grid_sentinel.consistency` (numpy only): `neighbour_spikes`, an hour more than 40% away from the mean of its two neighbours and from the day's level (a system load does not jump that far for one hour and back), and `forecast_faults`, a published forecast that is missing, non-positive or more than half away from the outcome at some hour; `day_is_plausible` and `forecast_is_plausible` reduce them to a verdict for the day.
- `grid_sentinel.alignment` (`[pandas]` extra): `alignment_table` and `alignment_check` measure, per year, whether a forecast moved by one or two hours matches its load better than as published, the signature of a forecast and a load stamped on different hour conventions. It says the two series disagree on the hour label, not which one is wrong.

Changed
- `fetch_isd_lite` retries an unreachable archive and then treats the year as missing instead of failing.
- The v0.3 functions (`sentinel`, `sentinel_v2`, `sentinel_v3`, `stuck_values`, `profile_residual`, the baselines, `repair`, `RecursiveTEDA`, ...) live in `grid_sentinel.legacy` with a `DeprecationWarning`, unchanged in behaviour, and are still reachable as `grid_sentinel.<name>`; removed in 1.0.
- `StuckValues` is causal: a run is flagged from its `min_run`-th identical reading on; a missing reading breaks the run.
- `Sentinel` looks at the reading and the previous one when confirming a base alarm against the profile (v0.3 also looked at the next one); missing readings are unflagged; the weather cross-check uses yesterday's mean temperature for the regime (v0.3 used the same day's). On PJM 2024 the causal composite flags nothing that v0.3 did not, and v0.3 flags at most 0.5% of readings more, all explained by the two rules above.
- `SparseAutoencoder.predict` requires `fit`; `fit_predict` does both; `causal=True` scores only the window ending at each reading, which is what `update` does.

Regression on the 20 BA-year benchmark (0.5% injected faults, adjusted metrics), classes against the v0.3 functions run side by side: `Sentinel()` F1 0.974 against 0.974 for `legacy.sentinel_v3`, with identical recall by fault type (spike 0.95, dip 0.71, zero 1.00, stuck 1.00, scale 0.99); the v0.2 configuration 0.972 against 0.973 (the causal frozen-value rule and the missing look-ahead cost 0.001). `Sentinel(half_life_hours=336)` (forgetting) reaches recall 0.99 on dips at the cost of adjusted precision (0.94 against 0.98) and F1 0.947; it stays an option. `StuckValues` alone: F1 0.097 against 0.114 for the v0.3 rule, the price of deciding a frozen run without seeing its end. Cost: `Sentinel.predict` is a Python replay, 3.4 s per BA-year against 0.29 s for the vectorised v0.3 function it reproduces.

On the four winter events (`docs/storms.md`) the streaming composite reproduces the v0.3 tables: peak kept and recall in the window identical for `Sentinel()`, the v0.2 configuration and every reference detector; the largest difference in any variant is one BA out of 43 (the profile-and-neighbours variant keeps one more peak in Elliott and in Gerri and Heather, the autoencoder one fewer in Gerri and Heather).

With the opt-in `feeder_loss` fault added to the injection (`results/feeder_loss/`), every detector misses it: recall 0.03 for the composites and at most 0.04 for anything else. A partial loss of 5% to 30% of the load for up to three days looks like a plausible day to every method here; this is the known weakness left for 1.0, now measured.

## 0.3.0 (2026-09-17)

Added
- `sentinel_v3`: the v0.2 composite with the profile regime read from the daily mean temperature and a rule for readings flagged above the expected value. Such a reading stays a fault only if at least two of the available cross-checks fail to confirm it as genuine: the profile (ratio to the expectation within 35%), the neighbours (the median normalised load of the BA's interchange partners is above its own recent 95th percentile at that hour) and the weather (the hour's temperature is in the BA's own seasonal tail for the heating or cooling regime). With two checks available the reading must fail both; with fewer, the v0.2 decision stands. Readings below the expectation, frozen runs and non-positive values are unchanged. Output adds `above_expected`, `check_profile`, `check_neighbors`, `check_weather`, and `confirmed_by = "extreme kept"` for alarms withdrawn by the rule.
- `crosscheck`: `confirm_neighbors`, `confirm_weather`, `preserve_extremes`.
- `neighbors`: neighbour table from the public EIA-930 interchange record (median absolute hourly interchange as weight; at least two partners with load data, else the BAs whose stations lie within 400 km). Published as `docs/neighbors.csv` (2024 interchange, 44 BAs, 194 interchange pairs, 3 BAs on the distance fallback) and `docs/neighbors_distance.csv` (distance only, for comparison). `scripts/make_neighbors.py` rebuilds them.
- `weather`: hourly ISD-Lite temperature per BA aligned to the load's local index, daily mean, seasonal 5th and 95th percentiles from the same calendar weeks of previous years.
- `stations`: the BA-to-airport map and distances between stations.
- `events`: the four winter events with an official under-forecast figure and the 72-hour window around each BA's peak.
- `scripts/storms.py`: for every event, BA and detector, the share of the window flagged, whether the peak reading survives, and the recall on faults injected into the same window; results in `results/storms/` and `docs/storms.md`.
- `data`: `load_series_local` and `utc_offset` in one module.
- Benchmark: a per-series context (temperature and neighbours) passed to every detector; variants `sentinel_v3_profile_neighbors`, `sentinel_v3_profile_weather`, `sentinel_v3_distance_neighbors`, `sentinel_v3_regime_only`.

Changed
- Detector callables in the benchmark take `(series, context)`.
- `requests` is a dependency (downloads of ISD-Lite and interchange files).

Regression on the v0.2 benchmark (20 BA-years, 0.5% injected faults, adjusted metrics): `sentinel_v3` F1 0.974 against 0.973 for `sentinel_v2`, adjusted precision 0.977 against 0.962, recall by type spike 0.95 / dip 0.71 / zero 1.00 / stuck 1.00 / scale 0.99 (v0.2: 0.95 / 0.70 / 1.00 / 1.00 / 1.00).

On the four winter events (`docs/storms.md`, 43 to 44 BAs each, 72 hours around each BA's peak, series as reported): the peak reading survives `sentinel_v3` in 95% of BAs in Uri, 95% in Elliott, 100% in Gerri and Heather and 100% in January 2025, against 95 / 88 / 91 / 100% for `sentinel_v2` and 86 / 58 / 77 / 88% for the modified z-score. Recall on faults injected into the same windows: 0.97 / 0.96 / 0.90 / 1.00 for `sentinel_v3`, 0.99 / 0.96 / 0.90 / 0.99 for `sentinel_v2`. Readings outside 1/3 to 3x of the expectation stay hard faults regardless of the cross-checks: letting them into the rule cost 11 points of recall on unit-error runs during storms. The profile cross-check band was set to 35%, the same as the v0.2 alarm band: with a wider band the rule confirmed injected spikes of up to 1.5x as genuine on cold hours and spike recall fell to 0.90. The neighbour percentile (95 or 98) makes no difference on this benchmark.

## 0.2.0 (2026-09-11)

Added
- `profile_residual` / `expected_profile`: calendar-aware expectation (day types with U.S. special days, optional regime label, references = last two weeks plus the same three weeks of the previous year, causal level from recent same-type days) and a residual detector scaled by the recent MAD.
- `sentinel_v2`: the v0.1 composite with each flag confirmed by the profile (residual beyond 5 robust standard deviations, or more than 35% from the expected value, on the reading or a neighbour); frozen runs, non-positive readings and readings outside 1/3 to 3× of the expected value are kept without the check. The expectation is built with the base-flagged readings masked out, so that a fault does not enter the reference set of the days that follow it.
- `intervals_from_mask` / `summarize_intervals`: flags grouped into intervals with a duration class.
- `repair(method="equivalent_days")`: straight line up to one hour; longer gaps filled with the mean of the same interval one to four weeks before and after, shifted to meet the neighbouring good readings; falls back to a line when no clean equivalent exists; the log now records the method.
- `modified_zscore` and `relative_deviation` baselines.
- `RecursiveTEDA(robust=True)`: winsorised update of the running statistics when a sample is flagged.
- `regime_from_peak_hour` and `regime_from_temperature` helpers; `day_types` calendar.
- `scripts/real_cases.py`: survey of zero hours, frozen runs, jumps and unit errors in every BA of the public record, and a gallery of named cases (`docs/real_cases.md`).
- Benchmark: adjusted precision and F1 that do not count flags on readings the source series already had wrong; `load_series_local` (local hour beginning) for profile-based detectors.

Chosen by grid (20 BA-years, 0.5% injected faults, adjusted metrics): base detectors without the winsorised update, profile threshold 5, band 35%. The winsorised base reaches recall 1.00 on every fault type at the cost of adjusted precision 0.80, and is kept as an option.

## 0.1.0 (2026-09-11)

First release: recursive TEDA (levels and differences), sparse autoencoder in NumPy, stuck-value rule, composite detector, labelled fault injection, audited linear repair, benchmark on EIA-930 with three reference detectors.
