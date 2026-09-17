# Changelog

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
