# Changelog

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
