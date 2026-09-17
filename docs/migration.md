# Migrating from 0.3 to 0.4

0.4 replaces the function-based API with estimator classes on one contract, makes the core independent
of pandas, and runs the whole composite in streaming. Every 0.3 name still works for one minor version
through `grid_sentinel.legacy` (and as `grid_sentinel.<name>`), with a `DeprecationWarning` and the
exact 0.3 behaviour; they are removed in 1.0.

## Name by name

| 0.3 | 0.4 |
|---|---|
| `RecursiveTEDA(m, diff, robust).detect(series)` | `TEDA(m, diff, robust).predict(values)` (also `half_life_hours`) |
| `stuck_values(series, min_run)` | `StuckValues(min_run).predict(values)` |
| `profile_residual(series, regime, k)` | `ProfileResidual(k, regime="load"/"temperature"/None).predict(values, timestamps, context)` |
| `expected_profile(series, ...)` | the `expected` field of `ProfileResidual(...).predict(...)` |
| `regime_from_peak_hour`, `regime_from_temperature` | `ProfileResidual(regime="load")`, `ProfileResidual(regime="temperature")` |
| `sentinel(series)` | no equivalent without the profile; `legacy.sentinel` for the v0.1 composite |
| `sentinel_v2(series, regime=...)` | `Sentinel(cross_checks=(), regime="load", extremes="off").predict(values, timestamps)` |
| `sentinel_v3(series, temperature_f, neighbors)` | `Sentinel().predict(values, timestamps, Context(temperature_f=(temp, t), neighbors={...}))` |
| `rolling_zscore`, `hampel`, `iqr`, `modified_zscore`, `relative_deviation` | `RollingZScore`, `Hampel`, `IQR`, `ModifiedZScore`, `RelativeDeviation` |
| `SparseAutoencoder(...).detect(series, fit=True)` | `SparseAutoencoder(...).fit_predict(values)`; `fit` then `predict` for new data |
| `repair(series, flags, method, max_gap)` | `Repairer(method, max_gap_hours).transform(values, timestamps, flags)` → `(values, Audit)` |
| `intervals_from_mask(mask)` (DataFrame) | `intervals_from_mask(mask)` (structured array) |
| `summarize_intervals(iv)` (DataFrame) | `summarize_intervals(iv)` (dict) |
| `confirm_neighbors`, `confirm_weather`, `preserve_extremes` | `detectors.crosscheck.NeighborCheck`, `WeatherCheck`, `preserve_one` |

## Inputs and outputs

- The core takes numpy: `values` (float) and `timestamps` (`datetime64[ns]`, local hour beginning, regular hourly grid). From a pandas Series: `from grid_sentinel.pandas import from_series, regularize` then `v, t = from_series(regularize(s))`, or `predict_series(detector, s)` for the whole round trip.
- Results are `DetectionResult` objects, not DataFrames; `result.to_pandas()` gives the old shape (`value`, `score`, `threshold`, `is_anomaly`, `reason`, `expected`, `check_*`). The 0.3 column `confirmed_by` is now `reason`, with fixed codes (see `api.md`).
- Context (temperature, neighbours) is a `Context` object instead of keyword arguments.

## Behaviour that changed

These are the differences between `Sentinel()` and `legacy.sentinel_v3`, all consequences of deciding each reading when it arrives:

1. **Frozen values.** A run of identical readings is flagged from its `min_run`-th member on (0.3 flagged from the second member once the run reached `min_run`). With the benchmark's tolerance of one reading the effect is at most one reading per run.
2. **No look-ahead in the profile confirmation.** 0.3 kept a base alarm if the reading, the previous one or the next one was implausible against the profile; 0.4 looks at the reading and the previous one.
3. **Missing readings** come out unflagged with reason `""`; 0.3 flagged NaN as non-positive.
4. **Weather regime for the cross-check** is yesterday's mean temperature (0.3 used the same day's, which a streaming detector cannot know); the profile's regime was already yesterday's.
5. **Autoencoder streaming** scores the window ending at the reading (`causal=True` makes batch do the same); the default batch scoring is unchanged.

On PJM 2024 the causal composite flags nothing that 0.3 did not, and 0.3 flags at most 0.5% of readings more, all explained by items 1 and 2. On the 20 BA-year benchmark and on the four winter events the numbers are within the tolerances recorded in the changelog.

## Installation

`pip install grid-data-sentinel` now installs numpy only. `pip install grid-data-sentinel[pandas]` adds pandas, pyarrow and requests, needed by the pandas adapter, the public-data loaders, the benchmark, the scripts and `grid_sentinel.legacy`.
