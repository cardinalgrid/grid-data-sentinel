# API reference (v0.4)

The core depends on numpy only. Every detector follows one contract; batch and streaming are the same
object used through different methods.

## Inputs

| Argument | Type | Rules |
|---|---|---|
| `values` | float array, 1-d | Readings in the series' units (MW). NaN means missing. Lists, numpy arrays and pandas Series are accepted; a Series with a datetime index supplies `timestamps` on its own. |
| `timestamps` | `datetime64[ns]` array, 1-d | Local time, naive, one reading per hour on a regular grid. A gap or a duplicate raises `ValueError` naming the first offending position; `grid_sentinel.pandas.regularize` produces the grid. Detectors without a calendar (`TEDA`, `StuckValues`, the baselines, `SparseAutoencoder`) accept `values` alone. |
| `context` | `Context` | Optional auxiliary series, each a `(values, timestamps)` pair on its own grid: `temperature_f` (hourly, °F, at the BA's station), `neighbors` (name → hourly load of a neighbouring BA), `regime` (a daily label per hour, overriding the detector's rule). `Context.aligned(timestamps)` reindexes them onto the series grid. |

## Outputs

`DetectionResult` (one series) has arrays of the series' length: `timestamps`, `value`, `score`, `threshold`, `is_anomaly`, `reason`, and, where the detector has them, `expected` and `checks` (name → 1.0 confirmed / 0.0 not confirmed / NaN unavailable). Methods: `intervals()` (structured array `i0`, `i1`, `n`), `summary()`, `explain(i)`, `to_pandas()` (pandas extra). `Decision` is the same content for one reading, returned by `update`.

`reason` takes one of: `""` (not flagged), `teda_level`, `teda_diff`, `stuck`, `non_positive`, `gross_ratio`, `profile`, `baseline`, `dropped` (a base alarm the profile did not confirm; not flagged), `extreme_kept` (an alarm the cross-checks withdrew as a genuine extreme; not flagged).

## The contract

```python
class Detector:
    supports_streaming: bool
    def fit(values, timestamps=None, context=None) -> self     # learn from a history (streaming detectors replay it into their state)
    def predict(values, timestamps=None, context=None) -> DetectionResult
    def score(values, timestamps=None, context=None) -> ndarray
    def fit_predict(values, timestamps=None, context=None) -> DetectionResult
    def update(timestamp, value, **context) -> Decision          # one reading; NotImplementedError when not supports_streaming
    def reset() -> None
    def get_state() -> dict                                       # plain types and arrays; see state_to_json / state_from_json
    def set_state(state) -> self
    def get_params() -> dict; def set_params(**params) -> self   # parameters live only in the constructor
```

For every streaming detector, `predict` is exactly the sequence of `update` calls over the series (the batch result is the stream replayed), so a detector fed reading by reading reaches the same decisions as one given the whole series. The tests assert this for each class.

## Detectors

| Class | Streams | What it does | Reason codes |
|---|---|---|---|
| `TEDA(m=4.0, diff=False, robust=False, half_life_hours=None)` | yes | Recursive typicality-and-eccentricity detector on levels or first differences; optional winsorised update; optional exponential forgetting with the given half-life. | `teda_level`, `teda_diff` |
| `StuckValues(min_run=3)` | yes | A reading is flagged when it is the `min_run`-th or a later member of a run of identical values; a missing reading breaks the run. | `stuck` |
| `ProfileResidual(k=4.0, recent_weeks=2, analog_years=1, half_days=21, min_ref=2, level_days=5, mad_window_days=28, regime="load")` | yes | Expected value from the median shape of same-group days (day type × regime) in the last two weeks and around the same date a year earlier, scaled by the recent level; residual scaled by the recent MAD. `regime`: `"load"`, `"temperature"` (needs `temperature_f`) or `None`. `update(..., exclude=True)` keeps a reading out of the reference. | `profile` |
| `Sentinel(m_level=4.0, m_diff=3.0, min_run=3, k_profile=5.0, band=0.35, scale_ratio=3.0, regime="temperature", cross_checks=("profile", "neighbors", "weather"), extremes="preserve", profile_band=0.35, neighbor_pct=95.0, half_life_hours=None, robust=False)` | yes | The composite: base alarms confirmed against the profile, hard rules for frozen, non-positive and gross-ratio readings, and the two-of-three cross-checks that keep genuine extremes. `Sentinel(cross_checks=(), regime="load", extremes="off")` is the v0.2 composite; the defaults are v0.3. `update(timestamp, value, temperature_f=nan, neighbors=None)`. | `teda_level`, `teda_diff`, `stuck`, `gross_ratio`, `non_positive`, `dropped`, `extreme_kept` |
| `SparseAutoencoder(window=4, encoding_dim=2, l1=1e-4, epochs=30, batch_size=64, lr=1e-2, factor=20.0, trim=0.1, seed=0, causal=False)` | scoring | Window autoencoder trained by `fit`; the threshold is fixed at fit time. `causal=True` scores only the window ending at each reading, which is what `update` does. | `baseline` |
| `ModifiedZScore(window=720, k=3.5)` | yes | Iglewicz-Hoaglin modified z-score on the trailing window. | `baseline` |
| `RollingZScore(window=168, k=3.0)`, `Hampel(window=24, k=3.0)`, `IQR(k=1.5)`, `RelativeDeviation(window=5, k=0.15)` | no | Centred-window or global reference rules. | `baseline` |

Cross-check components (`grid_sentinel.detectors.crosscheck`): `NeighborCheck`, `WeatherCheck`, `preserve_one`. They are used by `Sentinel` and can be used on their own.

## Repair

```python
Repairer(method="equivalent_days", max_gap_hours=48, offsets_days=(7, 14, 21, 28), reason="flagged")
values_repaired, audit = Repairer().transform(values, timestamps, flags)
```

`method`: `"linear"`, `"equivalent_days"` (a line up to one hour; longer intervals take the mean of the same interval one to four weeks before and after, shifted to meet the neighbouring good readings; a line when no clean equivalent exists) or `"nan"`. Intervals longer than `max_gap_hours` are left open. `Audit` has `index`, `timestamp`, `original`, `repaired`, `method`, `reason` and `to_pandas()`. Detection never modifies the series; repair is this separate, explicit step.

`intervals_from_mask(mask)` returns the runs of a boolean mask with a duration class (`up to 1 h`, `up to 1 day`, `up to 1 week`, `up to 1 month`, `over 1 month`); `summarize_intervals(iv)` counts intervals and hours per class.

## State

`detector.get_state()` returns a dict of plain types and numpy arrays; `state_to_json` / `state_from_json` serialise it. `set_state` restores it on a detector built with the same parameters. A detector saved in the middle of a series and restored makes the same decisions as one run without interruption.

## pandas extra

`pip install grid-data-sentinel[pandas]` adds `grid_sentinel.pandas` (`from_series`, `to_series`, `regularize`, `context_from`, `predict_series`), the public-data loaders (`data`, `weather`, `neighbors`, `stations`, `events`), the benchmark and the scripts. `grid_sentinel.legacy` holds the v0.3 functions with a deprecation warning; see `migration.md`.

### Hour-label alignment (`grid_sentinel.alignment`)

A forecast and the load it refers to can be published on different hour conventions (one labelled by hour beginning, the other by hour ending). `alignment_table(load, forecast, shifts=(-2, -1, 0, 1, 2), freq="YS")` gives, per period, the MAPE of the forecast against the load for each candidate shift, the best shift, the gain in points over the published alignment and the hours compared; hours in which the two differ by more than half of the load are dropped as reporting faults. `alignment_check(load, forecast, min_gain=0.3)` returns the periods in which a moved forecast beats the published one by at least `min_gain` points (empty when the labels agree). `change_point(load, forecast, shift)` gives the first day from which the moved forecast is better on seven consecutive days. The check says that the two series disagree on the hour label, not which one is wrong. On EIA-930, PJM's day-ahead forecast is stamped one hour earlier than its demand from February 2019 on (MAPE 3.4 to 3.9 % as published, 2.3 to 2.5 % moved by one hour).
