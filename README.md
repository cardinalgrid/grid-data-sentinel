# Grid Data Sentinel

**Anomaly detection and repair for load telemetry, benchmarked on public data.**

During the January 2024 Arctic storms, FERC found operators flagging genuinely high load readings as "bad data" ([FERC 2024](https://www.ferc.gov/news-events/news/presentation-system-performance-review-january-2024-arctic-storms)). NERC lists data validation, anomaly detection and provenance as preconditions for trustworthy operator-facing AI ([NERC 2024](https://www.nerc.com/pa/rrm/bpsa/Documents/Whitepaper-AI%20and%20ML%20in%20Real-Time%20System%20Operations.pdf)). This package is a small, dependency-light toolkit for that problem: detect corrupted readings in a load series, repair them with an audit trail, and measure how well each method does on public EIA-930 data with known, injected faults.

> Status: **v0.1.0** (batch mode). Streaming API, extreme-preserving cross-checks and PyPI packaging are on the roadmap below.

## Methods

| Detector | What it does | Origin |
|---|---|---|
| `RecursiveTEDA` | Streaming detector based on typicality and eccentricity data analytics: keeps a recursive mean and variance and flags a reading when its normalised eccentricity exceeds (m²+1)/2k. No window, no training, no distributional assumption. Runs on levels or on first differences. | Angelov (2014); the maintainer's M.Sc. work on outlier detection in demand curves |
| `SparseAutoencoder` | Window autoencoder (default 4 readings, code of size 2, L1 activity penalty) trained to reconstruct the series; the reconstruction error is the anomaly score. Implemented in NumPy, so there is no deep-learning dependency. | Guerra Filho et al., [*Energies* 2024, 17(24), 6403](https://doi.org/10.3390/en17246403) |
| `stuck_values` | Rule: a run of identical consecutive readings is a frozen telemetry value, not a measurement. | Operational practice |
| `sentinel` | Composite v0.1 detector: TEDA on levels ∪ TEDA on differences ∪ stuck-value rule. | This package |
| `rolling_zscore`, `hampel`, `iqr` | Reference detectors. | Standard |

Plus `inject_anomalies` (labelled spikes, dips, zeros, stuck runs and unit errors for benchmarking), `repair` (interpolation with a per-reading audit log; gaps longer than 48 readings are left open rather than invented) and `score_labels` (precision, recall, F1, MCC, with an optional ±n tolerance).

## Benchmark on EIA-930

Ten large balancing authorities, hourly demand for 2023 and 2024 from the EIA-930 balance files (via [ba-forecast-scorecard](https://github.com/cardinalgrid/ba-forecast-scorecard)), 0.5% of readings corrupted with labelled faults, every detector run on the corrupted series with its default settings.

<!-- results:start -->
Benchmark v0.1.0 run 2026-09-11: 20 BA-years (PJM, MISO, ERCO, CISO, SWPP, NYIS, ISNE, TVA, DUK, BPAT; 2023, 2024), 0.5% of readings corrupted, tolerance ±1 reading. Means over series.

| Detector | Precision | Recall | F1 | MCC | Spike | Dip | Zero | Stuck | Scale | s/series |
|---|---|---|---|---|---|---|---|---|---|---|
| Sparse autoencoder (window 4, code 2, factor 20) | 0.89 | 0.93 | 0.91 | 0.91 | 0.90 | 0.88 | 0.90 | 0.04 | 0.98 | 0.3 |
| Recursive TEDA (levels, m=3) | 0.97 | 0.82 | 0.89 | 0.89 | 0.60 | 0.21 | 1.00 | 0.00 | 0.92 | 0.0 |
| sentinel | 0.87 | 0.92 | 0.87 | 0.88 | 1.00 | 0.64 | 1.00 | 1.00 | 0.94 | 0.1 |
| Recursive TEDA (differenced, m=3) | 0.72 | 0.20 | 0.29 | 0.36 | 1.00 | 0.64 | 1.00 | 0.00 | 0.20 | 0.0 |
| Global IQR (k=1.5) | 0.17 | 0.93 | 0.27 | 0.38 | 0.60 | 0.80 | 1.00 | 0.10 | 1.00 | 0.0 |
| Rolling z-score (168 h, k=3) | 0.69 | 0.10 | 0.17 | 0.25 | 0.95 | 0.80 | 1.00 | 0.00 | 0.00 | 0.0 |
| Hampel filter (24 h, k=3) | 0.07 | 0.20 | 0.10 | 0.11 | 0.95 | 0.72 | 1.00 | 0.06 | 0.15 | 0.0 |
| stuck_rule | 0.46 | 0.06 | 0.10 | 0.16 | 0.00 | 0.00 | 0.00 | 1.00 | 0.01 | 0.0 |

Columns Spike to Scale are recall by anomaly type. Full table: `results/benchmark_by_series.csv`.
<!-- results:end -->

What the table says:

- **The composite `sentinel` is the only detector that catches every fault type.** It finds all spikes, zeros and stuck runs and 94% of unit-error runs, at 0.87 precision. Dips are its weak spot (64%), because a drop of 30 to 90% in one hour is within what the level detector has seen on a large system.
- **The autoencoder has the best F1 but misses stuck values**, which no reconstruction method sees: a frozen value is a perfectly reconstructable reading. The rule catches them all.
- **Reference detectors fail in the expected ways.** The global IQR rule flags every legitimate winter and summer peak (precision 0.17); the Hampel filter and rolling z-score are blind to unit-error runs longer than their window.
- **The stuck rule's precision of 0.46 is a finding about the data, not the rule:** the raw EIA-930 series already contain runs of identical hourly values that nobody labelled. They are counted as false positives here. Note 6 in the [Cardinal Grid notes](https://github.com/cardinalgrid/notes) will look at them.

Defaults for the two learned detectors were chosen from a sensitivity grid on four BAs for 2023 (`scripts/tune.py`, output in `results/tuning.csv`). The autoencoder threshold factor in particular is not portable across data: 3× the trimmed-mean error was right for 15-minute substation data in the 2024 paper; 20× is right for hourly BA data, where the error distribution has a much heavier tail.

## Install and use

```bash
pip install git+https://github.com/cardinalgrid/grid-data-sentinel
```

```python
import pandas as pd
from grid_sentinel import sentinel, repair

demand = pd.read_csv("my_series.csv")["demand"]
flags = sentinel(demand)                       # value, score, threshold, is_anomaly
fixed, audit = repair(demand, flags["is_anomaly"], reason="sentinel v0.1")
```

Command line:

```bash
grid-sentinel detect my_series.csv --column demand --method sentinel --repair --out detections.csv
grid-sentinel benchmark --tidy path/to/ba-forecast-scorecard/data/tidy --out results
python scripts/render_readme.py    # refresh the table above from results/summary.json
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

- The benchmark measures sensitivity to *injected* faults. Faults already present in the source series are unlabelled and count against precision.
- Injected faults are simple by design (single-reading spikes and dips, zero runs, frozen values, ×10 and ×0.1 runs). Subtler failures such as a slow drift or a partial feeder loss are not modelled yet.
- `RecursiveTEDA` keeps all history. On a decade of data the running variance is dominated by the seasonal cycle, and sensitivity to single spikes falls. A forgetting factor is planned for v0.2.
- No detector here uses weather, neighbouring series or the day-ahead forecast. The extreme-preserving cross-checks that would distinguish a real record peak from a bad reading are v0.2 work.
- One series, one BA, one detector at a time. No claims are made about any operator's internal data quality.

## Roadmap

| Version | Target | Scope |
|---|---|---|
| v0.1 | September 2026 | TEDA + autoencoder + rules, batch mode, benchmark on EIA-930 (this release) |
| v0.2 | November 2026 | Streaming API with forgetting, extreme-preserving cross-checks against weather and neighbours, evaluation on the January 2024 Arctic-storm period |
| v1.0 | December 2026 | Audit trail format, documentation, examples on public data, PyPI |

## Development

```bash
pip install -e ".[dev]"
ruff check src tests scripts && pytest -q
```

## Citing

See `CITATION.cff`. Each release is archived on Zenodo with a DOI.

## License

Apache-2.0 for code; text and figures CC BY 4.0. Analyses rely on public data only and represent the maintainer's own views.
