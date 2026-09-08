# Grid Data Sentinel

**Anomaly detection and correction for load telemetry that keeps the genuine extremes.**

During the January 2024 Arctic storms, FERC found operators flagging genuinely high load readings as "bad data" ([FERC 2024](https://www.ferc.gov/news-events/news/presentation-system-performance-review-january-2024-arctic-storms)). NERC lists data validation, anomaly detection and provenance as preconditions for trustworthy operator-facing AI ([NERC 2024](https://www.nerc.com/pa/rrm/bpsa/Documents/Whitepaper-AI%20and%20ML%20in%20Real-Time%20System%20Operations.pdf)). This package addresses both: detect and repair corrupted readings, preserve real extreme events, and log every change.

> Status: **v0.1 planned for October 2026**.

## Methods

- **Recursive TEDA** (typicality and eccentricity data analytics): streaming, distribution-free anomaly scores; from the maintainer's M.Sc. thesis on outlier detection in energy demand curves.
- **Sparse autoencoders** for detection and correction, published in [Guerra Filho et al., *Energies* 2024, 17(24), 6403](https://www.mdpi.com/1996-1073/17/24/6403).
- **Extreme-preserving rules**: cross-checks against weather and neighbouring series before any correction.
- **Audit trail**: every correction recorded with a reason code.

## Validation plan

- EIA-930 series with documented and synthetic anomalies across 65 balancing authorities.
- ERCOT and PJM zonal load, including the January 2024 Arctic storms period.
- Baselines: z-score, IQR, Hampel filter, isolation forest, one published method.
- Metrics: precision, recall, F1, and the downstream effect on day-ahead forecast error.
- Results are published with their limitations, whatever they show.

## Roadmap

| Version | Target | Scope |
|---|---|---|
| v0.1 | October 2026 | TEDA + autoencoder, batch mode, first benchmark on EIA-930 |
| v0.2 | November 2026 | Streaming mode, extreme-preserving rules |
| v1.0 | December 2026 | Audit trail, documentation, examples on public data, PyPI |

## Install

```
pip install grid-data-sentinel   # available with v0.1
```

## Citing

Each release is archived on Zenodo with a DOI; the citation record will be listed here with v0.1.

## License

Apache-2.0.
