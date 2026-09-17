# Detectors on the four winter events

Generated 2026-09-17 by `scripts/storms.py`, grid-data-sentinel 0.3.0. For each event and each BA with a station, the 72 hours around the BA's peak. `flagged_share` and `peak_kept` come from the series as reported: the share of the window a detector marks as fault and the share of BAs whose peak reading survives. `recall_in_window` and `precision_adj_in_window` come from a second run with faults injected into the same window at a rate of 2% (one random draw per BA); the adjusted precision ignores readings the source already had wrong.

## Winter Storm Uri (2021-02-13 to 2021-02-18, 44 BAs)

| Detector | flagged_share | peak_kept | recall_in_window | precision_adj_in_window |
|---|---|---|---|---|
| TEDA | 0.001 | 0.95 | 0.68 | 0.69 |
| v0.3 dist. | 0.002 | 0.98 | 0.97 | 0.95 |
| v0.3 P+N | 0.002 | 0.98 | 0.97 | 0.95 |
| v0.3 P+W | 0.002 | 0.98 | 0.97 | 0.95 |
| v0.2 | 0.002 | 0.95 | 0.99 | 0.97 |
| v0.3 | 0.002 | 0.95 | 0.97 | 0.95 |
| v0.3 regime | 0.003 | 0.95 | 0.99 | 0.97 |
| rel. dev. | 0.004 | 0.98 | 0.64 | 0.58 |
| AE | 0.020 | 0.91 | 0.71 | 0.66 |
| mod. z | 0.032 | 0.86 | 0.70 | 0.61 |

BAs: AEC, AECI, AVA, AZPS, BANC, BPAT, CISO, CPLE, CPLW, DUK, EPE, ERCO, FMPP, FPC, FPL, GCPD, IPCO, ISNE, JEA, LDWP, LGEE, MISO, NEVP, NWMT, NYIS, PACE, PACW, PGE, PJM, PNM, PSCO, PSEI, SC, SCEG, SCL, SOCO, SRP, SWPP, TEC, TEPC, TPWR, TVA, WACM, WALC

## Winter Storm Elliott (2022-12-22 to 2022-12-26, 43 BAs)

| Detector | flagged_share | peak_kept | recall_in_window | precision_adj_in_window |
|---|---|---|---|---|
| v0.3 P+W | 0.001 | 0.95 | 0.88 | 0.90 |
| v0.3 dist. | 0.001 | 0.95 | 0.95 | 0.97 |
| v0.3 | 0.002 | 0.95 | 0.95 | 0.95 |
| rel. dev. | 0.003 | 0.95 | 0.67 | 0.58 |
| v0.3 P+N | 0.006 | 0.93 | 0.91 | 0.91 |
| v0.2 | 0.015 | 0.88 | 0.95 | 0.94 |
| v0.3 regime | 0.015 | 0.88 | 0.95 | 0.94 |
| TEDA | 0.017 | 0.88 | 0.47 | 0.43 |
| AE | 0.061 | 0.65 | 0.75 | 0.59 |
| mod. z | 0.164 | 0.58 | 0.74 | 0.52 |

BAs: AECI, AVA, AZPS, BANC, BPAT, CISO, CPLE, CPLW, DUK, EPE, ERCO, FMPP, FPC, FPL, GCPD, IPCO, ISNE, JEA, LDWP, LGEE, MISO, NEVP, NWMT, NYIS, PACE, PACW, PGE, PJM, PNM, PSCO, PSEI, SC, SCEG, SCL, SOCO, SRP, SWPP, TEC, TEPC, TPWR, TVA, WACM, WALC

## Winter Storms Gerri and Heather (2024-01-12 to 2024-01-17, 43 BAs)

| Detector | flagged_share | peak_kept | recall_in_window | precision_adj_in_window |
|---|---|---|---|---|
| rel. dev. | 0.000 | 1.00 | 0.65 | 0.61 |
| v0.3 P+W | 0.002 | 1.00 | 0.90 | 0.89 |
| v0.3 dist. | 0.003 | 1.00 | 0.90 | 0.89 |
| v0.3 | 0.003 | 1.00 | 0.90 | 0.88 |
| v0.3 P+N | 0.004 | 0.98 | 0.88 | 0.88 |
| TEDA | 0.007 | 0.88 | 0.57 | 0.60 |
| v0.3 regime | 0.008 | 0.93 | 0.92 | 0.89 |
| v0.2 | 0.010 | 0.91 | 0.90 | 0.88 |
| AE | 0.027 | 0.74 | 0.76 | 0.62 |
| mod. z | 0.034 | 0.77 | 0.34 | 0.21 |

BAs: AECI, AVA, AZPS, BANC, BPAT, CISO, CPLE, CPLW, DUK, EPE, ERCO, FMPP, FPC, FPL, GCPD, IPCO, ISNE, JEA, LDWP, LGEE, MISO, NEVP, NWMT, NYIS, PACE, PACW, PGE, PJM, PNM, PSCO, PSEI, SC, SCEG, SCL, SOCO, SRP, SWPP, TEC, TEPC, TPWR, TVA, WACM, WALC

## January 2025 Arctic events (2025-01-19 to 2025-01-24, 43 BAs)

| Detector | flagged_share | peak_kept | recall_in_window | precision_adj_in_window |
|---|---|---|---|---|
| TEDA | 0.000 | 1.00 | 0.67 | 0.72 |
| v0.3 | 0.001 | 1.00 | 1.00 | 1.00 |
| v0.3 dist. | 0.001 | 1.00 | 0.98 | 0.98 |
| v0.3 P+N | 0.001 | 1.00 | 0.97 | 0.97 |
| v0.3 P+W | 0.001 | 1.00 | 0.93 | 0.93 |
| v0.3 regime | 0.001 | 1.00 | 1.00 | 1.00 |
| rel. dev. | 0.002 | 0.98 | 0.63 | 0.56 |
| v0.2 | 0.002 | 1.00 | 0.99 | 0.99 |
| mod. z | 0.009 | 0.88 | 0.74 | 0.71 |
| AE | 0.027 | 0.72 | 0.66 | 0.58 |

BAs: AECI, AVA, AZPS, BANC, BPAT, CISO, CPLE, CPLW, DUK, EPE, ERCO, FMPP, FPC, FPL, GCPD, IPCO, ISNE, JEA, LDWP, LGEE, MISO, NEVP, NWMT, NYIS, PACE, PACW, PGE, PJM, PNM, PSCO, PSEI, SC, SCEG, SCL, SOCO, SRP, SWPP, TEC, TEPC, TPWR, TVA, WACM, WALC

## Limitations

One NOAA station per BA; neighbours from the aggregate 2024 interchange record, not per tie line; a 72-hour window around each BA's peak, chosen here; events are the four with an official under-forecast figure, not a random sample; injected faults follow the benchmark protocol; nothing here describes any operator's internal data.
