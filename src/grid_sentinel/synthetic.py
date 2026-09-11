"""Injection of labelled anomalies into a clean series, for benchmarking.

Five anomaly types are modelled on what is seen in operational load telemetry:

- ``spike``  a single reading multiplied by (1 + a), a in [0.3, 1.0]
- ``dip``    a single reading multiplied by (1 - a), a in [0.3, 0.9]
- ``zero``   a run of 1 to 3 readings set to zero (telemetry loss reported as zero)
- ``stuck``  a run of 3 to 12 readings frozen at the last good value (stale SCADA value)
- ``scale``  a run of 6 to 48 readings multiplied by 10 or 0.1 (unit error)

Every injected reading is labelled 1. Readings that were already wrong in the source data are not
labelled, so a detector that flags them is counted as a false positive: the benchmark measures
sensitivity to the injected faults, not the absolute quality of the source series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TYPES = ("spike", "dip", "zero", "stuck", "scale")


def inject_anomalies(
    series: pd.Series,
    rate: float = 0.005,
    types: tuple[str, ...] = TYPES,
    seed: int = 0,
) -> pd.DataFrame:
    """Return a frame with ``clean``, ``value`` (corrupted), ``label`` (0/1) and ``kind``.

    ``rate`` is the target share of readings corrupted; runs count each reading.
    """
    rng = np.random.default_rng(seed)
    clean = series.astype(float).to_numpy()
    values = clean.copy()
    label = np.zeros(len(values), dtype=np.int8)
    kind = np.array([""] * len(values), dtype=object)
    target = int(rate * len(values))
    n_done = 0
    guard = 0
    while n_done < target and guard < 50 * target + 100:
        guard += 1
        t = types[rng.integers(len(types))]
        if t in ("spike", "dip"):
            length = 1
        elif t == "zero":
            length = int(rng.integers(1, 4))
        elif t == "stuck":
            length = int(rng.integers(3, 13))
        else:
            length = int(rng.integers(6, 49))
        start = int(rng.integers(1, len(values) - length))
        seg = slice(start, start + length)
        if label[seg].any() or np.isnan(clean[seg]).any():
            continue
        if t == "spike":
            values[seg] = clean[seg] * (1.0 + rng.uniform(0.3, 1.0))
        elif t == "dip":
            values[seg] = clean[seg] * (1.0 - rng.uniform(0.3, 0.9))
        elif t == "zero":
            values[seg] = 0.0
        elif t == "stuck":
            values[seg] = clean[start - 1]
        else:
            values[seg] = clean[seg] * (10.0 if rng.random() < 0.5 else 0.1)
        label[seg] = 1
        kind[seg] = t
        n_done += length
    return pd.DataFrame({"clean": clean, "value": values, "label": label, "kind": kind}, index=series.index)
