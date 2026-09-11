"""Deterministic rules for failure modes that statistical detectors miss, and a composite detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from grid_sentinel.teda import RecursiveTEDA


def _frame(series: pd.Series | np.ndarray, score: np.ndarray, thr: float, flags: np.ndarray) -> pd.DataFrame:
    index = series.index if isinstance(series, pd.Series) else pd.RangeIndex(len(score))
    return pd.DataFrame(
        {"value": np.asarray(series, dtype=float), "score": score, "threshold": thr, "is_anomaly": flags},
        index=index,
    )


def stuck_values(series: pd.Series | np.ndarray, min_run: int = 3) -> pd.DataFrame:
    """Flag runs of ``min_run`` or more identical consecutive readings (a frozen telemetry value).

    Load never repeats to the last megawatt for hours; a run of identical values is a stale point,
    not a measurement. The score is the run length; the first reading of the run is kept, since it
    is usually the last good value.
    """
    v = np.asarray(series, dtype=float)
    same = np.zeros(len(v), dtype=bool)
    same[1:] = (v[1:] == v[:-1]) & ~np.isnan(v[1:])
    run = np.zeros(len(v), dtype=int)
    for i in range(1, len(v)):
        run[i] = run[i - 1] + 1 if same[i] else 0
    # propagate the final run length back over the whole run
    length = run.copy()
    for i in range(len(v) - 2, -1, -1):
        if same[i + 1]:
            length[i] = length[i + 1]
    flags = same & (length >= min_run - 1)
    return _frame(series, length.astype(float), float(min_run), flags)


def sentinel(
    series: pd.Series | np.ndarray,
    m_level: float = 4.0,
    m_diff: float = 3.0,
    min_run: int = 3,
) -> pd.DataFrame:
    """Composite v0.1 detector: recursive TEDA on levels, recursive TEDA on differences, stuck-value rule.

    A reading is flagged if any component flags it. The score is the largest of the three
    component scores, each expressed as a ratio to its own threshold.
    """
    a = RecursiveTEDA(m=m_level, diff=False).detect(series)
    b = RecursiveTEDA(m=m_diff, diff=True).detect(series)
    c = stuck_values(series, min_run=min_run)
    score = np.nanmax(
        np.vstack([a["score"].to_numpy(), b["score"].to_numpy(), c["score"].to_numpy() / c["threshold"].to_numpy()]),
        axis=0,
    )
    flags = a["is_anomaly"].to_numpy() | b["is_anomaly"].to_numpy() | c["is_anomaly"].to_numpy()
    return _frame(series, score, 1.0, flags)
