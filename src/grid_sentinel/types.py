"""Input validation, context and result types shared by every detector.

Inputs are a one-dimensional float array of readings and, when the method needs a calendar, an array
of local-time timestamps on a regular hourly grid. Outputs are ``DetectionResult`` (a whole series)
or ``Decision`` (one reading in streaming). Nothing here imports pandas; ``to_pandas`` imports it lazily.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

HOUR = np.timedelta64(1, "h")

REASONS = (
    "",             # not flagged
    "teda_level",   # eccentricity of the level
    "teda_diff",    # eccentricity of the first difference
    "stuck",        # frozen telemetry value
    "non_positive", # zero or negative reading
    "gross_ratio",  # outside 1/scale_ratio .. scale_ratio of the expectation
    "profile",      # residual against the expected profile
    "baseline",     # one of the reference detectors
    "dropped",      # flagged by a base detector, not confirmed by the profile
    "extreme_kept", # flagged, then withdrawn by the cross-checks as a genuine extreme
)
REASON_DTYPE = "<U12"


def as_series(values: Any, timestamps: Any = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate ``values`` (float64, 1-d) and ``timestamps`` (datetime64[ns], naive, strictly hourly).

    A pandas-like object (anything with ``to_numpy`` and a datetime ``index``) is accepted; its index is
    used when ``timestamps`` is not given. ``timestamps`` may be omitted for detectors without a calendar.
    """
    if timestamps is None and hasattr(values, "index") and hasattr(values, "to_numpy"):
        idx = np.asarray(values.index)
        if np.issubdtype(idx.dtype, np.datetime64):
            timestamps = idx
    if hasattr(values, "to_numpy"):
        values = values.to_numpy(dtype=float)
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {v.shape}")
    if timestamps is None:
        return v, None
    t = np.asarray(timestamps).astype("datetime64[ns]")
    if t.ndim != 1 or len(t) != len(v):
        raise ValueError(f"timestamps length {len(t)} does not match values length {len(v)}")
    if len(t) > 1:
        steps = np.diff(t)
        bad = np.flatnonzero(steps != HOUR)
        if len(bad):
            i = int(bad[0]) + 1
            raise ValueError(
                f"timestamps must be a regular hourly grid; step before position {i} is {steps[bad[0]]} "
                "(use grid_sentinel.pandas.regularize to fill or reindex)"
            )
    return v, t


def _align(pair: tuple[Any, Any], grid: np.ndarray) -> np.ndarray:
    """Values of ``pair`` = (values, timestamps) reindexed onto ``grid``; NaN where the grid has no match."""
    vals = np.asarray(pair[0], dtype=np.float64)
    ts = np.asarray(pair[1]).astype("datetime64[ns]")
    out = np.full(len(grid), np.nan)
    pos = np.searchsorted(ts, grid)
    pos = np.clip(pos, 0, max(len(ts) - 1, 0))
    if len(ts):
        hit = ts[pos] == grid
        out[hit] = vals[pos[hit]]
    return out


@dataclass(frozen=True)
class AlignedContext:
    temperature_f: np.ndarray | None
    neighbors: dict[str, np.ndarray]
    regime: np.ndarray | None

    def at(self, i: int) -> dict[str, Any]:
        """Scalar context for one position, in the keyword form ``Detector.update`` takes."""
        out: dict[str, Any] = {}
        if self.temperature_f is not None:
            out["temperature_f"] = float(self.temperature_f[i])
        if self.neighbors:
            out["neighbors"] = {k: float(v[i]) for k, v in self.neighbors.items()}
        if self.regime is not None:
            out["regime"] = float(self.regime[i])
        return out


@dataclass(frozen=True)
class Context:
    """Auxiliary series a detector may use: each field is a (values, timestamps) pair on its own grid."""

    temperature_f: tuple[Any, Any] | None = None
    neighbors: dict[str, tuple[Any, Any]] | None = None
    regime: tuple[Any, Any] | None = None

    def aligned(self, timestamps: np.ndarray) -> AlignedContext:
        grid = np.asarray(timestamps).astype("datetime64[ns]")
        return AlignedContext(
            temperature_f=_align(self.temperature_f, grid) if self.temperature_f is not None else None,
            neighbors={k: _align(v, grid) for k, v in (self.neighbors or {}).items()},
            regime=_align(self.regime, grid) if self.regime is not None else None,
        )


def intervals_from_mask(mask: np.ndarray) -> np.ndarray:
    """Runs of True in ``mask`` as a structured array with fields ``i0``, ``i1`` (inclusive) and ``n``."""
    m = np.asarray(mask, dtype=bool)
    dtype = np.dtype([("i0", "int64"), ("i1", "int64"), ("n", "int64")])
    if not m.any():
        return np.empty(0, dtype=dtype)
    padded = np.concatenate([[False], m, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[::2], edges[1::2] - 1
    return np.array(list(zip(starts, ends, ends - starts + 1)), dtype=dtype)


@dataclass
class DetectionResult:
    """Per-reading output of a detector over a whole series. Arrays share the series' length."""

    timestamps: np.ndarray | None
    value: np.ndarray
    score: np.ndarray
    threshold: np.ndarray
    is_anomaly: np.ndarray
    reason: np.ndarray
    expected: np.ndarray | None = None
    checks: dict[str, np.ndarray] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.value)

    def intervals(self) -> np.ndarray:
        return intervals_from_mask(self.is_anomaly)

    def summary(self) -> dict[str, Any]:
        reasons, counts = np.unique(self.reason[self.is_anomaly], return_counts=True)
        return {
            "n": len(self.value),
            "flagged": int(self.is_anomaly.sum()),
            "intervals": len(self.intervals()),
            "by_reason": {str(r): int(c) for r, c in zip(reasons, counts)},
        }

    def explain(self, i: int) -> dict[str, Any]:
        out: dict[str, Any] = {
            "timestamp": None if self.timestamps is None else self.timestamps[i],
            "value": float(self.value[i]),
            "score": float(self.score[i]),
            "threshold": float(self.threshold[i]),
            "is_anomaly": bool(self.is_anomaly[i]),
            "reason": str(self.reason[i]),
        }
        if self.expected is not None:
            out["expected"] = float(self.expected[i])
        for k, v in self.checks.items():
            out[f"check_{k}"] = float(v[i])
        return out

    def to_pandas(self):
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover
            raise ImportError("DetectionResult.to_pandas needs pandas: pip install grid-data-sentinel[pandas]") from e
        cols: dict[str, Any] = {
            "value": self.value, "score": self.score, "threshold": self.threshold,
            "is_anomaly": self.is_anomaly, "reason": self.reason,
        }
        if self.expected is not None:
            cols["expected"] = self.expected
        for k, v in self.checks.items():
            cols[f"check_{k}"] = v
        index = pd.DatetimeIndex(self.timestamps) if self.timestamps is not None else pd.RangeIndex(len(self.value))
        return pd.DataFrame(cols, index=index)


@dataclass(frozen=True)
class Decision:
    """Output of one streaming ``update``: the same content as one row of ``DetectionResult``."""

    timestamp: Any
    value: float
    score: float
    threshold: float
    is_anomaly: bool
    reason: str
    expected: float
    checks: dict[str, float]


def empty_decision(timestamp: Any, value: float) -> Decision:
    return Decision(timestamp, float(value), 0.0, float("inf"), False, "", float("nan"), {})
