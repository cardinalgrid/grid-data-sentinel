"""Adapter between pandas objects and the numpy contract of the core (the ``[pandas]`` extra)."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise ImportError("grid_sentinel.pandas needs pandas: pip install grid-data-sentinel[pandas]") from e

from grid_sentinel.types import Context, DetectionResult


def from_series(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(values, timestamps) of an hourly Series with a naive local DatetimeIndex."""
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return s.to_numpy(dtype=float), idx.to_numpy().astype("datetime64[ns]")


def to_series(values: np.ndarray, timestamps: np.ndarray, name: str | None = None) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(timestamps), name=name)


def regularize(s: pd.Series) -> pd.Series:
    """The Series on a complete hourly grid from its first to its last timestamp, duplicates dropped
    (first kept), missing hours as NaN. This is what the core's validation asks for."""
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s = pd.Series(s.to_numpy(dtype=float), index=idx)
    s = s[~s.index.duplicated(keep="first")].sort_index()
    grid = pd.date_range(s.index.min(), s.index.max(), freq="h")
    return s.reindex(grid)


def context_from(temperature_f: pd.Series | None = None, neighbors: dict[str, pd.Series] | None = None,
                 regime: pd.Series | None = None) -> Context:
    """A ``Context`` from pandas Series (each on its own DatetimeIndex)."""

    def pair(x: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        return from_series(x)

    return Context(
        temperature_f=pair(temperature_f) if temperature_f is not None else None,
        neighbors={k: pair(v) for k, v in neighbors.items()} if neighbors else None,
        regime=pair(regime) if regime is not None else None,
    )


def result_to_frame(result: DetectionResult) -> pd.DataFrame:
    return result.to_pandas()


def predict_series(detector: Any, s: pd.Series, context: Context | None = None) -> pd.DataFrame:
    """Run a detector on a pandas Series (regularised first) and return the result as a DataFrame."""
    v, t = from_series(regularize(s))
    return detector.predict(v, t, context).to_pandas()
