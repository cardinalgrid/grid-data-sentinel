"""Rolling statistics in numpy with the semantics of pandas' ``Series.rolling``.

``center=False`` uses the window ending at each position; ``center=True`` uses the window
``[i - window//2, i + (window - 1)//2]``. A position gets NaN when fewer than ``min_periods``
finite readings fall inside its window. ``trailing_quantile`` uses linear interpolation, like pandas.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _windows(x: np.ndarray, window: int, center: bool) -> np.ndarray:
    """(n, window) view with NaN padding so that row i is the window of position i."""
    x = np.asarray(x, dtype=np.float64)
    if center:
        before = window // 2
        after = window - 1 - before
    else:
        before, after = window - 1, 0
    padded = np.concatenate([np.full(before, np.nan), x, np.full(after, np.nan)])
    return sliding_window_view(padded, window)


def _apply(x: np.ndarray, window: int, min_periods: int, center: bool, fn) -> np.ndarray:
    w = _windows(x, window, center)
    count = np.isfinite(w).sum(axis=1)
    with np.errstate(all="ignore"):
        out = fn(w)
    out = np.where(count >= max(min_periods, 1), out, np.nan)
    return out


def rolling_mean(x, window: int, min_periods: int = 1, center: bool = False) -> np.ndarray:
    return _apply(x, window, min_periods, center, lambda w: np.nanmean(w, axis=1))


def rolling_median(x, window: int, min_periods: int = 1, center: bool = False) -> np.ndarray:
    return _apply(x, window, min_periods, center, lambda w: np.nanmedian(w, axis=1))


def rolling_std(x, window: int, min_periods: int = 1, center: bool = False) -> np.ndarray:
    return _apply(x, window, min_periods, center, lambda w: np.nanstd(w, axis=1, ddof=1))


def trailing_quantile(x, window: int, q: float, min_periods: int = 1) -> np.ndarray:
    return _apply(x, window, min_periods, False, lambda w: np.nanquantile(w, q, axis=1))
