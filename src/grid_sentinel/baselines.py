"""Reference detectors the two main methods are compared against."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _frame(series: pd.Series | np.ndarray, score: np.ndarray, thr: float, flags: np.ndarray) -> pd.DataFrame:
    index = series.index if isinstance(series, pd.Series) else pd.RangeIndex(len(score))
    return pd.DataFrame(
        {"value": np.asarray(series, dtype=float), "score": score, "threshold": thr, "is_anomaly": flags},
        index=index,
    )


def rolling_zscore(series: pd.Series | np.ndarray, window: int = 168, k: float = 3.0) -> pd.DataFrame:
    """Flag readings more than ``k`` rolling standard deviations from the rolling mean."""
    s = pd.Series(np.asarray(series, dtype=float))
    mean = s.rolling(window, min_periods=max(3, window // 4), center=True).mean()
    std = s.rolling(window, min_periods=max(3, window // 4), center=True).std()
    z = ((s - mean) / std).abs().to_numpy()
    flags = np.nan_to_num(z, nan=0.0) > k
    return _frame(series, z, k, flags)


def hampel(series: pd.Series | np.ndarray, window: int = 24, k: float = 3.0) -> pd.DataFrame:
    """Hampel filter: median and MAD over a centred window, flag if |x - median| > k * 1.4826 * MAD."""
    s = pd.Series(np.asarray(series, dtype=float))
    med = s.rolling(window, min_periods=max(3, window // 4), center=True).median()
    mad = (s - med).abs().rolling(window, min_periods=max(3, window // 4), center=True).median()
    z = ((s - med).abs() / (1.4826 * mad)).to_numpy()
    flags = np.nan_to_num(z, nan=0.0) > k
    return _frame(series, z, k, flags)


def iqr(series: pd.Series | np.ndarray, k: float = 1.5) -> pd.DataFrame:
    """Global inter-quartile rule: flag readings outside [Q1 - k IQR, Q3 + k IQR]."""
    v = np.asarray(series, dtype=float)
    q1, q3 = np.nanpercentile(v, [25, 75])
    width = q3 - q1
    dist = np.maximum(q1 - v, v - q3) / width if width > 0 else np.zeros_like(v)
    flags = np.nan_to_num(dist, nan=0.0) > k
    return _frame(series, dist, k, flags)


def modified_zscore(series: pd.Series | np.ndarray, window: int = 24 * 30, k: float = 3.5) -> pd.DataFrame:
    """Iglewicz-Hoaglin modified z-score on a trailing window: 0.6745 (x - median) / MAD, flag if above k."""
    s = pd.Series(np.asarray(series, dtype=float))
    med = s.rolling(window, min_periods=max(24, window // 4)).median().shift(1)
    mad = (s - med).abs().rolling(window, min_periods=max(24, window // 4)).median().shift(1)
    z = (0.6745 * (s - med) / mad).abs().to_numpy()
    flags = np.nan_to_num(z, nan=0.0) > k
    return _frame(series, z, k, flags)


def relative_deviation(series: pd.Series | np.ndarray, window: int = 5, k: float = 0.15) -> pd.DataFrame:
    """Relative deviation from a centred moving average: |x - mean| / mean, flag if above k (15% by default).
    The one-line rule common in utility practice."""
    s = pd.Series(np.asarray(series, dtype=float))
    mean = s.rolling(window, min_periods=max(2, window // 2), center=True).mean()
    dev = ((s - mean).abs() / mean).to_numpy()
    flags = np.nan_to_num(dev, nan=0.0) > k
    return _frame(series, dev, k, flags)
