"""Recursive TEDA (typicality and eccentricity data analytics) for streaming anomaly detection.

The detector keeps a recursive mean and variance of everything it has seen and scores each new
sample by its eccentricity, a distribution-free measure of how far the sample sits from the data
seen so far. A sample is flagged when its normalised eccentricity exceeds the threshold
(m^2 + 1) / (2k), where k is the number of samples processed and m plays the role of a number of
standard deviations (Angelov, 2014). No window, no training set and no distributional assumption
are needed, which makes the method suitable for telemetry that arrives one reading at a time.

Differenced mode (``diff=True``) runs the recursion on first differences x[k] - x[k-1]. On load
curves with a strong daily cycle this removes most of the cycle before scoring and gives far fewer
false alarms; it also flags the sample after a real outlier (the return to normal is itself a large
difference), so ``detect`` clears that second flag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TEDAResult:
    """State of the detector after one update."""

    k: int
    mean: float
    variance: float
    eccentricity: float
    norm_eccentricity: float
    threshold: float
    is_anomaly: bool


class RecursiveTEDA:
    """Streaming anomaly detector based on recursive eccentricity.

    Parameters
    ----------
    m
        Sensitivity, in the sense of a Chebyshev-type bound: a sample is flagged when its squared
        distance to the running mean exceeds m^2 times the running variance.
    diff
        Score first differences instead of levels. Recommended for series with a daily cycle.
    """

    def __init__(self, m: float = 2.6, diff: bool = False) -> None:
        if m <= 0:
            raise ValueError("m must be positive")
        self.m = float(m)
        self.diff = bool(diff)
        self.reset()

    def reset(self) -> None:
        self.k = 0
        self.mean = 0.0
        self.variance = 0.0
        self._prev = None

    def update(self, x: float) -> TEDAResult:
        """Feed one sample and return the detector state."""
        x = float(x)
        self.k += 1
        if self.k == 1:
            if self.diff:
                self._prev = x
                value = 0.0
            else:
                value = x
            self.mean = value
            self.variance = 0.0
            return TEDAResult(self.k, self.mean, self.variance, 0.0, 0.0, float("inf"), False)

        if self.diff:
            value = x - self._prev
            self._prev = x
        else:
            value = x

        k = self.k
        self.mean = ((k - 1) / k) * self.mean + value / k
        self.variance = ((k - 1) / k) * self.variance + (value - self.mean) ** 2 / (k - 1)

        if self.variance > 0:
            eccentricity = 1.0 / k + (value - self.mean) ** 2 / (k * self.variance)
        else:
            eccentricity = 1.0 / k
        norm_eccentricity = eccentricity / 2.0
        threshold = (self.m**2 + 1.0) / (2.0 * k)
        return TEDAResult(
            k, self.mean, self.variance, eccentricity, norm_eccentricity, threshold, norm_eccentricity > threshold
        )

    def detect(self, series: pd.Series | np.ndarray) -> pd.DataFrame:
        """Run the detector over a whole series (in order) and return per-sample scores.

        Columns: ``value``, ``norm_eccentricity``, ``threshold``, ``score`` (ratio of the two;
        greater than 1 means flagged) and ``is_anomaly``.
        """
        values = np.asarray(series, dtype=float)
        index = series.index if isinstance(series, pd.Series) else pd.RangeIndex(len(values))
        self.reset()
        ecc = np.zeros(len(values))
        thr = np.full(len(values), np.inf)
        flags = np.zeros(len(values), dtype=bool)
        for i, x in enumerate(values):
            if np.isnan(x):
                continue
            r = self.update(x)
            ecc[i] = r.norm_eccentricity
            thr[i] = r.threshold
            flags[i] = r.is_anomaly
        if self.diff:
            # the sample after a flagged one is the return to normal, not a second anomaly
            after = np.zeros_like(flags)
            after[1:] = flags[:-1]
            flags = flags & ~after
        with np.errstate(divide="ignore", invalid="ignore"):
            score = np.where(np.isfinite(thr) & (thr > 0), ecc / thr, 0.0)
        return pd.DataFrame(
            {"value": values, "norm_eccentricity": ecc, "threshold": thr, "score": score, "is_anomaly": flags},
            index=index,
        )
