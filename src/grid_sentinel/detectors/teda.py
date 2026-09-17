"""Recursive typicality-and-eccentricity detector (TEDA), streaming by nature.

A reading is flagged when its squared distance to the running mean exceeds m^2 times the running
variance, expressed through the normalised eccentricity and the Chebyshev-type threshold (m^2 + 1)/(2 k).
``diff=True`` scores first differences (recommended for series with a daily cycle). ``robust=True``
uses the winsorised update: a flagged sample enters the statistics clipped to the acceptance boundary.
``half_life_hours`` adds exponential forgetting: the running mean and variance weight the past with a
half-life, so that the detector adapts to level shifts; the effective sample count is capped accordingly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from grid_sentinel.base import Detector
from grid_sentinel.types import Decision, DetectionResult


class TEDA(Detector):
    supports_streaming = True

    def __init__(self, m: float = 4.0, diff: bool = False, robust: bool = False, half_life_hours: float | None = None):
        if m <= 0:
            raise ValueError("m must be positive")
        if half_life_hours is not None and half_life_hours <= 0:
            raise ValueError("half_life_hours must be positive or None")
        self.m = float(m)
        self.diff = bool(diff)
        self.robust = bool(robust)
        self.half_life_hours = None if half_life_hours is None else float(half_life_hours)
        self.reset()

    def reset(self) -> None:
        self.k = 0
        self.mean = 0.0
        self.variance = 0.0
        self._prev: float | None = None

    @property
    def _lambda(self) -> float | None:
        return None if self.half_life_hours is None else 0.5 ** (1.0 / self.half_life_hours)

    def update(self, timestamp: Any, value: float, **context: Any) -> Decision:
        x = float(value)
        reason = "teda_diff" if self.diff else "teda_level"
        self.k += 1
        if self.k == 1:
            if self.diff:
                self._prev = x
                sample = 0.0
            else:
                sample = x
            self.mean = sample
            self.variance = 0.0
            return Decision(timestamp, x, 0.0, float("inf"), False, "", float("nan"), {})
        if self.diff:
            sample = x - self._prev
            self._prev = x
        else:
            sample = x
        k = self.k
        lam = self._lambda
        k_eff = k if lam is None else min(k, 1.0 / (1.0 - lam))
        prev_mean, prev_var = self.mean, self.variance
        if lam is None or k < 1.0 / (1.0 - lam):
            self.mean = ((k - 1) / k) * prev_mean + sample / k
            self.variance = ((k - 1) / k) * prev_var + (sample - self.mean) ** 2 / (k - 1)
        else:
            self.mean = lam * prev_mean + (1.0 - lam) * sample
            self.variance = lam * prev_var + (1.0 - lam) * (sample - self.mean) ** 2
        if self.variance > 0:
            ecc = 1.0 / k_eff + (sample - self.mean) ** 2 / (k_eff * self.variance)
        else:
            ecc = 1.0 / k_eff
        norm_ecc = ecc / 2.0
        threshold = (self.m**2 + 1.0) / (2.0 * k_eff)
        flag = norm_ecc > threshold
        if flag and self.robust:
            sigma = float(np.sqrt(prev_var)) if prev_var > 0 else 0.0
            clipped = float(np.clip(sample, prev_mean - self.m * sigma, prev_mean + self.m * sigma))
            if lam is None or k < 1.0 / (1.0 - lam):
                self.mean = ((k - 1) / k) * prev_mean + clipped / k
                self.variance = ((k - 1) / k) * prev_var + (clipped - self.mean) ** 2 / (k - 1)
            else:
                self.mean = lam * prev_mean + (1.0 - lam) * clipped
                self.variance = lam * prev_var + (1.0 - lam) * (clipped - self.mean) ** 2
        score = norm_ecc / threshold if threshold > 0 else 0.0
        return Decision(timestamp, x, float(score), float(threshold), bool(flag), reason if flag else "", float("nan"), {})

    def predict(self, values: Any, timestamps: Any = None, context: Any = None) -> DetectionResult:
        self.reset()
        out = self.replay(values, timestamps, context)
        if self.diff:
            # the sample after a flagged one is the return to normal, not a second anomaly
            after = np.zeros_like(out.is_anomaly)
            after[1:] = out.is_anomaly[:-1]
            drop = out.is_anomaly & after
            out.is_anomaly[drop] = False
            out.reason[drop] = ""
        return out

    def get_state(self) -> dict[str, Any]:
        return {"k": self.k, "mean": self.mean, "variance": self.variance, "prev": self._prev}

    def set_state(self, state: dict[str, Any]) -> TEDA:
        self.k = int(state["k"])
        self.mean = float(state["mean"])
        self.variance = float(state["variance"])
        self._prev = None if state["prev"] is None else float(state["prev"])
        return self
