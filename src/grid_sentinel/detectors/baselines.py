"""Reference detectors, including the two rules most common in utility practice (the Iglewicz-Hoaglin
modified z-score and a fixed relative-deviation band). Centred-window detectors are batch only; the
modified z-score, which looks back only, streams."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from grid_sentinel.base import Detector
from grid_sentinel.rolling import rolling_mean, rolling_median, rolling_std
from grid_sentinel.types import REASON_DTYPE, Decision, DetectionResult, as_series

MAD_TO_SIGMA = 1.4826


def _result(v, t, score, k) -> DetectionResult:
    flags = np.nan_to_num(score, nan=0.0) > k
    reason = np.where(flags, "baseline", "").astype(REASON_DTYPE)
    return DetectionResult(t, v, score, np.full(len(v), float(k)), flags, reason)


class RollingZScore(Detector):
    """|x - rolling mean| / rolling std over a centred window."""

    def __init__(self, window: int = 168, k: float = 3.0):
        self.window = int(window)
        self.k = float(k)

    def predict(self, values: Any, timestamps: Any = None, context: Any = None) -> DetectionResult:
        v, t = as_series(values, timestamps)
        mp = max(3, self.window // 4)
        mean = rolling_mean(v, self.window, mp, center=True)
        std = rolling_std(v, self.window, mp, center=True)
        with np.errstate(all="ignore"):
            z = np.abs((v - mean) / std)
        return _result(v, t, z, self.k)


class Hampel(Detector):
    """|x - rolling median| / (1.4826 rolling MAD) over a centred window."""

    def __init__(self, window: int = 24, k: float = 3.0):
        self.window = int(window)
        self.k = float(k)

    def predict(self, values: Any, timestamps: Any = None, context: Any = None) -> DetectionResult:
        v, t = as_series(values, timestamps)
        mp = max(3, self.window // 4)
        med = rolling_median(v, self.window, mp, center=True)
        mad = rolling_median(np.abs(v - med), self.window, mp, center=True)
        with np.errstate(all="ignore"):
            z = np.abs(v - med) / (MAD_TO_SIGMA * mad)
        return _result(v, t, z, self.k)


class IQR(Detector):
    """Global inter-quartile rule: distance outside [Q1, Q3] in units of the IQR."""

    def __init__(self, k: float = 1.5):
        self.k = float(k)

    def predict(self, values: Any, timestamps: Any = None, context: Any = None) -> DetectionResult:
        v, t = as_series(values, timestamps)
        q1, q3 = np.nanpercentile(v, [25, 75])
        width = q3 - q1
        dist = np.maximum(q1 - v, v - q3) / width if width > 0 else np.zeros_like(v)
        return _result(v, t, dist, self.k)


class ModifiedZScore(Detector):
    """Iglewicz-Hoaglin modified z-score on a trailing window: 0.6745 |x - median| / MAD, both computed on
    the ``window`` readings before the current one."""

    supports_streaming = True

    def __init__(self, window: int = 24 * 30, k: float = 3.5):
        self.window = int(window)
        self.k = float(k)
        self.reset()

    def reset(self) -> None:
        self._buf: deque = deque(maxlen=self.window)   # readings
        self._dev: deque = deque(maxlen=self.window)   # each reading's deviation from the median before it

    @property
    def _min_periods(self) -> int:
        return max(24, self.window // 4)

    @staticmethod
    def _median(buf: deque, min_periods: int) -> float:
        arr = np.fromiter(buf, dtype=float, count=len(buf))
        finite = arr[np.isfinite(arr)]
        return float(np.median(finite)) if len(finite) >= min_periods else float("nan")

    def update(self, timestamp: Any, value: float, **context: Any) -> Decision:
        x = float(value)
        med = self._median(self._buf, self._min_periods)      # median of the readings before this one
        mad = self._median(self._dev, self._min_periods)      # median of the deviations before this one
        dev = abs(x - med)
        with np.errstate(all="ignore"):
            score = float(0.6745 * dev / mad) if np.isfinite(mad) and np.isfinite(dev) else float("nan")
        self._buf.append(x)
        self._dev.append(dev)
        flag = bool(np.nan_to_num(score, nan=0.0) > self.k)
        return Decision(timestamp, x, score, self.k, flag, "baseline" if flag else "", float("nan"), {})

    def predict(self, values: Any, timestamps: Any = None, context: Any = None) -> DetectionResult:
        v, t = as_series(values, timestamps)
        mp = self._min_periods
        med = rolling_median(v, self.window, mp, center=False)
        med = np.concatenate([[np.nan], med[:-1]])            # median of the window before each reading
        dev = np.abs(v - med)
        mad = rolling_median(dev, self.window, mp, center=False)
        mad = np.concatenate([[np.nan], mad[:-1]])
        with np.errstate(all="ignore"):
            z = np.abs(0.6745 * dev / mad)
        self.reset()
        for x, d in zip(v, dev):
            self._buf.append(float(x))
            self._dev.append(float(d))
        return _result(v, t, z, self.k)

    def get_state(self) -> dict[str, Any]:
        return {"buf": np.array(self._buf, dtype=float), "dev": np.array(self._dev, dtype=float)}

    def set_state(self, state: dict[str, Any]) -> ModifiedZScore:
        self.reset()
        for x in state["buf"]:
            self._buf.append(float(x))
        for d in state["dev"]:
            self._dev.append(float(d))
        return self


class RelativeDeviation(Detector):
    """|x - centred moving average| / moving average above ``k`` (15% by default)."""

    def __init__(self, window: int = 5, k: float = 0.15):
        self.window = int(window)
        self.k = float(k)

    def predict(self, values: Any, timestamps: Any = None, context: Any = None) -> DetectionResult:
        v, t = as_series(values, timestamps)
        mean = rolling_mean(v, self.window, max(2, self.window // 2), center=True)
        with np.errstate(all="ignore"):
            dev = np.abs(v - mean) / mean
        return _result(v, t, dev, self.k)
