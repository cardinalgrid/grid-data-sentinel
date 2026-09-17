"""Window autoencoder with a tanh bottleneck and an L1 activity penalty, trained by Adam in numpy.

``fit`` trains the network on a history and fixes the threshold (``factor`` times the trimmed mean of
the training scores). ``predict`` scores a series with the trained model: by default each reading's
score is the mean reconstruction error over the windows that contain it (the v0.3 definition, which
looks a few readings ahead); with ``causal=True`` only the window ending at the reading is used, which
is what ``update`` does in streaming, so that streaming and batch agree exactly.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from grid_sentinel.base import Detector
from grid_sentinel.types import REASON_DTYPE, Decision, DetectionResult, as_series


def _windows(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values) - window + 1
    if n <= 0:
        raise ValueError("series shorter than the window")
    return np.lib.stride_tricks.sliding_window_view(values, window)[:n]


def _trim_mean(x: np.ndarray, proportion: float) -> float:
    s = np.sort(x)
    k = int(len(s) * proportion)
    return float(np.mean(s[k : len(s) - k])) if len(s) - 2 * k > 0 else float(np.mean(s))


class SparseAutoencoder(Detector):
    supports_streaming = True

    def __init__(
        self,
        window: int = 4,
        encoding_dim: int = 2,
        l1: float = 1e-4,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 1e-2,
        factor: float = 20.0,
        trim: float = 0.1,
        seed: int = 0,
        causal: bool = False,
    ):
        self.window = int(window)
        self.encoding_dim = int(encoding_dim)
        self.l1 = float(l1)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.factor = float(factor)
        self.trim = float(trim)
        self.seed = int(seed)
        self.causal = bool(causal)
        self.reset()

    def reset(self) -> None:
        self._fitted = False
        self._lo = self._hi = float("nan")
        self._thr = float("nan")
        self._median = float("nan")
        self._buf: deque = deque(maxlen=self.window)

    # ---- network -----------------------------------------------------------------------------------
    def _scale(self, values: np.ndarray) -> np.ndarray:
        return (values - self._lo) / (self._hi - self._lo)

    def _init(self, rng: np.random.Generator) -> None:
        w, h = self.window, self.encoding_dim
        self.W1 = rng.normal(0, np.sqrt(1.0 / w), (w, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, np.sqrt(1.0 / h), (h, w))
        self.b2 = np.zeros(w)
        self._params = [self.W1, self.b1, self.W2, self.b2]
        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]
        self._t = 0

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H = np.tanh(X @ self.W1 + self.b1)
        return H, H @ self.W2 + self.b2

    def _step(self, X: np.ndarray) -> float:
        H, Y = self._forward(X)
        err = Y - X
        loss = float(np.mean(err**2)) + self.l1 * float(np.mean(np.abs(H)))
        dY = 2.0 * err / err.size
        dW2 = H.T @ dY
        db2 = dY.sum(axis=0)
        dH = dY @ self.W2.T + self.l1 * np.sign(H) / H.size
        dZ = dH * (1.0 - H**2)
        dW1 = X.T @ dZ
        db1 = dZ.sum(axis=0)
        grads = [dW1, db1, dW2, db2]
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for p, g, m, v in zip(self._params, grads, self._m, self._v):
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g**2
            p -= self.lr * (m / (1 - b1**self._t)) / (np.sqrt(v / (1 - b2**self._t)) + eps)
        return loss

    # ---- contract ----------------------------------------------------------------------------------
    def fit(self, values: Any, timestamps: Any = None, context: Any = None) -> SparseAutoencoder:
        v, _ = as_series(values, timestamps)
        clean = v[~np.isnan(v)]
        self.reset()
        self._lo, self._hi = float(np.min(clean)), float(np.max(clean))
        if self._hi <= self._lo:
            raise ValueError("series is constant")
        self._median = float(np.median(clean))
        X = _windows(self._scale(clean), self.window)
        rng = np.random.default_rng(self.seed)
        self._init(rng)
        self.history: list[float] = []
        for _ in range(self.epochs):
            order = rng.permutation(len(X))
            losses = [self._step(X[order[s : s + self.batch_size]]) for s in range(0, len(X), self.batch_size)]
            self.history.append(float(np.mean(losses)))
        self._fitted = True
        scores = self._scores(v)
        self._thr = self.factor * _trim_mean(scores[~np.isnan(scores)], self.trim)
        return self

    def _scores(self, v: np.ndarray) -> np.ndarray:
        filled = np.where(np.isnan(v), self._median, v)
        X = _windows(self._scale(filled), self.window)
        _, Y = self._forward(X)
        err = (Y - X) ** 2
        out = np.full(len(v), np.nan)
        if self.causal:
            out[self.window - 1 :] = err.mean(axis=1)
        else:
            acc = np.zeros(len(v))
            cnt = np.zeros(len(v))
            for j in range(self.window):
                acc[j : j + len(X)] += err[:, j]
                cnt[j : j + len(X)] += 1
            out = acc / np.maximum(cnt, 1)
        out[np.isnan(v)] = np.nan
        return out

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError("call fit (or fit_predict) before predict, score or update")

    def predict(self, values: Any, timestamps: Any = None, context: Any = None) -> DetectionResult:
        self._require_fit()
        v, t = as_series(values, timestamps)
        scores = self._scores(v)
        flags = np.nan_to_num(scores, nan=0.0) > self._thr
        reason = np.where(flags, "baseline", "").astype(REASON_DTYPE)
        self._buf.clear()
        for x in v[-self.window :]:
            self._buf.append(float(x))
        return DetectionResult(t, v, scores, np.full(len(v), self._thr), flags, reason)

    def fit_predict(self, values: Any, timestamps: Any = None, context: Any = None) -> DetectionResult:
        return self.fit(values, timestamps, context).predict(values, timestamps, context)

    def update(self, timestamp: Any, value: float, **context: Any) -> Decision:
        self._require_fit()
        x = float(value)
        self._buf.append(self._median if np.isnan(x) else x)
        score = float("nan")
        if len(self._buf) == self.window and not np.isnan(x):
            X = self._scale(np.fromiter(self._buf, dtype=float, count=self.window))[None, :]
            _, Y = self._forward(X)
            score = float(np.mean((Y - X) ** 2))
        flag = bool(np.nan_to_num(score, nan=0.0) > self._thr)
        return Decision(timestamp, x, score, self._thr, flag, "baseline" if flag else "", float("nan"), {})

    def get_state(self) -> dict[str, Any]:
        self._require_fit()
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2, "lo": self._lo, "hi": self._hi,
                "thr": self._thr, "median": self._median, "buf": np.array(self._buf, dtype=float)}

    def set_state(self, state: dict[str, Any]) -> SparseAutoencoder:
        self.reset()
        self.W1, self.b1, self.W2, self.b2 = (np.array(state[k], dtype=float) for k in ("W1", "b1", "W2", "b2"))
        self._params = [self.W1, self.b1, self.W2, self.b2]
        self._lo, self._hi, self._thr, self._median = (float(state[k]) for k in ("lo", "hi", "thr", "median"))
        for x in state["buf"]:
            self._buf.append(float(x))
        self._fitted = True
        return self
