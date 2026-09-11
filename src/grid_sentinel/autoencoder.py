"""Sparse autoencoder for anomaly detection on a univariate load series.

The series is min-max scaled, cut into sliding windows of ``window`` consecutive readings, and an
autoencoder with a small bottleneck and an L1 penalty on the code activations is trained to
reconstruct the windows. Readings that the model cannot reconstruct get a high error; a reading's
score is the mean reconstruction error over the windows that contain it. The default threshold is
``factor`` times the trimmed mean of the scores, the form of rule used in the published version of
the method (Guerra Filho et al., Energies 2024). The default factor was set on hourly EIA-930 data
with injected anomalies (see ``scripts/tune.py``); on 15-minute substation data a factor of 3 was
used in the paper.

The network is implemented in NumPy so that the package has no deep-learning dependency. It is
small on purpose: the published models have one hidden layer of a few units, and training on a
decade of hourly data takes seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _windows(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values) - window + 1
    if n <= 0:
        raise ValueError("series shorter than window")
    return np.lib.stride_tricks.sliding_window_view(values, window)[:n]


def _trim_mean(x: np.ndarray, proportion: float) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    cut = int(proportion * len(x))
    if cut > 0:
        x = x[cut:-cut]
    return float(np.mean(x)) if len(x) else float("nan")


class SparseAutoencoder:
    """Window autoencoder with a tanh bottleneck and L1 activity penalty.

    Parameters
    ----------
    window
        Number of consecutive readings per input vector.
    encoding_dim
        Size of the bottleneck.
    l1
        Weight of the L1 penalty on the bottleneck activations (sparsity).
    epochs, batch_size, lr
        Training schedule (Adam optimiser, mean squared error loss).
    factor, trim
        Threshold rule: ``factor`` times the trimmed mean (``trim`` cut on each side) of the scores.
    seed
        Random seed for initialisation and batching.
    """

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
    ) -> None:
        self.window = int(window)
        self.encoding_dim = int(encoding_dim)
        self.l1 = float(l1)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.factor = float(factor)
        self.trim = float(trim)
        self.seed = int(seed)
        self._fitted = False

    # ---------------------------------------------------------------- scaling
    def _scale(self, values: np.ndarray) -> np.ndarray:
        return (values - self._lo) / (self._hi - self._lo)

    # ---------------------------------------------------------------- network
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
        Y = H @ self.W2 + self.b2  # linear output, same shape as X
        return H, Y

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
        # Adam
        self._t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for p, g, m, v in zip(self._params, grads, self._m, self._v):
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g**2
            m_hat = m / (1 - b1**self._t)
            v_hat = v / (1 - b2**self._t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        return loss

    # ---------------------------------------------------------------- public API
    def fit(self, series: pd.Series | np.ndarray) -> SparseAutoencoder:
        values = np.asarray(series, dtype=float)
        values = values[~np.isnan(values)]
        self._lo, self._hi = float(np.min(values)), float(np.max(values))
        if self._hi <= self._lo:
            raise ValueError("series is constant")
        X = _windows(self._scale(values), self.window)
        rng = np.random.default_rng(self.seed)
        self._init(rng)
        self.history: list[float] = []
        for _ in range(self.epochs):
            order = rng.permutation(len(X))
            losses = []
            for start in range(0, len(X), self.batch_size):
                losses.append(self._step(X[order[start : start + self.batch_size]]))
            self.history.append(float(np.mean(losses)))
        self._fitted = True
        return self

    def score(self, series: pd.Series | np.ndarray) -> np.ndarray:
        """Per-reading reconstruction error (mean over the windows containing the reading)."""
        if not self._fitted:
            raise RuntimeError("call fit first")
        values = np.asarray(series, dtype=float)
        filled = np.where(np.isnan(values), np.nanmedian(values), values)
        X = _windows(self._scale(filled), self.window)
        _, Y = self._forward(X)
        err = (Y - X) ** 2
        acc = np.zeros(len(values))
        cnt = np.zeros(len(values))
        for j in range(self.window):
            acc[j : j + len(X)] += err[:, j]
            cnt[j : j + len(X)] += 1
        out = acc / np.maximum(cnt, 1)
        out[np.isnan(values)] = np.nan
        return out

    def threshold(self, scores: np.ndarray) -> float:
        s = scores[~np.isnan(scores)]
        return self.factor * _trim_mean(s, self.trim)

    def detect(self, series: pd.Series | np.ndarray, fit: bool = True) -> pd.DataFrame:
        """Fit (optional) and flag readings whose score exceeds the threshold."""
        if fit or not self._fitted:
            self.fit(series)
        scores = self.score(series)
        thr = self.threshold(scores)
        index = series.index if isinstance(series, pd.Series) else pd.RangeIndex(len(scores))
        flags = np.nan_to_num(scores, nan=0.0) > thr
        return pd.DataFrame(
            {
                "value": np.asarray(series, dtype=float),
                "score": scores,
                "threshold": thr,
                "is_anomaly": flags,
            },
            index=index,
        )
