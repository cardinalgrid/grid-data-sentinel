"""The estimator contract every detector follows.

Batch: ``fit``, ``predict``, ``score``, ``fit_predict``. Streaming: ``update`` one reading at a time,
for detectors that declare ``supports_streaming``. State: ``get_state`` / ``set_state`` (plain types and
numpy arrays, serialisable with ``grid_sentinel.state``). Parameters live only in the constructor and are
reported by ``get_params`` / ``set_params``, in the scikit-learn convention, without depending on it.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from grid_sentinel.types import REASON_DTYPE, Context, Decision, DetectionResult, as_series, empty_decision


class Detector:
    supports_streaming: bool = False

    # ---- parameters -------------------------------------------------------------------------------
    @classmethod
    def _param_names(cls) -> list[str]:
        sig = inspect.signature(cls.__init__)
        return [p.name for p in sig.parameters.values() if p.name != "self" and p.kind == p.POSITIONAL_OR_KEYWORD]

    def get_params(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._param_names()}

    def set_params(self, **params: Any) -> Detector:
        allowed = self._param_names()
        for name, value in params.items():
            if name not in allowed:
                raise ValueError(f"unknown parameter {name!r} for {type(self).__name__}; valid: {allowed}")
            setattr(self, name, value)
        self.reset()
        return self

    def __repr__(self) -> str:
        sig = inspect.signature(type(self).__init__)
        parts = []
        for name in self._param_names():
            default = sig.parameters[name].default
            value = getattr(self, name)
            if default is inspect.Parameter.empty or value != default:
                parts.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    # ---- lifecycle --------------------------------------------------------------------------------
    def reset(self) -> None:
        """Forget everything learned or seen. Subclasses override and call nothing here."""

    def fit(self, values: Any, timestamps: Any = None, context: Context | None = None) -> Detector:
        """Learn from a history. Streaming detectors replay it into their state; others learn nothing."""
        self.reset()
        if self.supports_streaming:
            self.replay(values, timestamps, context)
        return self

    def predict(self, values: Any, timestamps: Any = None, context: Context | None = None) -> DetectionResult:
        raise NotImplementedError

    def score(self, values: Any, timestamps: Any = None, context: Context | None = None) -> np.ndarray:
        return self.predict(values, timestamps, context).score

    def fit_predict(self, values: Any, timestamps: Any = None, context: Context | None = None) -> DetectionResult:
        self.reset()
        return self.predict(values, timestamps, context)

    def update(self, timestamp: Any, value: float, **context: Any) -> Decision:
        raise NotImplementedError(f"{type(self).__name__} does not support streaming; use predict on a series")

    # ---- state ------------------------------------------------------------------------------------
    def get_state(self) -> dict[str, Any]:
        return {}

    def set_state(self, state: dict[str, Any]) -> Detector:
        return self

    # ---- helpers for subclasses -------------------------------------------------------------------
    def replay(self, values: Any, timestamps: Any = None, context: Context | None = None) -> DetectionResult:
        """Feed the series through ``update`` in order and collect a ``DetectionResult``.

        NaN readings are not fed; they come out unflagged with score 0. The detector's state is left as it
        is after the last reading, which is what ``fit`` relies on.
        """
        v, t = as_series(values, timestamps)
        ctx = context.aligned(t) if (context is not None and t is not None) else None
        n = len(v)
        score = np.zeros(n)
        thr = np.full(n, np.inf)
        flags = np.zeros(n, dtype=bool)
        reason = np.full(n, "", dtype=REASON_DTYPE)
        expected = np.full(n, np.nan)
        checks: dict[str, np.ndarray] = {}
        for i in range(n):
            ts = t[i] if t is not None else i
            if np.isnan(v[i]):
                d = empty_decision(ts, v[i])
            else:
                d = self.update(ts, v[i], **(ctx.at(i) if ctx is not None else {}))
            score[i], thr[i], flags[i], reason[i], expected[i] = d.score, d.threshold, d.is_anomaly, d.reason, d.expected
            for k, c in d.checks.items():
                if k not in checks:
                    checks[k] = np.full(n, np.nan)
                checks[k][i] = c
        has_expected = np.isfinite(expected).any()
        return DetectionResult(t, v, score, thr, flags, reason, expected if has_expected else None, checks)
