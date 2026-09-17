"""Frozen telemetry: a run of identical consecutive readings is a stale value, not a measurement.

Causal semantics (v0.4): a reading is flagged when it is the ``min_run``-th or a later member of a run
of identical values. Earlier members of the run are not flagged, because a streaming detector cannot
know yet that the run will reach ``min_run``. The score is the length of the run so far. A missing
reading breaks the run.
"""

from __future__ import annotations

from typing import Any

from grid_sentinel.base import Detector
from grid_sentinel.types import Decision, DetectionResult


class StuckValues(Detector):
    supports_streaming = True

    def __init__(self, min_run: int = 3):
        if min_run < 2:
            raise ValueError("min_run must be at least 2")
        self.min_run = int(min_run)
        self.reset()

    def reset(self) -> None:
        self._prev: float | None = None
        self._run = 0

    def update(self, timestamp: Any, value: float, **context: Any) -> Decision:
        x = float(value)
        if self._prev is not None and x == self._prev:
            self._run += 1
        else:
            self._run = 1
        self._prev = x
        flag = self._run >= self.min_run
        return Decision(timestamp, x, float(self._run), float(self.min_run), flag, "stuck" if flag else "", float("nan"), {})

    def predict(self, values: Any, timestamps: Any = None, context: Any = None) -> DetectionResult:
        self.reset()
        return self.replay(values, timestamps, context)

    def get_state(self) -> dict[str, Any]:
        return {"prev": self._prev, "run": self._run}

    def set_state(self, state: dict[str, Any]) -> StuckValues:
        self._prev = None if state["prev"] is None else float(state["prev"])
        self._run = int(state["run"])
        return self
