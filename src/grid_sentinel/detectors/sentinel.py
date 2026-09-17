"""The composite detector, one reading at a time.

Base alarms come from the recursive eccentricity detectors on levels and on first differences and from
the frozen-value rule. An alarm is kept only if the reading, or the one before it, is implausible
against the expected profile (residual beyond ``k_profile`` robust standard deviations, more than
``band`` from the expectation, or a ratio outside ``1/scale_ratio``..``scale_ratio``). Frozen runs,
non-positive readings and gross ratios are faults regardless. A kept alarm on a reading *above* the
expectation then faces the cross-checks (profile, neighbours, weather): it stays a fault only if at
least two of the available checks fail to confirm it as genuine, and is otherwise withdrawn as
``extreme_kept``. The expectation is built with base-flagged readings masked out. ``cross_checks=()``
with ``regime="load"`` and ``extremes="off"`` is the v0.2 composite; the defaults are v0.3.

Reasons: ``teda_level`` / ``teda_diff`` / ``stuck`` for kept base alarms, ``gross_ratio`` and
``non_positive`` for the hard rules, ``dropped`` for a base alarm the profile did not confirm and
``extreme_kept`` for one the cross-checks withdrew (both unflagged). Missing readings are unflagged.
"""

from __future__ import annotations

import math
from typing import Any

from grid_sentinel.base import Detector
from grid_sentinel.detectors.crosscheck import NeighborCheck, WeatherCheck, preserve_one
from grid_sentinel.detectors.profile import ProfileResidual
from grid_sentinel.detectors.stuck import StuckValues
from grid_sentinel.detectors.teda import TEDA
from grid_sentinel.types import Decision, DetectionResult, as_series

CHECK_NAMES = ("profile", "neighbors", "weather")


class Sentinel(Detector):
    supports_streaming = True
    handles_missing = True

    def __init__(
        self,
        m_level: float = 4.0,
        m_diff: float = 3.0,
        min_run: int = 3,
        k_profile: float = 5.0,
        band: float = 0.35,
        scale_ratio: float = 3.0,
        regime: str | None = "temperature",
        cross_checks: tuple[str, ...] = CHECK_NAMES,
        extremes: str = "preserve",
        profile_band: float = 0.35,
        neighbor_pct: float = 95.0,
        half_life_hours: float | None = None,
        robust: bool = False,
    ):
        if extremes not in ("preserve", "off"):
            raise ValueError("extremes must be 'preserve' or 'off'")
        unknown = set(cross_checks) - set(CHECK_NAMES)
        if unknown:
            raise ValueError(f"unknown cross-checks {sorted(unknown)}; valid: {CHECK_NAMES}")
        self.m_level = float(m_level)
        self.m_diff = float(m_diff)
        self.min_run = int(min_run)
        self.k_profile = float(k_profile)
        self.band = float(band)
        self.scale_ratio = float(scale_ratio)
        self.regime = regime
        self.cross_checks = tuple(cross_checks)
        self.extremes = extremes
        self.profile_band = float(profile_band)
        self.neighbor_pct = float(neighbor_pct)
        self.half_life_hours = half_life_hours
        self.robust = bool(robust)
        self.reset()

    def reset(self) -> None:
        self._level = TEDA(m=self.m_level, diff=False, robust=self.robust, half_life_hours=self.half_life_hours)
        self._diff = TEDA(m=self.m_diff, diff=True, robust=self.robust, half_life_hours=self.half_life_hours)
        self._stuck = StuckValues(min_run=self.min_run)
        self._profile = ProfileResidual(k=self.k_profile, regime=self.regime)
        self._neighbors = NeighborCheck(pct=self.neighbor_pct)
        self._weather = WeatherCheck()
        self._prev_diff_raw = False
        self._prev_implausible = False

    # ---- one reading -------------------------------------------------------------------------------
    def update(self, timestamp: Any, value: float, temperature_f: float = float("nan"),
               neighbors: dict[str, float] | None = None, regime: float = float("nan"), **context: Any) -> Decision:
        x = float(value)
        checks: dict[str, float] = {}
        if "neighbors" in self.cross_checks:
            checks["neighbors"] = self._neighbors.update(timestamp, neighbors)
        if "weather" in self.cross_checks:
            checks["weather"] = self._weather.update(timestamp, temperature_f)
        if math.isnan(x):
            self._stuck.update(timestamp, x)
            self._profile.update(timestamp, x, temperature_f=temperature_f, regime=regime, exclude=True)
            self._prev_diff_raw = False
            self._prev_implausible = False
            return Decision(timestamp, x, 0.0, 1.0, False, "", float("nan"), checks)

        lvl = self._level.update(timestamp, x)
        dif = self._diff.update(timestamp, x)
        diff_flag = dif.is_anomaly and not self._prev_diff_raw  # the reading after a difference alarm is the return to normal
        self._prev_diff_raw = dif.is_anomaly
        stk = self._stuck.update(timestamp, x)
        base = lvl.is_anomaly or diff_flag or stk.is_anomaly
        base_reason = "stuck" if stk.is_anomaly else "teda_level" if lvl.is_anomaly else "teda_diff" if diff_flag else ""
        score = max(lvl.score, dif.score, stk.score / self.min_run)

        prof = self._profile.update(timestamp, x, temperature_f=temperature_f, regime=regime, exclude=base)
        expected = prof.expected
        ratio = x / expected if (math.isfinite(expected) and expected != 0) else float("nan")
        pscore = prof.score if math.isfinite(prof.score) else 0.0
        r = ratio if math.isfinite(ratio) else 1.0
        grossly = math.isfinite(ratio) and (ratio > self.scale_ratio or ratio < 1.0 / self.scale_ratio)
        implausible = pscore > self.k_profile or abs(r - 1.0) > self.band or grossly
        wide = implausible or self._prev_implausible
        self._prev_implausible = implausible
        nonsense = not x > 0
        no_expectation = not math.isfinite(expected)

        if nonsense:
            flag, reason = True, "non_positive"
        elif grossly:
            flag, reason = True, "gross_ratio"
        elif base and (stk.is_anomaly or no_expectation or wide):
            flag, reason = True, base_reason
        elif base:
            flag, reason = False, "dropped"
        else:
            flag, reason = False, ""

        if "profile" in self.cross_checks:
            checks["profile"] = float(abs(ratio - 1.0) <= self.profile_band) if math.isfinite(ratio) else float("nan")
        if self.extremes == "preserve" and flag and reason not in ("stuck", "non_positive", "gross_ratio"):
            above = math.isfinite(expected) and x > expected
            used = [checks[name] for name in CHECK_NAMES if name in self.cross_checks]
            if above and not preserve_one(True, True, used):
                flag, reason = False, "extreme_kept"
        return Decision(timestamp, x, float(score), 1.0, bool(flag), reason, expected, checks)

    # ---- series ------------------------------------------------------------------------------------
    def predict(self, values: Any, timestamps: Any = None, context: Any = None) -> DetectionResult:
        v, t = as_series(values, timestamps)
        if t is None:
            raise ValueError("Sentinel needs timestamps (local hour beginning)")
        self.reset()
        return self.replay(v, t, context)

    def get_state(self) -> dict[str, Any]:
        return {
            "level": self._level.get_state(), "diff": self._diff.get_state(), "stuck": self._stuck.get_state(),
            "profile": self._profile.get_state(), "neighbors": self._neighbors.get_state(),
            "weather": self._weather.get_state(),
            "prev_diff_raw": self._prev_diff_raw, "prev_implausible": self._prev_implausible,
        }

    def set_state(self, state: dict[str, Any]) -> Sentinel:
        self.reset()
        self._level.set_state(state["level"])
        self._diff.set_state(state["diff"])
        self._stuck.set_state(state["stuck"])
        self._profile.set_state(state["profile"])
        self._neighbors.set_state(state["neighbors"])
        self._weather.set_state(state["weather"])
        self._prev_diff_raw = bool(state["prev_diff_raw"])
        self._prev_implausible = bool(state["prev_implausible"])
        return self
