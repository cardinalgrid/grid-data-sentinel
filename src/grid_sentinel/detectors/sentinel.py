"""The composite detector, one reading at a time.

Base alarms come from the recursive eccentricity detectors on levels and on first differences and from
the frozen-value rule. An alarm is kept only if the reading, or the one before it, is implausible
against the expected profile (residual beyond ``k_profile`` robust standard deviations, more than
``band`` from the expectation, or a ratio outside ``1/scale_ratio``..``scale_ratio``). Frozen runs,
non-positive readings and gross ratios are faults regardless. A kept alarm on a reading *above* the
expectation then faces the cross-checks (profile, neighbours, weather): it stays a fault only if at
least two of the available checks fail to confirm it as genuine, and is otherwise withdrawn as
``extreme_kept``. The expectation is built with base-flagged readings masked out.

Since v0.5 three rules guard the statistics and the extremes (all on by default):

- ``exclude_faults`` / ``exclude_from``: a condemned reading does not update the eccentricity detector on first
  differences (``"diff"``), whose next difference is taken from the expectation instead, so that earlier faults
  do not widen the spread a later one is judged against; the threshold on differences is ``m_diff=4``.
- ``keep_built``: a reading above the expectation whose predecessor already carried at least that share of its
  excess is kept whatever the cross-checks say; a heat wave or a cold morning builds over hours, a fault does not.
- ``jump_guard``: a reading far above the expectation whose predecessor was near it (carried less than that share
  of the excess) is a lone jump, and the cross-checks do not keep it, unless the neighbours' median load stepped up
  in the same hour by at least ``jump_guard_share`` of the BA's own step. Without neighbour readings, the first hour
  of a genuine extreme that arrives as a step is condemned.

``Sentinel.v04()`` returns the v0.4 configuration (and v0.3 behaviour); ``cross_checks=()`` with
``regime="load"``, ``extremes="off"`` and the v0.4 configuration is the v0.2 composite.

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
# reasons that no real demand can produce: with exclude_faults="gross" only these are kept out of the TEDA statistics
GROSS_REASONS = ("non_positive", "gross_ratio", "stuck")


class Sentinel(Detector):
    supports_streaming = True
    handles_missing = True

    def __init__(
        self,
        m_level: float = 4.0,
        m_diff: float = 4.0,
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
        exclude_faults: bool | str = "all",
        exclude_from: str = "diff",
        jump_guard: float | None = 0.5,
        keep_built: float | None = 0.5,
        jump_guard_share: float = 0.5,
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
        if exclude_faults is True:
            exclude_faults = "all"
        if exclude_faults not in (False, None, "all", "gross"):
            raise ValueError("exclude_faults must be False, True, 'all' or 'gross'")
        if exclude_from not in ("both", "level", "diff"):
            raise ValueError("exclude_from must be 'both', 'level' or 'diff'")
        self.exclude_faults = exclude_faults or False
        self.exclude_from = exclude_from
        self.jump_guard = jump_guard
        self.jump_guard_share = float(jump_guard_share)
        self.keep_built = keep_built
        self.reset()

    @classmethod
    def v04(cls, **kw: Any) -> Sentinel:
        """The v0.4 composite: no exclusion of condemned readings, ``m_diff=3``, extremes kept by the cross-checks only."""
        base = {"m_diff": 3.0, "exclude_faults": False, "exclude_from": "both", "jump_guard": None, "keep_built": None}
        return cls(**{**base, **kw})

    def reset(self) -> None:
        self._level = TEDA(m=self.m_level, diff=False, robust=self.robust, half_life_hours=self.half_life_hours)
        self._diff = TEDA(m=self.m_diff, diff=True, robust=self.robust, half_life_hours=self.half_life_hours)
        self._stuck = StuckValues(min_run=self.min_run)
        self._profile = ProfileResidual(k=self.k_profile, regime=self.regime)
        self._neighbors = NeighborCheck(pct=self.neighbor_pct)
        self._weather = WeatherCheck()
        self._prev_diff_raw = False
        self._prev_implausible = False
        self._prev_ratio = float("nan")
        self._prev_x = float("nan")
        self._prev_nb: dict[str, float] = {}

    # ---- one reading -------------------------------------------------------------------------------
    def update(self, timestamp: Any, value: float, temperature_f: float = float("nan"),
               neighbors: dict[str, float] | None = None, regime: float = float("nan"), **context: Any) -> Decision:
        x = float(value)
        checks: dict[str, float] = {}
        if "neighbors" in self.cross_checks:
            checks["neighbors"] = self._neighbors.update(timestamp, neighbors)
        # the median hour-to-hour change of the neighbours, for the jump guard: a step they make too is not a lone jump
        nb_step = float("nan")
        if neighbors:
            steps = [neighbors[k] / self._prev_nb[k] - 1.0 for k in neighbors
                     if k in self._prev_nb and self._prev_nb[k] > 0 and math.isfinite(neighbors[k]) and neighbors[k] > 0]
            if steps:
                steps.sort()
                m = len(steps) // 2
                nb_step = steps[m] if len(steps) % 2 else 0.5 * (steps[m - 1] + steps[m])
            self._prev_nb = {k: float(v) for k, v in neighbors.items() if math.isfinite(v)}
        else:
            self._prev_nb = {}
        if "weather" in self.cross_checks:
            checks["weather"] = self._weather.update(timestamp, temperature_f)
        if math.isnan(x):
            self._stuck.update(timestamp, x)
            self._profile.update(timestamp, x, temperature_f=temperature_f, regime=regime, exclude=True)
            self._prev_diff_raw = False
            self._prev_implausible = False
            self._prev_ratio = float("nan")
            self._prev_x = float("nan")
            return Decision(timestamp, x, 0.0, 1.0, False, "", float("nan"), checks)

        before = (self._level.get_state(), self._diff.get_state()) if self.exclude_faults else None
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
            # a genuine extreme builds over hours; a reading far above the expectation whose predecessor was near it
            # is a lone jump, and the cross-checks (which see the weather, not the jump) do not keep it
            own_step = x / self._prev_x - 1.0 if math.isfinite(self._prev_x) and self._prev_x > 0 else float("nan")
            shared = math.isfinite(nb_step) and math.isfinite(own_step) and own_step > 0 and nb_step >= self.jump_guard_share * own_step
            lone_jump = (self.jump_guard is not None and math.isfinite(ratio) and math.isfinite(self._prev_ratio)
                         and (self._prev_ratio - 1.0) < self.jump_guard * (ratio - 1.0) and not shared)
            # with keep_built, a reading above the expectation whose predecessor already carried at least that share
            # of its excess is kept whatever the cross-checks say: a fault does not build up over hours
            built = (self.keep_built is not None and math.isfinite(ratio) and math.isfinite(self._prev_ratio)
                     and ratio > 1.0 and (self._prev_ratio - 1.0) >= self.keep_built * (ratio - 1.0))
            if above and (built or (not lone_jump and not preserve_one(True, True, used))):
                flag, reason = False, "extreme_kept"
        excluded = flag and (self.exclude_faults == "all" or reason in GROSS_REASONS)
        if before is not None and excluded:
            # a condemned reading does not teach the eccentricity detectors: their statistics go back to what they
            # were, and the next difference is taken from the expectation (or the last good reading), so that the
            # return to normal is not scored as a second jump
            lvl_state, dif_state = before
            if self.exclude_from in ("both", "level"):
                self._level.set_state(lvl_state)
            if self.exclude_from in ("both", "diff"):
                if math.isfinite(expected) and expected > 0:
                    dif_state = {**dif_state, "prev": expected}
                self._diff.set_state(dif_state)
        self._prev_ratio = ratio
        self._prev_x = x
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
            "prev_ratio": self._prev_ratio, "prev_x": self._prev_x, "prev_nb": dict(self._prev_nb),
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
        self._prev_ratio = float(state.get("prev_ratio", float("nan")))
        self._prev_x = float(state.get("prev_x", float("nan")))
        self._prev_nb = {k: float(v) for k, v in state.get("prev_nb", {}).items()}
        return self
