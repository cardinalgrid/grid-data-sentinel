"""Calendar-aware expected profile and residual detector, computed one reading at a time.

The expectation of an hour is the median shape of the same-group days (day type x regime) in the last
``recent_weeks`` weeks and in a window of ``half_days`` around the same date one year earlier, scaled by
the median level of the last ``level_days`` same-group days of the last four weeks. Everything is causal:
a day's expectation is fixed when its first reading arrives, from the days already closed, so that the
batch result is exactly the stream result replayed. The residual is scaled by the median absolute
residual of the previous ``mad_window_days`` days (shifted by one reading) and flagged beyond ``k``.

``regime``: ``"load"`` (yesterday's peak before noon means a heating morning), ``"temperature"`` (three
classes from yesterday's mean temperature, given per reading as ``temperature_f``), or ``None``. A regime
label may also be passed per reading as ``regime``; it then overrides the rule for that day.
``exclude=True`` on a reading keeps it out of the day buffer (a fault the caller already flagged) while
still scoring it against the expectation.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from grid_sentinel.base import Detector
from grid_sentinel.calendar_us import day_types
from grid_sentinel.types import Decision, DetectionResult, as_series

MAD_TO_SIGMA = 1.4826
HEAT_F, COOL_F = 59.0, 72.0


def regime_from_temperature_f(tmean_f: float, heat_f: float = HEAT_F, cool_f: float = COOL_F) -> float:
    """0 heating, 1 mild, 2 cooling; NaN without a temperature."""
    if math.isnan(tmean_f):
        return float("nan")
    return 0.0 if tmean_f < heat_f else 2.0 if tmean_f > cool_f else 1.0


class _Day:
    __slots__ = ("group", "masked", "raw", "regime", "temp")

    def __init__(self) -> None:
        self.masked = np.full(24, np.nan)
        self.raw = np.full(24, np.nan)
        self.temp = np.full(24, np.nan)
        self.regime = float("nan")
        self.group = float("nan")


class ProfileResidual(Detector):
    supports_streaming = True
    handles_missing = True

    def __init__(
        self,
        k: float = 4.0,
        recent_weeks: int = 2,
        analog_years: int = 1,
        half_days: int = 21,
        min_ref: int = 2,
        level_days: int = 5,
        mad_window_days: int = 28,
        regime: str | None = "load",
    ):
        if regime not in ("load", "temperature", None):
            raise ValueError("regime must be 'load', 'temperature' or None")
        self.k = float(k)
        self.recent_weeks = int(recent_weeks)
        self.analog_years = int(analog_years)
        self.half_days = int(half_days)
        self.min_ref = int(min_ref)
        self.level_days = int(level_days)
        self.mad_window_days = int(mad_window_days)
        self.regime = regime
        self.reset()

    # ---- state ------------------------------------------------------------------------------------
    def reset(self) -> None:
        self._ords: list[int] = []            # closed days, in order
        self._shapes: list[np.ndarray] = []   # 24 values / day level (NaN where missing)
        self._levels: list[float] = []
        self._groups: list[float] = []
        self._valid: list[bool] = []
        self._cur: _Day | None = None
        self._cur_ord: int | None = None
        self._exp_shape = np.full(24, np.nan)
        self._exp_level = float("nan")
        self._mad: deque = deque(maxlen=24 * self.mad_window_days)

    def get_state(self) -> dict[str, Any]:
        cur = self._cur
        return {
            "ords": np.array(self._ords, dtype=np.int64),
            "shapes": np.array(self._shapes, dtype=float).reshape(-1, 24),
            "levels": np.array(self._levels, dtype=float),
            "groups": np.array(self._groups, dtype=float),
            "valid": np.array(self._valid, dtype=bool),
            "cur_ord": self._cur_ord,
            "cur_masked": None if cur is None else cur.masked,
            "cur_raw": None if cur is None else cur.raw,
            "cur_temp": None if cur is None else cur.temp,
            "cur_regime": None if cur is None else cur.regime,
            "cur_group": None if cur is None else cur.group,
            "exp_shape": self._exp_shape,
            "exp_level": self._exp_level,
            "mad": np.array(self._mad, dtype=float),
        }

    def set_state(self, state: dict[str, Any]) -> ProfileResidual:
        self.reset()
        self._ords = [int(o) for o in state["ords"]]
        self._shapes = [np.array(s, dtype=float) for s in np.asarray(state["shapes"], dtype=float).reshape(-1, 24)]
        self._levels = [float(x) for x in state["levels"]]
        self._groups = [float(x) for x in state["groups"]]
        self._valid = [bool(x) for x in state["valid"]]
        self._cur_ord = None if state["cur_ord"] is None else int(state["cur_ord"])
        if self._cur_ord is not None:
            self._cur = _Day()
            self._cur.masked = np.array(state["cur_masked"], dtype=float)
            self._cur.raw = np.array(state["cur_raw"], dtype=float)
            self._cur.temp = np.array(state["cur_temp"], dtype=float)
            self._cur.regime = float(state["cur_regime"])
            self._cur.group = float(state["cur_group"])
        self._exp_shape = np.array(state["exp_shape"], dtype=float)
        self._exp_level = float(state["exp_level"])
        for x in state["mad"]:
            self._mad.append(float(x))
        return self

    # ---- day bookkeeping ---------------------------------------------------------------------------
    def _close_day(self) -> None:
        d = self._cur
        if d is None:
            return
        finite = np.isfinite(d.masked)
        level = float(np.nanmean(d.masked)) if finite.any() else float("nan")
        with np.errstate(invalid="ignore", divide="ignore"):
            shape = d.masked / level
        self._ords.append(self._cur_ord)
        self._shapes.append(shape)
        self._levels.append(level)
        self._groups.append(d.group)
        self._valid.append(bool(math.isfinite(level) and finite.sum() >= 20))

    def _regime_for_new_day(self, override: float) -> float:
        if not math.isnan(override):
            return float(override)
        prev = self._cur
        if self.regime is None or prev is None:
            return 0.0 if self.regime is None else float("nan")
        if self.regime == "temperature":
            t = prev.temp[np.isfinite(prev.temp)]
            return regime_from_temperature_f(float(np.mean(t)) if len(t) >= 18 else float("nan"))
        ok = np.isfinite(prev.raw).sum() >= 20
        if not ok:
            return float("nan")
        peak = int(np.argmax(np.where(np.isfinite(prev.raw), prev.raw, -np.inf)))
        return 1.0 if peak <= 11 else 0.0

    def _open_day(self, ord_: int, regime_override: float) -> None:
        reg = self._regime_for_new_day(regime_override)
        dtype = int(day_types(np.array([ord_], dtype="int64").astype("datetime64[D]"))[0])
        group = float("nan") if math.isnan(reg) else (dtype if self.regime is None else dtype * 10 + reg)
        self._close_day()
        self._cur = _Day()
        self._cur_ord = ord_
        self._cur.regime = reg
        self._cur.group = group
        self._exp_shape, self._exp_level = self._expectation(ord_, group)

    def _expectation(self, ord_: int, group: float) -> tuple[np.ndarray, float]:
        if math.isnan(group) or not self._ords:
            return np.full(24, np.nan), float("nan")
        ords = np.array(self._ords)
        groups = np.array(self._groups)
        valid = np.array(self._valid)
        same = (groups == group) & valid
        span = 7 * self.recent_weeks
        m = (ords >= ord_ - span) & (ords < ord_)
        for y in range(1, self.analog_years + 1):
            c = ord_ - round(365.25 * y)
            m |= (ords >= c - self.half_days) & (ords <= c + self.half_days)
        m &= same
        shape = np.full(24, np.nan)
        if m.sum() >= self.min_ref:
            with np.errstate(all="ignore"):
                shape = np.nanmedian(np.array([self._shapes[i] for i in np.flatnonzero(m)]), axis=0)
        recent = (ords >= ord_ - 28) & (ords < ord_) & same
        level = float("nan")
        if recent.any():
            level = float(np.median(np.array(self._levels)[recent][-self.level_days :]))
        return shape, level

    # ---- contract ----------------------------------------------------------------------------------
    def update(self, timestamp: Any, value: float, temperature_f: float = float("nan"), regime: float = float("nan"),
               exclude: bool = False, **context: Any) -> Decision:
        ts = np.datetime64(timestamp, "ns")
        ord_ = int(ts.astype("datetime64[D]").astype("int64"))
        hour = int(ts.astype("datetime64[h]").astype("int64") % 24)
        if ord_ != self._cur_ord:
            self._open_day(ord_, float(regime))
        x = float(value)
        d = self._cur
        d.raw[hour] = x
        d.masked[hour] = np.nan if exclude else x
        d.temp[hour] = float(temperature_f)
        expected = float(self._exp_shape[hour] * self._exp_level)
        resid = x - expected
        finite = np.fromiter((m for m in self._mad if not math.isnan(m)), dtype=float)
        scale = float(np.median(finite)) * MAD_TO_SIGMA if len(finite) >= 48 else float("nan")
        with np.errstate(all="ignore"):
            score = abs(resid / scale) if scale > 0 else float("nan")
        self._mad.append(abs(resid))
        flag = bool(np.nan_to_num(score, nan=0.0) > self.k)
        return Decision(timestamp, x, float(score), self.k, flag, "profile" if flag else "", expected, {})

    def predict(self, values: Any, timestamps: Any = None, context: Any = None) -> DetectionResult:
        v, t = as_series(values, timestamps)
        if t is None:
            raise ValueError("ProfileResidual needs timestamps (local hour beginning)")
        self.reset()
        return self.replay(v, t, context)
