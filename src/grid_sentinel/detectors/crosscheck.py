"""Cross-checks that decide whether a high reading is a genuine extreme or a fault, one reading at a time.

Each check answers 1.0 (confirmed as genuine), 0.0 (not confirmed) or NaN (not available).
``preserve_one`` applies the rule: a flagged reading above the expectation stays a fault only if at
least two of the available checks fail to confirm it; with exactly two available it must fail both;
with fewer than two, the original decision stands. Batch results are these updates replayed.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any

import numpy as np

from grid_sentinel.detectors.profile import regime_from_temperature_f


def _finite(values) -> np.ndarray:
    arr = np.fromiter((v for v in values if not math.isnan(v)), dtype=float)
    return arr


class NeighborCheck:
    """The median normalised load of the neighbours at this hour is above its own recent percentile.

    Per neighbour, the reading is divided by the causal mean of the same hour over the previous
    ``window_days`` days (at least 7 of them). The median across neighbours with a value (at least
    ``min_neighbors``) is compared with the ``pct`` percentile of the medians of the previous
    ``recent_days`` days (at least 7 days of them).
    """

    def __init__(self, window_days: int = 14, pct: float = 95.0, recent_days: int = 28, min_neighbors: int = 2):
        self.window_days = int(window_days)
        self.pct = float(pct)
        self.recent_days = int(recent_days)
        self.min_neighbors = int(min_neighbors)
        self.reset()

    def reset(self) -> None:
        self._hist: dict[str, list[deque]] = {}   # neighbour -> 24 deques of past readings at that hour
        self._med: deque = deque(maxlen=24 * self.recent_days)

    def _hour_buffers(self, name: str) -> list[deque]:
        if name not in self._hist:
            self._hist[name] = [deque(maxlen=self.window_days) for _ in range(24)]
        return self._hist[name]

    def update(self, timestamp: Any, neighbors: dict[str, float] | None) -> float:
        hour = int(np.datetime64(timestamp, "ns").astype("datetime64[h]").astype("int64") % 24)
        norms = []
        for name, x in (neighbors or {}).items():
            buf = self._hour_buffers(name)[hour]
            past = _finite(buf)
            if len(past) >= 7 and not math.isnan(x):
                ref = float(np.mean(past))
                norms.append(x / ref if ref != 0 else float("nan"))
            buf.append(float(x))
        norms = [n for n in norms if not math.isnan(n)]
        if len(neighbors or {}) < self.min_neighbors:
            return float("nan")
        med = float(np.median(norms)) if len(norms) >= self.min_neighbors else float("nan")
        past_meds = _finite(self._med)
        thr = float(np.quantile(past_meds, self.pct / 100.0)) if len(past_meds) >= 24 * 7 else float("nan")
        self._med.append(med)
        if math.isnan(med) or math.isnan(thr):
            return float("nan")
        return 1.0 if med > thr else 0.0

    def get_state(self) -> dict[str, Any]:
        return {
            "hist": {name: [np.array(b, dtype=float) for b in bufs] for name, bufs in self._hist.items()},
            "med": np.array(self._med, dtype=float),
        }

    def set_state(self, state: dict[str, Any]) -> NeighborCheck:
        self.reset()
        for name, bufs in state["hist"].items():
            hb = self._hour_buffers(name)
            for h, b in enumerate(bufs):
                for x in b:
                    hb[h].append(float(x))
        for m in state["med"]:
            self._med.append(float(m))
        return self


class WeatherCheck:
    """The hour's temperature is in the station's own seasonal tail for the day's regime.

    On a heating day (yesterday's mean below ``heat_f``) the hour is confirmed when its temperature is
    below the 5th percentile, on a cooling day (above ``cool_f``) above the 95th percentile, of the hourly
    temperatures in the same calendar weeks (+/- ``half_weeks``) of the previous years; with fewer than
    ``min_years`` previous years, of the previous ``fallback_weeks`` weeks. The tail needs a week of readings.
    """

    def __init__(self, half_weeks: int = 3, min_years: int = 2, fallback_weeks: int = 8, heat_f: float = 59.0,
                 cool_f: float = 72.0):
        self.half_weeks = int(half_weeks)
        self.min_years = int(min_years)
        self.fallback_weeks = int(fallback_weeks)
        self.heat_f = float(heat_f)
        self.cool_f = float(cool_f)
        self.reset()

    def reset(self) -> None:
        self._days: dict[int, np.ndarray] = {}   # closed days -> 24 temperatures
        self._cur_ord: int | None = None
        self._cur = np.full(24, np.nan)
        self._first_year: int | None = None
        self._regime = float("nan")
        self._p05 = float("nan")
        self._p95 = float("nan")

    def _open_day(self, ord_: int) -> None:
        if self._cur_ord is not None:
            self._days[self._cur_ord] = self._cur
            prev = self._cur[np.isfinite(self._cur)]
            self._regime = regime_from_temperature_f(float(np.mean(prev)) if len(prev) >= 18 else float("nan"),
                                                     self.heat_f, self.cool_f)
        else:
            self._regime = float("nan")
        self._cur_ord = ord_
        self._cur = np.full(24, np.nan)
        year = int(np.int64(ord_).astype("datetime64[D]").astype("datetime64[Y]").astype(int)) + 1970
        if self._first_year is None:
            self._first_year = year
        self._p05, self._p95 = self._tail(ord_, year)

    def _tail(self, ord_: int, year: int) -> tuple[float, float]:
        if not self._days:
            return float("nan"), float("nan")
        ords = np.fromiter(self._days.keys(), dtype=np.int64, count=len(self._days))
        years_avail = year - self._first_year
        half = 7 * self.half_weeks
        if years_avail >= self.min_years:
            m = np.zeros(len(ords), dtype=bool)
            for y in range(1, years_avail + 1):
                c = ord_ - round(365.25 * y)
                m |= (ords >= c - half) & (ords <= c + half)
        else:
            m = (ords >= ord_ - 7 * self.fallback_weeks) & (ords < ord_)
        if not m.any():
            return float("nan"), float("nan")
        vals = np.concatenate([self._days[int(o)] for o in ords[m]])
        vals = vals[np.isfinite(vals)]
        if len(vals) < 24 * 7:
            return float("nan"), float("nan")
        p05, p95 = np.percentile(vals, [5, 95])
        return float(p05), float(p95)

    def update(self, timestamp: Any, temperature_f: float) -> float:
        ts = np.datetime64(timestamp, "ns")
        ord_ = int(ts.astype("datetime64[D]").astype("int64"))
        hour = int(ts.astype("datetime64[h]").astype("int64") % 24)
        if ord_ != self._cur_ord:
            self._open_day(ord_)
        t = float(temperature_f)
        self._cur[hour] = t
        if math.isnan(t) or math.isnan(self._p05) or math.isnan(self._regime):
            return float("nan")
        cold = self._regime == 0.0 and t < self._p05
        hot = self._regime == 2.0 and t > self._p95
        return 1.0 if (cold or hot) else 0.0

    def get_state(self) -> dict[str, Any]:
        ords = np.array(sorted(self._days), dtype=np.int64)
        return {
            "ords": ords, "temps": np.array([self._days[int(o)] for o in ords], dtype=float).reshape(-1, 24),
            "cur_ord": self._cur_ord, "cur": self._cur, "first_year": self._first_year,
            "regime": self._regime, "p05": self._p05, "p95": self._p95,
        }

    def set_state(self, state: dict[str, Any]) -> WeatherCheck:
        self.reset()
        temps = np.asarray(state["temps"], dtype=float).reshape(-1, 24)
        self._days = {int(o): temps[i] for i, o in enumerate(state["ords"])}
        self._cur_ord = None if state["cur_ord"] is None else int(state["cur_ord"])
        self._cur = np.array(state["cur"], dtype=float)
        self._first_year = None if state["first_year"] is None else int(state["first_year"])
        self._regime, self._p05, self._p95 = (float(state[k]) for k in ("regime", "p05", "p95"))
        return self


def preserve_one(flagged: bool, above: bool, checks: list[float]) -> bool:
    """The flag after the two-of-three rule for one reading (True = stays a fault)."""
    if not (flagged and above):
        return bool(flagged)
    available = sum(1 for c in checks if not math.isnan(c))
    failed = sum(1 for c in checks if c == 0.0)
    if available < 2:
        return True
    return failed >= 2
