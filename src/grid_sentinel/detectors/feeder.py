"""Partial loss of load telemetry: a feeder or a metering point drops out of the total and comes back.

The signature is a step down, a plateau below the expected level, and a step back up of about the same size.
A real change of demand rarely looks like that: weather moves the load gradually. Holidays and the days around
them can make that shape (the profile maps the holiday itself to a weekend day type, not the bridge days), so a
loss that starts or ends between 24 December and 2 January is not confirmed (``skip_holidays="all"`` widens this to
the day before and after every U.S. federal holiday);
nor is one whose step a neighbouring authority makes in the same hour, nor one with an edge on a midnight where
the day type changes, where the profile's expectation jumps from one day group's level to another's. The steps are measured on the hour-to-hour change of the log load minus the hour-to-hour change of
the log expected profile (``ProfileResidual``'s expectation): the shape of the day cancels out and what is left
is small (about 1 % at a large balancing authority), so a drop of a few percent stands out. Its spread is
estimated causally, once a day, as the scaled median absolute deviation of the previous ``scale_days`` days.

An onset is an hour whose step is below ``-k`` spreads and at least ``min_drop`` in size, and not deeper than
``max_drop`` (deeper losses are zeros or unit errors, which other detectors handle). It is confirmed by a step
back up within ``min_len`` to ``max_len`` hours whose size matches the onset within ``match_tol``, with no
drift of the plateau larger than ``drift_tol`` of the step. The readings from the onset to the hour before the
recovery are flagged. Batch only: a recovery is needed to confirm a loss.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from grid_sentinel.base import Detector
from grid_sentinel.calendar_us import day_types
from grid_sentinel.detectors.profile import ProfileResidual
from grid_sentinel.types import REASON_DTYPE, Context, DetectionResult, as_series

MAD_TO_SIGMA = 1.4826


def _daily_scale(u: np.ndarray, days: int, floor: float) -> np.ndarray:
    """Scaled MAD of ``u`` over the previous ``days`` days, recomputed at each day boundary (causal)."""
    n = len(u)
    out = np.full(n, np.nan)
    window = 24 * days
    current = np.nan
    for start in range(0, n, 24):
        past = u[max(0, start - window):start]
        past = past[np.isfinite(past)]
        if len(past) >= 24 * 7:
            med = np.median(past)
            current = max(MAD_TO_SIGMA * float(np.median(np.abs(past - med))), floor)
        out[start:start + 24] = current
    return out


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> np.datetime64:
    """The ``n``-th ``weekday`` (Monday 0) of a month; ``n = -1`` is the last one."""
    first = np.datetime64(f"{year:04d}-{month:02d}-01")
    days = first + np.arange(31)
    days = days[days.astype("datetime64[M]") == first.astype("datetime64[M]")]
    hits = days[((days.astype(np.int64) + 3) % 7) == weekday]  # 1970-01-01 was a Thursday
    return hits[n] if n < 0 else hits[n - 1]


def holiday_days(years, christmas_only: bool = False) -> np.ndarray:
    """U.S. federal holidays (observed on the date itself) of the given years, with the day before and after each,
    and the days from 24 December to 1 January; with ``christmas_only``, only the latter."""
    out = []
    for y in sorted({int(v) for v in years}):
        if christmas_only:
            out.extend(np.datetime64(f"{y:04d}-12-24") + np.arange(9))
            out.append(np.datetime64(f"{y:04d}-01-01"))
            continue
        fixed = [np.datetime64(f"{y:04d}-{md}") for md in ("01-01", "06-19", "07-04", "11-11", "12-25")]
        moving = [_nth_weekday(y, 1, 0, 3), _nth_weekday(y, 2, 0, 3), _nth_weekday(y, 5, 0, -1), _nth_weekday(y, 9, 0, 1),
                  _nth_weekday(y, 10, 0, 2), _nth_weekday(y, 11, 3, 4)]
        for d in fixed + moving:
            out.extend([d - 1, d, d + 1])
        out.extend(np.datetime64(f"{y:04d}-12-24") + np.arange(9))
    return np.unique(np.array(out, dtype="datetime64[D]"))


def _neighbor_steps(values: np.ndarray, days: int) -> np.ndarray:
    """Hour-to-hour change of a neighbour's log load minus the median change at the same hour of the previous seven
    days, in units of its causal spread: a step that the shape of the day does not explain."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lv = np.where(values > 0, np.log(values), np.nan)
    d = np.full(len(lv), np.nan)
    d[1:] = np.diff(lv)
    lags = np.full((7, len(d)), np.nan)
    for k in range(1, 8):
        lags[k - 1, 24 * k:] = d[:-24 * k]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r = d - np.nanmedian(lags, axis=0)
    return r / _daily_scale(r, days, floor=1e-4)


class FeederLoss(Detector):
    supports_streaming = False

    def __init__(
        self,
        k: float = 5.0,
        min_drop: float = 0.05,
        max_drop: float = 0.5,
        min_len: int = 3,
        max_len: int = 96,
        match_tol: float = 0.5,
        drift_tol: float = 0.5,
        scale_days: int = 28,
        regime: str | None = None,
        discount: str = "full",
        skip_day_aligned: bool = True,
        max_gap: int = 3,
        neighbor_veto: float | None = 6.0,
        skip_holidays: bool | str = "christmas",
        confirm: str = "plateau",
        plateau_share: float = 0.8,
    ):
        self.k = float(k)
        self.min_drop = float(min_drop)
        self.max_drop = float(max_drop)
        self.min_len = int(min_len)
        self.max_len = int(max_len)
        self.match_tol = float(match_tol)
        self.drift_tol = float(drift_tol)
        self.scale_days = int(scale_days)
        if discount not in ("full", "shape"):
            raise ValueError("discount must be 'full' or 'shape'")
        self.regime = regime
        self.discount = discount
        self.skip_day_aligned = bool(skip_day_aligned)
        self.max_gap = int(max_gap)
        self.neighbor_veto = neighbor_veto
        if skip_holidays not in (True, False, "all", "christmas"):
            raise ValueError("skip_holidays must be True, False, 'all' or 'christmas'")
        self.skip_holidays = "all" if skip_holidays is True else skip_holidays
        if confirm not in ("drift", "plateau"):
            raise ValueError("confirm must be 'drift' or 'plateau'")
        self.confirm = confirm
        self.plateau_share = float(plateau_share)
        self.reset()

    def steps(self, values: Any, timestamps: Any = None, context: Context | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The step series ``u``, its causal spread and the expected profile."""
        v, ts = as_series(values, timestamps)
        # without a temperature the temperature regime cannot be read: fall back to the calendar-only profile
        regime = self.regime
        if regime == "temperature" and (context is None or context.temperature_f is None):
            regime = None
        prof = ProfileResidual(k=math.inf, regime=regime).predict(v, ts, context)
        expected = prof.expected if prof.expected is not None else np.full(len(v), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            lv = np.where(v > 0, np.log(v), np.nan)
            le = np.where(expected > 0, np.log(expected), np.nan)
        # only the expected shape of the day is discounted: the profile's level is reset at each midnight, and its
        # day-to-day change would otherwise appear as a step at the first hour of the day
        if self.discount == "shape" and ts is not None:
            day = np.asarray(ts).astype("datetime64[D]")
            _, inv = np.unique(day, return_inverse=True)
            sums = np.bincount(inv, weights=np.where(np.isfinite(le), le, 0.0))
            counts = np.bincount(inv, weights=np.isfinite(le).astype(float))
            with np.errstate(invalid="ignore", divide="ignore"):
                le = le - (sums / counts)[inv]
        u = np.full(len(v), np.nan)
        u[1:] = np.diff(lv) - np.diff(le)
        scale = _daily_scale(u, self.scale_days, floor=1e-4)
        return u, scale, expected

    def predict(self, values: Any, timestamps: Any = None, context: Context | None = None) -> DetectionResult:
        v, ts = as_series(values, timestamps)
        n = len(v)
        u, scale, expected = self.steps(v, ts, context)
        # a step that a neighbouring authority makes in the same hour is weather or a wide event, not telemetry
        nb_z = []
        if self.neighbor_veto is not None and context is not None and context.neighbors and ts is not None:
            nb_z = [_neighbor_steps(arr, self.scale_days) for arr in context.aligned(ts).neighbors.values()]

        def shared(i: int, sign: float) -> bool:
            for z in nb_z:
                w = z[max(0, i - 1):i + 2]
                if np.any(np.isfinite(w) & (sign * w > self.neighbor_veto)):
                    return True
            return False

        if self.skip_holidays and ts is not None and n:
            day = np.asarray(ts).astype("datetime64[D]")
            years = np.unique(day.astype("datetime64[Y]").astype(int) + 1970)
            near_holiday = np.isin(day, holiday_days(years, christmas_only=self.skip_holidays == "christmas"))
        else:
            near_holiday = np.zeros(n, bool)
        # the profile resets its expectation at each midnight, and a day group whose level lags (the weekend after a
        # level change) makes a step pair on two midnights; a loss whose both edges fall on a midnight is skipped
        midnight = (np.asarray(ts).astype("datetime64[h]").astype(np.int64) % 24 == 0) if ts is not None else np.zeros(n, bool)
        group_change = np.zeros(n, bool)
        if ts is not None and n > 1:
            dt = day_types(ts)
            group_change[1:] = midnight[1:] & (dt[1:] != dt[:-1])
        flag = np.zeros(n, dtype=bool)
        score = np.zeros(n)
        reason = np.array([""] * n, dtype=object)
        with np.errstate(divide="ignore", invalid="ignore"):
            resid = np.where((v > 0) & (expected > 0), np.log(v) - np.log(expected), np.nan)

        def plateau(t0: int, t1: int, step: float) -> bool:
            # the readings between the steps sit below the line joining the residual before and after, by about
            # the step: robust to a profile that wanders over a long plateau, which a summed drift is not
            pre = resid[max(0, t0 - 6):t0]
            post = resid[t1:t1 + 6]
            if not (np.isfinite(pre).any() and np.isfinite(post).any()):
                return False
            base = np.linspace(np.nanmedian(pre), np.nanmedian(post), t1 - t0)
            depth = resid[t0:t1] - base
            depth = depth[np.isfinite(depth)]
            if len(depth) < 0.5 * (t1 - t0):
                return False
            med = float(np.median(depth))
            low = float(np.mean(depth < -0.5 * step))
            return low >= self.plateau_share and -(1.0 + self.match_tol) * step <= med <= -(1.0 - self.match_tol) * step

        floor_down = math.log(1.0 - self.max_drop)
        min_step = -math.log(1.0 - self.min_drop)
        t = 1
        while t < n:
            ut, st = u[t], scale[t]
            if not (np.isfinite(ut) and np.isfinite(st)) or ut > -max(self.k * st, min_step) or ut < floor_down:
                t += 1
                continue
            found = -1
            drift = 0.0
            gap = 0
            for t1 in range(t + 1, min(n, t + self.max_len + 1)):
                u1 = u[t1]
                # a few missing readings inside the plateau do not end the search; the step across them is not known
                if not np.isfinite(u1):
                    gap += 1
                    if gap > self.max_gap:
                        break
                    continue
                matched = (t1 - t >= self.min_len and u1 > max(self.k * scale[t1], min_step)
                           and abs(u1 + ut) <= self.match_tol * abs(ut))
                if self.confirm == "plateau":
                    if matched and plateau(t, t1, abs(ut)):
                        found = t1
                        break
                    continue
                if matched:
                    if abs(drift) <= self.drift_tol * abs(ut):
                        found = t1
                    break
                drift += u1
                if abs(drift) > abs(ut) * (1.0 + self.drift_tol):
                    break
            if found > 0 and self.skip_day_aligned and ((midnight[t] and midnight[found]) or group_change[t] or group_change[found]):
                found = -1
            if found > 0 and (near_holiday[t] or near_holiday[found]):
                found = -1
            if found > 0 and nb_z and (shared(t, -1.0) or shared(found, 1.0)):
                found = -1
            if found > 0:
                flag[t:found] = True
                score[t:found] = abs(ut) / st
                reason[t:found] = "feeder_loss"
                t = found + 1
            else:
                t += 1
        thr = np.full(n, self.k)
        return DetectionResult(timestamps=ts, value=v, score=score, threshold=thr, is_anomaly=flag,
                               reason=reason.astype(REASON_DTYPE), expected=expected)
