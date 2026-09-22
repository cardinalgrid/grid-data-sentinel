"""Consistency checks on a load series and on a forecast published next to it (numpy only).

Two faults that the single-series detectors are not built for:

* ``neighbour_spikes``: an hour whose value is more than ``limit`` (relative) away from the mean of its two
  neighbours. A system-level load does not move by 40 % from one hour to the next and back; when it does in
  the published data, the hour is a telemetry or reporting fault. The first and last hour of the array are
  judged against their single neighbour. Unlike a rolling detector it needs no history and no fitting, so it
  can run on a single day.
* ``forecast_faults``: a published forecast that is missing, non-positive, or more than ``limit`` (relative)
  away from the outcome at some hour. A day-ahead forecast can be wrong by a few percent; one that is off by
  half at a single hour, or reads zero, was not forecast but misreported. These hours make any score of the
  forecast meaningless and are better left out for every model that is compared with it.

Both return boolean masks aligned with the input; ``day_is_plausible`` and ``forecast_is_plausible`` reduce
them to a verdict for a day. The hour-label alignment between the two series is the third check of the family
(``grid_sentinel.alignment``).
"""

from __future__ import annotations

import numpy as np

SPIKE_LIMIT = 0.4
FORECAST_LIMIT = 0.5


def neighbour_spikes(values, limit: float = SPIKE_LIMIT) -> np.ndarray:
    """True at the positions whose value is more than ``limit`` (relative) away from the mean of their
    neighbours; NaN values are neither spikes nor neighbours."""
    a = np.asarray(values, dtype=float)
    out = np.zeros(len(a), dtype=bool)
    for h in range(len(a)):
        if not np.isfinite(a[h]):
            continue
        nb = [a[k] for k in (h - 1, h + 1) if 0 <= k < len(a) and np.isfinite(a[k])]
        if nb:
            ref = float(np.mean(nb))
            out[h] = ref > 0 and abs(a[h] - ref) / ref > limit
    return out


def day_is_plausible(values, limit: float = SPIKE_LIMIT) -> bool:
    """False when the day holds a spike against its neighbours."""
    return not bool(neighbour_spikes(values, limit).any())


def forecast_faults(forecast, actual, limit: float = FORECAST_LIMIT) -> np.ndarray:
    """True at the positions where the published forecast is missing, non-positive, or more than ``limit``
    (relative) away from a positive outcome. Positions without an outcome are judged on the forecast alone."""
    f = np.asarray(forecast, dtype=float)
    a = np.asarray(actual, dtype=float)
    if f.shape != a.shape:
        raise ValueError("forecast and actual must have the same shape")
    bad = ~np.isfinite(f) | (f <= 0)
    both = np.isfinite(a) & (a > 0) & ~bad
    bad[both] = np.abs(f[both] / a[both] - 1.0) > limit
    return bad


def forecast_is_plausible(forecast, actual, limit: float = FORECAST_LIMIT, n_hours: int | None = 24) -> bool:
    """False when the published forecast has a fault at any position, or does not have ``n_hours`` values."""
    f = np.asarray(forecast, dtype=float)
    if n_hours is not None and f.shape != (n_hours,):
        return False
    return not bool(forecast_faults(f, actual, limit).any())


__all__ = ["FORECAST_LIMIT", "SPIKE_LIMIT", "day_is_plausible", "forecast_faults", "forecast_is_plausible", "neighbour_spikes"]
