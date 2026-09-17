"""The v0.3 function-based API, kept for one minor version with a deprecation warning.

Every name here delegates to the v0.3 implementation unchanged (pandas in, DataFrame out), so that
existing code and the published benchmarks keep their exact behaviour. The replacements are the classes
in ``grid_sentinel`` (``TEDA``, ``StuckValues``, ``ProfileResidual``, ``Sentinel``, ``Repairer``, ...);
see ``docs/migration.md``. Removed in 1.0.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any

try:
    import pandas  # noqa: F401
except ImportError as e:  # pragma: no cover
    raise ImportError("grid_sentinel.legacy needs pandas: pip install grid-data-sentinel[pandas]") from e

from grid_sentinel import autoencoder as _autoencoder
from grid_sentinel import baselines as _baselines
from grid_sentinel import crosscheck as _crosscheck
from grid_sentinel import intervals as _intervals
from grid_sentinel import profile as _profile
from grid_sentinel import repair as _repair
from grid_sentinel import rules as _rules
from grid_sentinel import synthetic as _synthetic
from grid_sentinel import teda as _teda
from grid_sentinel.calendar_us import day_types  # noqa: F401  (unchanged, re-exported for old imports)
from grid_sentinel.metrics import score_labels  # noqa: F401

REPLACEMENTS = {
    "sentinel": "Sentinel(cross_checks=(), regime=None, extremes='off')",
    "sentinel_v2": "Sentinel(cross_checks=(), regime='load', extremes='off')",
    "sentinel_v3": "Sentinel()",
    "stuck_values": "StuckValues",
    "profile_residual": "ProfileResidual",
    "expected_profile": "ProfileResidual (the 'expected' field of its result)",
    "regime_from_peak_hour": "ProfileResidual(regime='load')",
    "regime_from_temperature": "ProfileResidual(regime='temperature')",
    "rolling_zscore": "RollingZScore",
    "hampel": "Hampel",
    "iqr": "IQR",
    "modified_zscore": "ModifiedZScore",
    "relative_deviation": "RelativeDeviation",
    "repair": "Repairer.transform",
    "intervals_from_mask": "intervals_from_mask (numpy structured array)",
    "summarize_intervals": "summarize_intervals (dict)",
    "confirm_neighbors": "detectors.crosscheck.NeighborCheck",
    "confirm_weather": "detectors.crosscheck.WeatherCheck",
    "preserve_extremes": "detectors.crosscheck.preserve_one",
    "inject_anomalies": "inject_anomalies (unchanged, benchmark extra)",
    "RecursiveTEDA": "TEDA",
    "SparseAutoencoder": "SparseAutoencoder (Detector contract: fit, predict, update)",
}


def _deprecated(name: str, fn: Any) -> Any:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"grid_sentinel.{name} is deprecated since 0.4.0 and will be removed in 1.0; use {REPLACEMENTS[name]}",
            DeprecationWarning, stacklevel=2,
        )
        return fn(*args, **kwargs)

    return wrapper


sentinel = _deprecated("sentinel", _rules.sentinel)
sentinel_v2 = _deprecated("sentinel_v2", _rules.sentinel_v2)
sentinel_v3 = _deprecated("sentinel_v3", _rules.sentinel_v3)
stuck_values = _deprecated("stuck_values", _rules.stuck_values)
profile_residual = _deprecated("profile_residual", _profile.profile_residual)
expected_profile = _deprecated("expected_profile", _profile.expected_profile)
regime_from_peak_hour = _deprecated("regime_from_peak_hour", _profile.regime_from_peak_hour)
regime_from_temperature = _deprecated("regime_from_temperature", _profile.regime_from_temperature)
rolling_zscore = _deprecated("rolling_zscore", _baselines.rolling_zscore)
hampel = _deprecated("hampel", _baselines.hampel)
iqr = _deprecated("iqr", _baselines.iqr)
modified_zscore = _deprecated("modified_zscore", _baselines.modified_zscore)
relative_deviation = _deprecated("relative_deviation", _baselines.relative_deviation)
repair = _deprecated("repair", _repair.repair)
intervals_from_mask = _deprecated("intervals_from_mask", _intervals.intervals_from_mask)
summarize_intervals = _deprecated("summarize_intervals", _intervals.summarize_intervals)
confirm_neighbors = _deprecated("confirm_neighbors", _crosscheck.confirm_neighbors)
confirm_weather = _deprecated("confirm_weather", _crosscheck.confirm_weather)
preserve_extremes = _deprecated("preserve_extremes", _crosscheck.preserve_extremes)
inject_anomalies = _deprecated("inject_anomalies", _synthetic.inject_anomalies)


class RecursiveTEDA(_teda.RecursiveTEDA):
    """Deprecated: use ``grid_sentinel.TEDA``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn("grid_sentinel.RecursiveTEDA is deprecated since 0.4.0; use TEDA", DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


class SparseAutoencoder(_autoencoder.SparseAutoencoder):
    """Deprecated v0.3 class with ``detect``: use ``grid_sentinel.SparseAutoencoder`` (fit, predict, update)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn("grid_sentinel.legacy.SparseAutoencoder is deprecated since 0.4.0; use grid_sentinel.SparseAutoencoder",
                      DeprecationWarning, stacklevel=2)
        super().__init__(*args, **kwargs)


__all__ = list(REPLACEMENTS)
