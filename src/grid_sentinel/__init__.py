"""Grid Data Sentinel: anomaly detection and repair for load telemetry.

Core (numpy only): detectors on one estimator contract (``fit`` / ``predict`` / ``score`` in batch,
``update`` in streaming), a composite ``Sentinel`` that cross-checks genuine extremes, a ``Repairer``
with an audit trail. The ``[pandas]`` extra adds the pandas adapter, the public-data loaders, the
benchmark and the scripts. The v0.3 functions live in ``grid_sentinel.legacy`` with a deprecation warning
and are still reachable as ``grid_sentinel.<name>`` for one minor version.
"""

from __future__ import annotations

import warnings
from typing import Any

from grid_sentinel.base import Detector
from grid_sentinel.calendar_us import day_types
from grid_sentinel.detectors import (
    IQR,
    TEDA,
    Hampel,
    ModifiedZScore,
    ProfileResidual,
    RelativeDeviation,
    RollingZScore,
    Sentinel,
    SparseAutoencoder,
    StuckValues,
)
from grid_sentinel.metrics import score_labels
from grid_sentinel.repairer import Audit, Repairer, intervals_from_mask, summarize_intervals
from grid_sentinel.state import state_from_json, state_to_json
from grid_sentinel.types import Context, Decision, DetectionResult

__version__ = "0.4.0"

__all__ = [
    "IQR",
    "TEDA",
    "Audit",
    "Context",
    "Decision",
    "DetectionResult",
    "Detector",
    "Hampel",
    "ModifiedZScore",
    "ProfileResidual",
    "RelativeDeviation",
    "Repairer",
    "RollingZScore",
    "Sentinel",
    "SparseAutoencoder",
    "StuckValues",
    "__version__",
    "day_types",
    "intervals_from_mask",
    "score_labels",
    "state_from_json",
    "state_to_json",
    "summarize_intervals",
]

_LEGACY = {
    "sentinel", "sentinel_v2", "sentinel_v3", "stuck_values", "profile_residual", "expected_profile",
    "regime_from_peak_hour", "regime_from_temperature", "rolling_zscore", "hampel", "iqr", "modified_zscore",
    "relative_deviation", "repair", "confirm_neighbors", "confirm_weather", "preserve_extremes",
    "inject_anomalies", "RecursiveTEDA",
}


def __getattr__(name: str) -> Any:
    """Reach the v0.3 functions as ``grid_sentinel.<name>`` (deprecated; needs the pandas extra)."""
    if name in _LEGACY:
        from grid_sentinel import legacy

        warnings.warn(f"grid_sentinel.{name} is deprecated since 0.4.0; see grid_sentinel.legacy", DeprecationWarning,
                      stacklevel=2)
        return getattr(legacy, name)
    raise AttributeError(f"module 'grid_sentinel' has no attribute {name!r}")
