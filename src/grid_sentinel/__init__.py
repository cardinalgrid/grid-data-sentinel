"""Grid Data Sentinel: anomaly detection and repair for load telemetry."""

from grid_sentinel.autoencoder import SparseAutoencoder
from grid_sentinel.baselines import hampel, iqr, modified_zscore, relative_deviation, rolling_zscore
from grid_sentinel.calendar_us import day_types
from grid_sentinel.intervals import intervals_from_mask, summarize_intervals
from grid_sentinel.metrics import score_labels
from grid_sentinel.profile import (
    expected_profile,
    profile_residual,
    regime_from_peak_hour,
    regime_from_temperature,
)
from grid_sentinel.repair import repair
from grid_sentinel.rules import sentinel, sentinel_v2, stuck_values
from grid_sentinel.synthetic import inject_anomalies
from grid_sentinel.teda import RecursiveTEDA

__version__ = "0.2.0"

__all__ = [
    "RecursiveTEDA",
    "SparseAutoencoder",
    "__version__",
    "day_types",
    "expected_profile",
    "hampel",
    "inject_anomalies",
    "intervals_from_mask",
    "iqr",
    "modified_zscore",
    "profile_residual",
    "regime_from_peak_hour",
    "regime_from_temperature",
    "relative_deviation",
    "repair",
    "rolling_zscore",
    "score_labels",
    "sentinel",
    "sentinel_v2",
    "stuck_values",
    "summarize_intervals",
]
