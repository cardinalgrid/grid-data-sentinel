"""Detectors on the ``Detector`` contract."""

from grid_sentinel.detectors.autoencoder import SparseAutoencoder
from grid_sentinel.detectors.baselines import IQR, Hampel, ModifiedZScore, RelativeDeviation, RollingZScore
from grid_sentinel.detectors.profile import ProfileResidual
from grid_sentinel.detectors.stuck import StuckValues
from grid_sentinel.detectors.teda import TEDA

__all__ = ["IQR", "TEDA", "Hampel", "ModifiedZScore", "ProfileResidual", "RelativeDeviation", "RollingZScore", "SparseAutoencoder", "StuckValues"]
