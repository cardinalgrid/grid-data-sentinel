"""Detectors on the ``Detector`` contract."""

from grid_sentinel.detectors.stuck import StuckValues
from grid_sentinel.detectors.teda import TEDA

__all__ = ["TEDA", "StuckValues"]
