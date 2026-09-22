import numpy as np
import pytest

from grid_sentinel.consistency import (
    day_is_plausible,
    forecast_faults,
    forecast_is_plausible,
    neighbour_spikes,
)


def test_a_spike_is_flagged_and_a_ramp_is_not():
    day = 1000.0 + 200.0 * np.sin(np.linspace(0, np.pi, 24))
    assert not neighbour_spikes(day).any()
    assert day_is_plausible(day)
    spiked = day.copy()
    spiked[22] = 4500.0
    assert neighbour_spikes(spiked).tolist() == [False] * 22 + [True, False]   # the neighbours of the spike stay clean
    assert not day_is_plausible(spiked)
    assert day_is_plausible(np.linspace(1000.0, 1800.0, 24))                  # 3.5 % an hour
    edge = day.copy()
    edge[23] = day[22] * 1.6
    assert neighbour_spikes(edge)[23]                                          # last hour, single neighbour
    with_gap = spiked.copy()
    with_gap[21] = np.nan                                                      # a missing neighbour is ignored
    assert neighbour_spikes(with_gap)[22]
    assert not neighbour_spikes(with_gap)[21]
    plateau = day.copy()
    plateau[22:] = 4500.0                                                      # a two-hour spike at the end of the day
    assert neighbour_spikes(plateau)[22]                                       # the first hour of the plateau is the jump
    assert not neighbour_spikes(plateau)[:22].any()                            # the hour before it stays clean


def test_forecast_faults_cover_zeros_gaps_and_gross_misses():
    actual = np.full(24, 1000.0)
    assert not forecast_faults(np.full(24, 1030.0), actual).any()
    zeros = np.full(24, 1030.0)
    zeros[1:6] = 0.0
    assert forecast_faults(zeros, actual)[1:6].all()
    assert not forecast_faults(zeros, actual)[6:].any()
    off = np.full(24, 1030.0)
    off[5] = 1600.0
    assert forecast_faults(off, actual)[5]
    missing = np.full(24, 1030.0)
    missing[3] = np.nan
    assert forecast_faults(missing, actual)[3]
    partial = actual.copy()
    partial[7] = np.nan                                                        # no outcome: judged on the forecast alone
    assert not forecast_faults(np.full(24, 1030.0), partial).any()
    with pytest.raises(ValueError):
        forecast_faults(np.full(23, 1.0), actual)


def test_forecast_is_plausible_wants_a_full_day():
    actual = np.full(24, 1000.0)
    assert forecast_is_plausible(np.full(24, 990.0), actual)
    assert not forecast_is_plausible(np.full(23, 990.0), actual[:23])
    assert forecast_is_plausible(np.full(23, 990.0), actual[:23], n_hours=None)
