import numpy as np

from grid_sentinel.detectors import FeederLoss


def _series(days: int = 70, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(np.datetime64("2023-01-02T00"), np.datetime64("2023-01-02T00") + np.timedelta64(24 * days, "h"),
                  np.timedelta64(1, "h"))
    h = np.arange(len(t)) % 24
    shape = 1.0 + 0.25 * np.sin((h - 6) / 24 * 2 * np.pi)
    level = 10_000.0 * (1.0 + 0.02 * np.sin(np.arange(len(t)) / (24 * 9)))
    v = level * shape * (1.0 + rng.normal(0, 0.004, len(t)))
    return v, t


def test_flags_a_partial_loss_and_nothing_else():
    v, t = _series()
    i0, n = 24 * 44 + 9, 10  # Wednesday 15 February 2023
    v = v.copy()
    v[i0:i0 + n] *= 0.92
    res = FeederLoss().predict(v, t)
    assert res.is_anomaly[i0:i0 + n].all()
    outside = np.ones(len(v), dtype=bool)
    outside[i0:i0 + n] = False
    assert not res.is_anomaly[outside].any()
    assert set(res.reason[res.is_anomaly]) == {"feeder_loss"}


def test_clean_series_has_no_flags():
    v, t = _series(seed=3)
    assert not FeederLoss().predict(v, t).is_anomaly.any()


def test_gradual_change_is_not_a_loss():
    v, t = _series(seed=4)
    v = v.copy()
    ramp = np.ones(len(v))
    i0 = 24 * 50
    ramp[i0:i0 + 48] = np.linspace(1.0, 0.85, 48)
    ramp[i0 + 48:] = 0.85
    assert not FeederLoss().predict(v * ramp, t).is_anomaly.any()


def test_single_hour_dip_is_left_to_the_other_detectors():
    v, t = _series(seed=5)
    v = v.copy()
    v[24 * 55 + 14] *= 0.85
    assert not FeederLoss().predict(v, t).is_anomaly.any()


def test_loss_without_recovery_is_not_confirmed():
    v, t = _series(seed=6)
    v = v.copy()
    v[24 * 60:] *= 0.9
    assert not FeederLoss().predict(v, t).is_anomaly.any()


def test_loss_next_to_a_holiday_is_not_confirmed():
    from grid_sentinel.detectors.feeder import holiday_days

    days = {str(d) for d in holiday_days([2023])}
    assert {"2023-02-20", "2023-11-23", "2023-12-25", "2023-05-29", "2023-09-04"} <= days
    v, t = _series()
    i0, n = 24 * 50 + 9, 10  # Tuesday 21 February 2023, the day after Presidents' Day
    v = v.copy()
    v[i0:i0 + n] *= 0.92
    assert not FeederLoss().predict(v, t).is_anomaly.any()
    assert FeederLoss(skip_holidays=False).predict(v, t).is_anomaly[i0:i0 + n].all()
