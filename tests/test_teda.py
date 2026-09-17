import numpy as np

from grid_sentinel.detectors import TEDA
from grid_sentinel.state import state_from_json, state_to_json
from grid_sentinel.teda import RecursiveTEDA  # the v0.3 class, kept until legacy.py takes over


def test_teda_matches_the_v03_class_in_batch(synthetic_year):
    v, t = synthetic_year
    v = v.copy()
    v[500] *= 1.8
    v[900:903] = v[899]
    v[1200] = np.nan
    for diff in (False, True):
        new = TEDA(m=4.0, diff=diff).predict(v, t)
        old = RecursiveTEDA(m=4.0, diff=diff).detect(v)
        assert np.array_equal(new.is_anomaly, old["is_anomaly"].to_numpy())
        assert np.allclose(new.score, old["score"].to_numpy(), equal_nan=True)
        assert set(np.unique(new.reason[new.is_anomaly])) <= {"teda_diff" if diff else "teda_level"}


def test_teda_stream_equals_batch(synthetic_year):
    v, t = synthetic_year
    v = v.copy()
    v[700] *= 2.0
    d = TEDA(m=4.0)
    batch = d.predict(v, t)
    d.reset()
    stream = np.array([d.update(ti, x).is_anomaly for ti, x in zip(t, v)])
    assert np.array_equal(batch.is_anomaly, stream)


def test_teda_state_roundtrip_mid_series(synthetic_year):
    v, t = synthetic_year
    d = TEDA(m=4.0, diff=True)
    full = d.predict(v, t)
    d.reset()
    half = len(v) // 2
    for i in range(half):
        d.update(t[i], v[i])
    e = TEDA(m=4.0, diff=True).set_state(state_from_json(state_to_json(d.get_state())))
    rest = np.array([e.update(t[i], v[i]).is_anomaly for i in range(half, len(v))])
    # the "return to normal" rule is applied by predict on the whole series; compare raw update flags
    d2 = TEDA(m=4.0, diff=True)
    raw = np.array([d2.update(t[i], v[i]).is_anomaly for i in range(len(v))])
    assert np.array_equal(rest, raw[half:])
    assert full.is_anomaly.sum() <= raw.sum()


def test_forgetting_tracks_a_level_shift_and_keeps_spikes():
    rng = np.random.default_rng(0)
    v = rng.normal(100, 1, 2000)
    v[1000:] += 30
    v[1500] += 15
    r = TEDA(m=4.0, half_life_hours=72).predict(v)
    assert r.is_anomaly[1500]  # a 15-sigma spike 500 h after the shift is seen again
    assert r.is_anomaly[1000:1300].sum() <= 10  # the shift itself costs a handful of flags, then the detector adapts
    assert r.is_anomaly[1300:1499].sum() == 0
    none = TEDA(m=4.0).predict(v)
    assert r.score[1500] > none.score[1500]  # without forgetting the variance inflated by the shift hides the spike
    assert not none.is_anomaly[1500]
