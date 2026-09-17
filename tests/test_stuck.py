import numpy as np

from grid_sentinel.detectors import StuckValues


def test_stuck_flags_from_the_min_run_th_reading():
    v = np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    r = StuckValues(min_run=3).predict(v)
    assert list(r.is_anomaly) == [False, False, True, True, False, False, False]
    assert list(r.reason[r.is_anomaly]) == ["stuck", "stuck"]
    assert list(r.score) == [1, 2, 3, 4, 1, 2, 1]


def test_stuck_nan_breaks_the_run_and_stream_equals_batch():
    v = np.array([5.0, 5.0, np.nan, 5.0, 5.0, 5.0])
    d = StuckValues(min_run=3)
    batch = d.predict(v)
    assert list(batch.is_anomaly) == [False, False, False, False, False, True]
    d.reset()
    stream = [d.update(i, x).is_anomaly if not np.isnan(x) else False for i, x in enumerate(v)]
    assert list(batch.is_anomaly) == stream


def test_stuck_state_roundtrip():
    from grid_sentinel.state import state_from_json, state_to_json

    d = StuckValues(min_run=3)
    for i, x in enumerate([7.0, 7.0]):
        d.update(i, x)
    e = StuckValues(min_run=3).set_state(state_from_json(state_to_json(d.get_state())))
    assert e.update(2, 7.0).is_anomaly
