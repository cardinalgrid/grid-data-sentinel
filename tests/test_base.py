import numpy as np
import pytest

from grid_sentinel.base import Detector
from grid_sentinel.state import state_from_json, state_to_json
from grid_sentinel.types import Decision


class Dummy(Detector):
    supports_streaming = True

    def __init__(self, k: float = 2.0, window: int = 3):
        self.k = k
        self.window = window
        self.reset()

    def reset(self):
        self.buf = []

    def update(self, timestamp, value, **context):
        self.buf.append(float(value))
        flag = value > self.k
        return Decision(timestamp, float(value), float(value), self.k, bool(flag), "teda_level" if flag else "",
                        float("nan"), {})

    def predict(self, values, timestamps=None, context=None):
        return self.replay(values, timestamps, context)

    def get_state(self):
        return {"buf": np.array(self.buf)}

    def set_state(self, state):
        self.buf = list(state["buf"])
        return self


class BatchOnly(Detector):
    def __init__(self, k: float = 1.0):
        self.k = k

    def predict(self, values, timestamps=None, context=None):
        raise NotImplementedError


def test_params_repr_and_set_params():
    d = Dummy(k=3.0)
    assert d.get_params() == {"k": 3.0, "window": 3}
    assert repr(d) == "Dummy(k=3.0)" and repr(Dummy()) == "Dummy()"
    with pytest.raises(ValueError, match="unknown parameter"):
        d.set_params(zeta=1)
    d.set_params(window=5)
    assert d.window == 5


def test_replay_equals_update_sequence_and_state_roundtrip():
    d = Dummy(k=1.5)
    r = d.predict(np.array([1.0, 2.0, np.nan, 1.0]))
    assert list(r.is_anomaly) == [False, True, False, False]
    assert list(r.reason) == ["", "teda_level", "", ""]
    assert d.buf == [1.0, 2.0, 1.0]  # NaN is skipped, not fed
    js = state_to_json(d.get_state())
    e = Dummy(k=1.5).set_state(state_from_json(js))
    assert e.buf == [1.0, 2.0, 1.0]


def test_fit_and_score_defaults():
    d = Dummy(k=1.5)
    assert d.fit(np.array([1.0, 2.0])) is d and d.buf == [1.0, 2.0]
    assert list(d.score(np.array([0.5, 3.0]))) == [0.5, 3.0]
    assert d.fit_predict(np.array([0.5, 3.0])).is_anomaly[1]


def test_batch_only_detector_refuses_update():
    with pytest.raises(NotImplementedError, match="streaming"):
        BatchOnly().update(None, 1.0)


def test_state_json_handles_arrays_and_datetimes():
    st = {"a": np.arange(3.0), "t": np.datetime64("2024-01-01T05", "ns"), "n": 2, "s": "x", "nested": {"b": np.array([True])}}
    back = state_from_json(state_to_json(st))
    assert np.array_equal(back["a"], st["a"]) and back["a"].dtype == np.float64
    assert back["t"] == st["t"] and back["n"] == 2 and back["s"] == "x" and back["nested"]["b"].dtype == bool
