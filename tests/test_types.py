import numpy as np
import pytest

from grid_sentinel.types import REASONS, Context, Decision, DetectionResult, as_series


def hourly(start, hours):
    return (np.datetime64(start, "h") + np.arange(hours).astype("timedelta64[h]")).astype("datetime64[ns]")


def test_as_series_accepts_list_and_validates_grid():
    v, t = as_series([1.0, 2.0, 3.0])
    assert v.dtype == np.float64 and t is None
    ts = hourly("2024-01-01T00", 3)
    v, t = as_series([1, 2, 3], ts)
    assert t.dtype == np.dtype("datetime64[ns]") and v.dtype == np.float64
    bad = np.array(["2024-01-01T00", "2024-01-01T01", "2024-01-01T03"], dtype="datetime64[ns]")
    with pytest.raises(ValueError, match="position 2"):
        as_series([1, 2, 3], bad)
    with pytest.raises(ValueError, match="length"):
        as_series([1, 2], ts)
    with pytest.raises(ValueError, match="one-dimensional"):
        as_series([[1, 2], [3, 4]])


def test_as_series_reads_a_pandas_like_index():
    class FakeSeries:
        def __init__(self):
            self.index = hourly("2024-01-01T00", 2)

        def to_numpy(self, dtype=None):
            return np.array([5.0, 6.0])

    v, t = as_series(FakeSeries())
    assert list(v) == [5.0, 6.0] and t is not None and t[1] == np.datetime64("2024-01-01T01", "ns")


def test_context_aligns_to_grid():
    ts = hourly("2024-01-01T00", 48)
    temp_ts = ts[::2]
    temp = np.arange(len(temp_ts), dtype=float)
    ctx = Context(temperature_f=(temp, temp_ts)).aligned(ts)
    assert len(ctx.temperature_f) == 48 and np.isnan(ctx.temperature_f[1]) and ctx.temperature_f[2] == 1.0
    assert ctx.neighbors == {} and ctx.regime is None
    ctx2 = Context(neighbors={"a": (temp * 2, temp_ts)}).aligned(ts)
    assert ctx2.neighbors["a"][4] == 4.0 and ctx2.temperature_f is None


def test_result_reasons_intervals_summary_explain():
    n = 10
    r = DetectionResult(None, np.ones(n), np.zeros(n), np.ones(n), np.zeros(n, bool), np.full(n, "", "<U12"))
    r.is_anomaly[3:6] = True
    r.reason[3:6] = "stuck"
    iv = r.intervals()
    assert list(iv["i0"]) == [3] and list(iv["i1"]) == [5] and list(iv["n"]) == [3]
    s = r.summary()
    assert s["flagged"] == 3 and s["n"] == 10 and s["by_reason"] == {"stuck": 3}
    assert r.explain(4)["reason"] == "stuck" and r.explain(0)["is_anomaly"] is False
    assert set(np.unique(r.reason)) <= set(REASONS) and len(r) == n


def test_decision_is_frozen():
    d = Decision(np.datetime64("2024-01-01T00", "ns"), 1.0, 0.5, 1.0, False, "", float("nan"), {})
    with pytest.raises(AttributeError):
        d.value = 2.0
