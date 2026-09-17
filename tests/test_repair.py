import numpy as np
import pandas as pd
from grid_sentinel.repairer import Audit, Repairer, intervals_from_mask, summarize_intervals

from grid_sentinel.repair import repair as legacy_repair  # v0.3 function


def gaps(synthetic_year):
    v, t = synthetic_year
    flags = np.zeros(len(v), dtype=bool)
    flags[24 * 40 + 6 : 24 * 40 + 18] = True  # 12 hours
    flags[24 * 100 + 3] = True                # 1 hour
    flags[24 * 200 : 24 * 200 + 60] = True     # 60 hours, beyond max_gap
    return v, t, flags


def test_intervals_and_summary():
    mask = np.zeros(200, dtype=bool)
    mask[10] = True
    mask[50:60] = True
    mask[100:180] = True
    iv = intervals_from_mask(mask)
    assert list(iv["n"]) == [1, 10, 80]
    assert list(iv["duration_class"]) == ["up to 1 h", "up to 1 day", "up to 1 week"]
    summ = summarize_intervals(iv)
    assert summ["up to 1 day"] == {"intervals": 1, "hours": 10}
    assert len(intervals_from_mask(np.zeros(5, dtype=bool))) == 0


def test_repairer_matches_the_v03_function(synthetic_year):
    v, t, flags = gaps(synthetic_year)
    s = pd.Series(v, index=pd.DatetimeIndex(t))
    for method in ("linear", "equivalent_days"):
        fixed, audit = Repairer(method=method, max_gap_hours=48).transform(v, t, flags)
        old, log = legacy_repair(s, flags, method=method, max_gap=48)
        assert np.allclose(fixed, old.to_numpy(), equal_nan=True), method
        assert isinstance(audit, Audit) and len(audit.index) == flags.sum()
        assert list(audit.method) == list(log["method"]) and list(audit.reason) == list(log["reason"])
    assert np.isnan(fixed[24 * 200 : 24 * 200 + 60]).all()
    assert (audit.reason[audit.index >= 24 * 200] == "gap too long").all()


def test_audit_to_pandas(synthetic_year):
    v, t, flags = gaps(synthetic_year)
    _, audit = Repairer().transform(v, t, flags)
    df = audit.to_pandas()
    assert list(df.columns) == ["original", "repaired", "method", "reason"] and len(df) == flags.sum()
    assert df.index[0] == pd.Timestamp(t[24 * 40 + 6])
