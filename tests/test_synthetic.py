import numpy as np
import pandas as pd

from grid_sentinel.synthetic import ALL_TYPES, TYPES, inject_anomalies


def test_default_types_unchanged_and_feeder_loss_opt_in(synthetic_year):
    v, t = synthetic_year
    s = pd.Series(v, index=pd.DatetimeIndex(t))
    assert TYPES == ("spike", "dip", "zero", "stuck", "scale") and ALL_TYPES == (*TYPES, "feeder_loss")
    inj = inject_anomalies(s, rate=0.02, seed=0)
    assert "feeder_loss" not in set(inj["kind"])
    inj2 = inject_anomalies(s, rate=0.05, types=("feeder_loss",), seed=1)
    kinds = inj2[inj2["label"] == 1]
    assert (kinds["kind"] == "feeder_loss").all() and len(kinds) > 0
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], inj2["label"].to_numpy(), [0]]))))[::2]
    assert runs.min() >= 6 and runs.max() <= 72
    ratio = (inj2["value"] / inj2["clean"])[inj2["label"] == 1]
    assert ratio.min() >= 0.70 - 1e-9 and ratio.max() <= 0.95 + 1e-9
