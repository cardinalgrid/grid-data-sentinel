import numpy as np
import pandas as pd
import pytest

from grid_sentinel import baselines as legacy
from grid_sentinel.detectors import IQR, Hampel, ModifiedZScore, RelativeDeviation, RollingZScore
from grid_sentinel.rolling import rolling_mean, rolling_median, rolling_std, trailing_quantile


def corrupted(synthetic_year):
    v, t = synthetic_year
    v = v.copy()
    v[500] *= 1.8
    v[1500] *= 0.4
    v[2000:2010] *= 10.0
    v[3000] = np.nan
    return v, t


@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("window", [5, 24, 168])
def test_rolling_functions_match_pandas(synthetic_year, center, window):
    v, _ = corrupted(synthetic_year)
    s = pd.Series(v)
    mp = max(3, window // 4)
    for fn, name in ((rolling_mean, "mean"), (rolling_median, "median"), (rolling_std, "std")):
        ours = fn(v, window, min_periods=mp, center=center)
        theirs = getattr(s.rolling(window, min_periods=mp, center=center), name)().to_numpy()
        assert np.allclose(ours, theirs, equal_nan=True), name
    q = trailing_quantile(v, 48, 0.95, min_periods=12)
    theirs = s.rolling(48, min_periods=12).quantile(0.95).to_numpy()
    assert np.allclose(q, theirs, equal_nan=True)


@pytest.mark.parametrize(
    "cls,fn",
    [
        (RollingZScore, legacy.rolling_zscore),
        (Hampel, legacy.hampel),
        (IQR, legacy.iqr),
        (ModifiedZScore, legacy.modified_zscore),
        (RelativeDeviation, legacy.relative_deviation),
    ],
)
def test_baselines_match_the_v03_functions(synthetic_year, cls, fn):
    v, t = corrupted(synthetic_year)
    new = cls().predict(v, t)
    old = fn(v)
    assert np.array_equal(new.is_anomaly, old["is_anomaly"].to_numpy())
    assert np.allclose(new.score, old["score"].to_numpy(), equal_nan=True)
    assert set(np.unique(new.reason[new.is_anomaly])) <= {"baseline"}


def test_modified_zscore_streams(synthetic_year):
    v, t = corrupted(synthetic_year)
    d = ModifiedZScore(window=200)
    batch = d.predict(v, t)
    d.reset()
    stream = np.array([d.update(ti, x).is_anomaly if not np.isnan(x) else False for ti, x in zip(t, v)])
    assert np.array_equal(batch.is_anomaly, stream)


def test_centred_baselines_refuse_streaming():
    with pytest.raises(NotImplementedError, match="streaming"):
        Hampel().update(None, 1.0)
