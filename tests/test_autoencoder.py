import numpy as np
import pytest

from grid_sentinel.autoencoder import SparseAutoencoder as Legacy
from grid_sentinel.detectors import SparseAutoencoder
from grid_sentinel.state import state_from_json, state_to_json


def corrupted(synthetic_year):
    v, t = synthetic_year
    v = v.copy()
    v[500] *= 1.8
    v[1500] *= 0.4
    v[2000:2010] *= 10.0
    return v[: 24 * 60], t[: 24 * 60]


def test_fit_predict_matches_the_v03_class(synthetic_year):
    v, t = corrupted(synthetic_year)
    kw = {"window": 4, "encoding_dim": 2, "epochs": 20, "factor": 20.0, "seed": 0}
    new = SparseAutoencoder(**kw).fit_predict(v, t)
    old = Legacy(**kw).detect(v, fit=True)
    assert np.array_equal(new.is_anomaly, old["is_anomaly"].to_numpy())
    assert np.allclose(new.score, old["score"].to_numpy(), equal_nan=True)
    assert set(np.unique(new.reason[new.is_anomaly])) <= {"baseline"}


def test_predict_requires_fit_and_streams_with_the_fitted_model(synthetic_year):
    v, t = corrupted(synthetic_year)
    d = SparseAutoencoder(window=4, encoding_dim=2, epochs=20, factor=20.0, seed=0, causal=True)
    with pytest.raises(RuntimeError, match="fit"):
        d.predict(v, t)
    d.fit(v, t)
    batch = d.predict(v, t)
    d._buf.clear()  # start the stream from the first reading, with the fitted model
    stream = np.array([d.update(ti, x).is_anomaly for ti, x in zip(t, v)])
    assert np.array_equal(batch.is_anomaly, stream)  # causal scoring: the window ending at each reading


def test_state_roundtrip_keeps_the_model(synthetic_year):
    v, t = corrupted(synthetic_year)
    d = SparseAutoencoder(window=4, encoding_dim=2, epochs=20, factor=20.0, seed=0).fit(v, t)
    e = SparseAutoencoder(window=4, encoding_dim=2, epochs=20, factor=20.0, seed=0)
    e.set_state(state_from_json(state_to_json(d.get_state())))
    assert np.allclose(d.score(v, t), e.score(v, t), equal_nan=True)
