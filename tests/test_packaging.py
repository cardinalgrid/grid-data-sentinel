import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import grid_sentinel
from grid_sentinel import Sentinel
from grid_sentinel.pandas import context_from, from_series, predict_series, regularize


def test_core_imports_without_pandas():
    code = (
        "import sys; sys.modules['pandas'] = None; sys.modules['requests'] = None\n"
        "import numpy as np\n"
        "import grid_sentinel\n"
        "from grid_sentinel import Sentinel, TEDA, StuckValues, Repairer, Context\n"
        "t = (np.datetime64('2024-01-01T00','h') + np.arange(24*40).astype('timedelta64[h]')).astype('datetime64[ns]')\n"
        "v = 100 + 10*np.sin(np.arange(24*40)/24*2*np.pi)\n"
        "r = Sentinel(cross_checks=(), regime=None, extremes='off').predict(v, t)\n"
        "fixed, audit = Repairer().transform(v, t, r.is_anomaly)\n"
        "print(len(r), grid_sentinel.__version__)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("0.4.0")
    code2 = "import sys; sys.modules['pandas'] = None\nimport grid_sentinel.benchmark\n"
    out2 = subprocess.run([sys.executable, "-c", code2], capture_output=True, text=True, check=False)
    assert out2.returncode != 0 and "grid-data-sentinel[pandas]" in out2.stderr


def test_legacy_names_warn_and_still_work(synthetic_year):
    v, t = synthetic_year
    s = pd.Series(v, index=pd.DatetimeIndex(t))
    with pytest.warns(DeprecationWarning, match="deprecated"):
        old = grid_sentinel.sentinel_v2(s)
    assert "confirmed_by" in old.columns
    from grid_sentinel.legacy import RecursiveTEDA

    with pytest.warns(DeprecationWarning):
        RecursiveTEDA(m=4.0)
    with pytest.raises(AttributeError):
        grid_sentinel.no_such_name  # noqa: B018


def test_pandas_adapter_round_trip(synthetic_year):
    v, t = synthetic_year
    s = pd.Series(v, index=pd.DatetimeIndex(t))
    s_gappy = s.drop(s.index[100:110])
    reg = regularize(s_gappy)
    assert len(reg) == len(s) and reg.iloc[100:110].isna().all()
    vv, tt = from_series(reg)
    assert vv.dtype == np.float64 and tt.dtype == np.dtype("datetime64[ns]")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = predict_series(Sentinel(cross_checks=(), regime="load", extremes="off"), s_gappy)
    assert list(df.columns[:5]) == ["value", "score", "threshold", "is_anomaly", "reason"] and len(df) == len(s)
    ctx = context_from(temperature_f=pd.Series(np.full(len(t), 40.0), index=pd.DatetimeIndex(t)))
    assert ctx.aligned(t).temperature_f[0] == 40.0
