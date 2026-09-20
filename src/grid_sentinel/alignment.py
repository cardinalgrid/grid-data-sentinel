"""Are a forecast and the load it forecasts stamped on the same hour convention?

A published forecast and the load it refers to sometimes come from different systems, and one may be
labelled by hour beginning while the other is labelled by hour ending. The result is a forecast that
looks one hour early or late: its signed error is negative while the load falls and positive while it
rises, and its accuracy looks worse than it is. ``alignment_table`` measures, per period (year by
default), how well the forecast matches the load when it is moved by each candidate shift;
``alignment_check`` reads the table and flags the periods in which a moved forecast beats the published
one by a clear margin. The check needs only the two hourly series and makes no assumption about the
forecaster. It says that the two series disagree on the hour label, not which one is wrong.
"""

from __future__ import annotations

try:
    import pandas as pd
except ImportError as e:  # pragma: no cover
    raise ImportError("grid_sentinel.alignment needs pandas: pip install grid-data-sentinel[pandas]") from e

SHIFTS = (-2, -1, 0, 1, 2)


def _clean(load: pd.Series, forecast: pd.Series) -> tuple[pd.Series, pd.Series]:
    load = load.astype(float)
    forecast = forecast.astype(float)
    load = load[~load.index.duplicated()].sort_index()
    forecast = forecast[~forecast.index.duplicated()].sort_index()
    return load[load > 0], forecast


def shifted_error(load: pd.Series, forecast: pd.Series, shift: int, fault_ratio: float = 0.5) -> pd.Series:
    """Absolute percentage error, per hour, when the forecast value stamped ``shift`` hours away is compared
    with the load: ``shift = -1`` compares the load at ``t`` with the forecast stamped ``t - 1 h``. Hours in
    which the two differ by more than ``fault_ratio`` of the load are dropped as reporting faults."""
    moved = forecast.copy()
    moved.index = moved.index - pd.Timedelta(hours=shift)
    rel = ((moved.reindex(load.index) - load) / load).abs()
    return rel[rel <= fault_ratio].dropna()


def alignment_table(load: pd.Series, forecast: pd.Series, shifts=SHIFTS, freq: str = "YS",
                    fault_ratio: float = 0.5, min_hours: int = 24 * 30) -> pd.DataFrame:
    """MAPE (%) of the forecast against the load for each shift, per period of ``freq`` (calendar year by
    default). Columns ``mape_shift_<k>h``, ``best_shift_h`` (the shift with the lowest MAPE), ``gain``
    (MAPE as published minus the best MAPE, in points) and ``hours`` (hours compared at shift 0). Periods
    with fewer than ``min_hours`` compared hours are left out."""
    load, forecast = _clean(load, forecast)
    cols = {}
    for k in shifts:
        err = shifted_error(load, forecast, k, fault_ratio) * 100.0
        cols[f"mape_shift_{k:+d}h"] = err.resample(freq).mean()
        if k == 0:
            hours = err.resample(freq).count()
    t = pd.DataFrame(cols)
    t["hours"] = hours.reindex(t.index).fillna(0).astype(int)
    t = t[t["hours"] >= min_hours]
    mape_cols = [f"mape_shift_{k:+d}h" for k in shifts]
    best = t[mape_cols].idxmin(axis=1)
    t["best_shift_h"] = best.map({f"mape_shift_{k:+d}h": k for k in shifts}).astype(int)
    t["gain"] = t["mape_shift_+0h"] - t[mape_cols].min(axis=1)
    t.index.name = "period"
    return t


def alignment_check(load: pd.Series, forecast: pd.Series, min_gain: float = 0.3, **kwargs) -> pd.DataFrame:
    """The periods of ``alignment_table`` in which a moved forecast beats the published one by at least
    ``min_gain`` points of MAPE; empty when the two series agree on the hour label throughout."""
    t = alignment_table(load, forecast, **kwargs)
    return t[(t["best_shift_h"] != 0) & (t["gain"] >= min_gain)]


def change_point(load: pd.Series, forecast: pd.Series, shift: int, fault_ratio: float = 0.5,
                 window_days: int = 7) -> pd.Timestamp | None:
    """The first day from which the forecast moved by ``shift`` beats the published one on every day of
    the following ``window_days`` days; None when that never happens. A rough date for when a labelling
    convention changed."""
    load, forecast = _clean(load, forecast)
    a = shifted_error(load, forecast, 0, fault_ratio).resample("D").mean()
    b = shifted_error(load, forecast, shift, fault_ratio).resample("D").mean()
    better = (b < a).reindex(a.index).fillna(False).astype(int)
    run = better.rolling(window_days).sum()
    hits = run[run == window_days]
    if hits.empty:
        return None
    return hits.index[0] - pd.Timedelta(days=window_days - 1)


__all__ = ["SHIFTS", "alignment_check", "alignment_table", "change_point", "shifted_error"]
