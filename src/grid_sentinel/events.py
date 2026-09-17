"""The winter events with an official under-forecast figure, and the evaluation window per BA.

The four events are those for which the joint FERC/NERC review of the January 2025 Arctic events
reports the largest day-ahead load under-forecast by a balancing authority (11.8%, 12.5%, 14% and
9.2%). Dates are local calendar days, inclusive, chosen to cover the coldest days of each event.
"""

from __future__ import annotations

import pandas as pd

EVENTS = [
    {"name": "uri", "label": "Winter Storm Uri", "start": "2021-02-13", "end": "2021-02-18"},
    {"name": "elliott", "label": "Winter Storm Elliott", "start": "2022-12-22", "end": "2022-12-26"},
    {"name": "gerri_heather", "label": "Winter Storms Gerri and Heather", "start": "2024-01-12", "end": "2024-01-17"},
    {"name": "january_2025", "label": "January 2025 Arctic events", "start": "2025-01-19", "end": "2025-01-24"},
]


def event_window(series: pd.Series, start: str, end: str, half_hours: int = 36) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    """The instant of the BA's peak inside the event and the hours within ``half_hours`` of it that exist
    in the series (a 72-hour window by default)."""
    s = series.loc[start:f"{end} 23:00"].dropna()
    if s.empty:
        raise ValueError("no readings in the event")
    peak = s.idxmax()
    idx = pd.date_range(peak - pd.Timedelta(hours=half_hours), peak + pd.Timedelta(hours=half_hours), freq="h")
    return peak, idx[idx.isin(series.index)]
