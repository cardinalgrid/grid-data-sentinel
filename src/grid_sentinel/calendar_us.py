"""U.S. day types for load profiles: workday, Saturday, Sunday, with special days mapped to the
weekend profile they resemble. The mapping follows the evidence in Cardinal Grid Note 2: Thanksgiving,
Christmas Day, Christmas Eve, New Year's Eve and the day after Thanksgiving behave like Saturdays;
Memorial Day, Labor Day, Independence Day and New Year's Day like Sundays; MLK Day, Presidents' Day,
Columbus Day and Veterans Day are not special for load. Super Bowl Sunday is treated as a Saturday."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

WORKDAY, SATURDAY, SUNDAY = 0, 1, 2

SUPER_BOWL = {2016: date(2016, 2, 7), 2017: date(2017, 2, 5), 2018: date(2018, 2, 4), 2019: date(2019, 2, 3),
              2020: date(2020, 2, 2), 2021: date(2021, 2, 7), 2022: date(2022, 2, 13), 2023: date(2023, 2, 12),
              2024: date(2024, 2, 11), 2025: date(2025, 2, 9), 2026: date(2026, 2, 8)}


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    if n > 0:
        d = date(year, month, 1)
        return d + timedelta(days=(weekday - d.weekday()) % 7 + 7 * (n - 1))
    d = date(year + (month == 12), (month % 12) + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def special_days(year: int) -> dict[date, int]:
    """Special days of one year mapped to the day type they resemble (SATURDAY or SUNDAY)."""
    out: dict[date, int] = {}

    def add(d: date, kind: int, observe: bool = True) -> None:
        out.setdefault(d, kind)
        if observe:
            o = _observed(d)
            if o != d:
                out.setdefault(o, kind)

    add(date(year, 1, 1), SUNDAY)
    add(_nth_weekday(year, 5, 0, -1), SUNDAY, observe=False)
    add(date(year, 7, 4), SUNDAY)
    add(_nth_weekday(year, 9, 0, 1), SUNDAY, observe=False)
    tg = _nth_weekday(year, 11, 3, 4)
    add(tg, SATURDAY, observe=False)
    add(tg + timedelta(days=1), SATURDAY, observe=False)
    add(date(year, 12, 24), SATURDAY, observe=False)
    add(date(year, 12, 25), SATURDAY)
    add(date(year, 12, 31), SATURDAY, observe=False)
    if year in SUPER_BOWL:
        add(SUPER_BOWL[year], SATURDAY, observe=False)
    return out


def day_types(index: pd.DatetimeIndex) -> np.ndarray:
    """Day type (0 workday, 1 Saturday, 2 Sunday) for each timestamp, special days included."""
    idx = pd.DatetimeIndex(index)
    years = range(int(idx.year.min()), int(idx.year.max()) + 1)
    cal: dict[date, int] = {}
    for y in years:
        cal.update(special_days(y))
    dates = idx.date
    wd = idx.weekday.to_numpy()
    base = np.where(wd < 5, WORKDAY, np.where(wd == 5, SATURDAY, SUNDAY))
    special = np.array([cal.get(d, -1) for d in dates])
    return np.where(special >= 0, special, base)
