# What v0.5 changes on the public record

The same public EIA-930 demand, as published, run through the v0.3/v0.4 composite (`Sentinel.v04()`) and the v0.5 composite (`Sentinel()`), each on the whole year with the BA's temperature and neighbours. Over the ten benchmark BAs from 2021 to 2024, v0.4 flags 856 readings and v0.5 flags 627, none of them new: the readings that leave are morning ramps flagged in the first weeks of each year, before the profile has history, and genuine heat-wave hours. Over every BA with a weather station in 2023 and 2024, v0.4 flags 2,300 and v0.5 1,680; 86 readings are flagged only by v0.5, and on inspection they are faults of the published series, such as the third case below. Produced by `scripts/v05_cases.py`; the counts by `scripts/real_diff.py`.

## CISO, June 2021: a heat wave that v0.4 flagged

The western heat wave of mid-June 2021. Demand runs 30% to 60% above what the calendar profile expects for days, and neither the temperature check (one airport, its own seasonal tail) nor the neighbour check confirms every hour. The v0.4 composite condemns the afternoons; v0.5 keeps them because each hour builds on an hour that was already high.

Flags in the window: v0.4 composite 32, v0.5 composite 0.

![CISO, June 2021: a heat wave that v0.4 flagged](figures/real/v05_ciso_2021-06-14.png)

## NYIS, late June 2021: the same in New York

Four hot days in a row; the readings v0.4 flagged are the peaks of the hottest afternoons.

Flags in the window: v0.4 composite 16, v0.5 composite 0.

![NYIS, late June 2021: the same in New York](figures/real/v05_nyis_2021-06-27.png)

## SRP, May 2023: the 23:00 reading returns to the afternoon level

From May to early July 2023 the 23:00 reading of Salt River Project goes back up to about the level of 17:00 to 18:00 and falls again at midnight, almost every day. This is not demand. The v0.4 composite let it pass because the repeated jump had widened its own spread of hourly changes; v0.5 keeps condemned readings out of that spread.

Flags in the window: v0.4 composite 0, v0.5 composite 5.

![SRP, May 2023: the 23:00 reading returns to the afternoon level](figures/real/v05_srp_2023-05-12.png)

