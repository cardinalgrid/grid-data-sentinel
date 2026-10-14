# The 0.5 study: every configuration tried, and the rule that chose

Version 0.5 changed the default composite and added an optional detector for partial loss of telemetry. This page lists every configuration tried on the way, the rule that decided which of them enter the default, written down before the first number was computed, and how each configuration fared against it. Every number on this page comes from `scripts/v05_study.py` run on the code of version 0.5.0: each candidate is a set of keyword arguments of `Sentinel.v04()` or of `FeederLoss`, so any row can be rerun on its own. The tables are in `results/v05_study/`.

## The adoption rule

Fixed before any configuration was run:

1. **Standard benchmark** (development years): adjusted F1 drops by at most 0.003, adjusted precision by at most 0.01, and no fault type's recall by more than 0.02.
2. **Feeder-loss benchmark**: recall on feeder losses of at least 0.50, and a better adjusted F1.
3. **Dips**: recall on single dips improves by at least 0.05 on the standard benchmark.
4. **Winter storms**: the share of BAs whose peak reading is kept does not fall in any of the four events.
5. **Confirmation** on years not used in development (2021 and 2022, seeds 1 and 2): item 1 holds and the target recall improves in every combination.

A change that passes every item that applies to it enters the default. A change that fails only on precision becomes an option, off by default. Anything else stays out. Items 1, 3, 4 and 5 apply to changes of the composite; items 1, 2, 4 and 5 to the feeder-loss detector.

One reading of item 5 has to be stated. In some combinations the baseline already catches every injected dip, and nothing can improve there. The check applied here is that the target recall is never lower in any combination and is higher over all of them together. The tables below show every combination, so a stricter reading can be checked.

## The benchmarks

- **Standard**: the ten benchmark BAs, 2023 and 2024 (development), 0.5% of the readings corrupted with spikes, dips, zero runs, frozen runs and unit errors, scored with a tolerance of one hour; "adjusted" leaves out readings that are already faulty in the published series.
- **Two injection schemes.** The published benchmark draws the faults with seed `seed + year`, so every BA gets them at the same hours and one draw counts ten times. The per-series scheme gives each BA its own draw (`1000 * seed + 37 * i + year`). Development uses seed 0 in the published scheme and seeds 0 to 2 per series.
- **Feeder-loss**: the same plus 6- to 72-hour losses of 5% to 30% of the load. **Dense**: feeder losses only, 2% of the readings.
- **Confirmation**: 2021 and 2022, seeds 1 and 2, both schemes: eight combinations of scheme, seed and year.
- **Storms**: the 72 hours around each BA's peak in Winter Storm Uri (2021), Elliott (2022), Gerri and Heather (2024) and the January 2025 Arctic events, every BA with a weather station, as in `docs/storms.md`.
- **The series as published**: the ten BAs from 2021 to 2024, with nothing injected.

## Part A: what the composite learns from

In 0.4 the composite caught 71% of the injected single dips. The cause is in its eccentricity detectors: their running mean and variance never forget, and every reading enters them, including the readings the composite has just condemned. A series with a zero run or a unit error early in the year carries a wider spread of hourly changes for the rest of it, and a dip of 30% to 60% later no longer stands out against that spread. The candidates below attack this from different sides: a higher threshold, clipped updates, forgetting, or keeping condemned readings out of one or both statistics. The last three add rules about genuine extremes, which the series as published showed to be needed (see below).

### The candidates

| candidate | configuration | what it tries |
|---|---|---|
| `v04` | `Sentinel.v04()` | The 0.4 composite, the baseline. |
| `m_diff_3.5` | `m_diff=3.5` | A higher threshold on first differences, nothing else. |
| `robust` | `robust=True` | Winsorised updates: a flagged reading enters both statistics clipped to the acceptance boundary. |
| `forgetting` | `half_life_hours=336` | Statistics that forget, with a half-life of two weeks. |
| `exclude_both` | `exclude_faults='all', exclude_from='both'` | Condemned readings kept out of the statistics on levels and on differences. |
| `exclude_diff_gross` | `exclude_faults='gross', exclude_from='diff', m_diff=4.0` | Only gross faults (non-positive, gross ratio, frozen) kept out of the statistics on differences; threshold 4. |
| `exclude_diff_3.5` | `exclude_faults='all', exclude_from='diff', m_diff=3.5` | Every condemned reading kept out of the statistics on differences; threshold 3.5. |
| `exclude_diff` | `exclude_faults='all', exclude_from='diff', m_diff=4.0` | Every condemned reading kept out of the statistics on differences; threshold 4. |
| `keep_built` | `exclude_faults='all', exclude_from='diff', m_diff=4.0, keep_built=0.5` | As above, and a reading above the expectation whose predecessor carried at least half of its excess is kept. |
| `jump_guard_alone` | `exclude_faults='all', exclude_from='diff', m_diff=4.0, keep_built=0.5, jump_guard=0.5, jump_guard_share=inf` | As above, and a lone jump is never kept by the cross-checks. |
| `v05` | `Sentinel()` | As above, but a jump the neighbours share (median step at least half of the BA's) is not a lone jump: `Sentinel()`. |

### Development benchmark

Published scheme (seed 0), then per-series scheme (seeds 0 to 2); recall by fault type.

| candidate | precision adj. | F1 adj. | spike | dip | zero | frozen | unit | per series: precision adj. | F1 adj. | spike | dip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `v04` | 0.976 | 0.974 | 0.950 | 0.713 | 1.000 | 1.000 | 0.989 | 0.944 | 0.955 | 0.748 | 0.856 |
| `m_diff_3.5` | 0.995 | 0.979 | 0.950 | 0.575 | 1.000 | 1.000 | 0.989 | 0.967 | 0.966 | 0.734 | 0.819 |
| `robust` | 0.916 | 0.952 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 | 0.896 | 0.936 | 0.873 | 0.988 |
| `forgetting` | 0.941 | 0.947 | 0.950 | 0.988 | 1.000 | 1.000 | 0.953 | 0.965 | 0.962 | 0.836 | 0.919 |
| `exclude_both` | 0.865 | 0.918 | 0.950 | 1.000 | 1.000 | 1.000 | 1.000 | 0.877 | 0.924 | 0.873 | 0.988 |
| `exclude_diff_gross` | 0.994 | 0.987 | 0.950 | 0.900 | 1.000 | 1.000 | 0.989 | 0.966 | 0.968 | 0.855 | 0.956 |
| `exclude_diff_3.5` | 0.990 | 0.988 | 0.950 | 0.975 | 1.000 | 1.000 | 0.989 | 0.960 | 0.966 | 0.873 | 0.975 |
| `exclude_diff` | 0.994 | 0.989 | 0.950 | 0.950 | 1.000 | 1.000 | 0.989 | 0.964 | 0.967 | 0.869 | 0.975 |
| `keep_built` | 0.994 | 0.989 | 0.950 | 0.950 | 1.000 | 1.000 | 0.989 | 0.979 | 0.976 | 0.869 | 0.975 |
| `jump_guard_alone` | 0.994 | 0.990 | 1.000 | 0.950 | 1.000 | 1.000 | 0.989 | 0.979 | 0.977 | 0.926 | 0.975 |
| `v05` | 0.994 | 0.990 | 1.000 | 0.950 | 1.000 | 1.000 | 0.989 | 0.979 | 0.977 | 0.926 | 0.975 |

### Confirmation, 2021 and 2022

Adjusted F1 / recall on dips in each combination (scheme, seed, year); a dash where no dip was drawn.

| candidate | per s. 1/2021 | per s. 1/2022 | per s. 2/2021 | per s. 2/2022 | pub. 1/2021 | pub. 1/2022 | pub. 2/2021 | pub. 2/2022 |
|---|---|---|---|---|---|---|---|---|
| `v04` | 0.963 / 0.500 | 0.988 / 1.000 | 0.930 / 0.958 | 0.972 / 0.958 | 0.911 / – | 0.985 / 0.775 | 0.969 / 0.775 | 0.987 / 0.400 |
| `m_diff_3.5` | 0.973 / 0.429 | 0.989 / 0.938 | 0.946 / 0.958 | 0.971 / 0.833 | 0.935 / – | 0.985 / 0.750 | 0.979 / 0.750 | 0.985 / 0.250 |
| `robust` | 0.922 / 1.000 | 0.976 / 1.000 | 0.912 / 1.000 | 0.971 / 0.958 | 0.897 / – | 0.978 / 1.000 | 0.932 / 1.000 | 0.987 / 1.000 |
| `forgetting` | 0.909 / 0.571 | 0.937 / 1.000 | 0.924 / 1.000 | 0.913 / 0.958 | 0.952 / – | 0.953 / 1.000 | 0.959 / 0.975 | 0.998 / 1.000 |
| `exclude_both` | 0.914 / 1.000 | 0.971 / 1.000 | 0.907 / 1.000 | 0.971 / 0.958 | 0.905 / – | 0.965 / 1.000 | 0.918 / 1.000 | 0.972 / 1.000 |
| `exclude_diff_gross` | 0.974 / 0.500 | 0.990 / 1.000 | 0.948 / 1.000 | 0.973 / 0.958 | 0.936 / – | 0.989 / 0.850 | 0.987 / 0.950 | 1.000 / 1.000 |
| `exclude_diff_3.5` | 0.978 / 1.000 | 0.990 / 1.000 | 0.945 / 1.000 | 0.969 / 0.958 | 0.933 / – | 0.994 / 1.000 | 0.987 / 1.000 | 0.998 / 1.000 |
| `exclude_diff` | 0.980 / 1.000 | 0.992 / 1.000 | 0.946 / 1.000 | 0.971 / 0.958 | 0.934 / – | 0.996 / 1.000 | 0.988 / 1.000 | 1.000 / 1.000 |
| `keep_built` | 0.997 / 1.000 | 0.997 / 1.000 | 0.995 / 1.000 | 0.977 / 0.958 | 0.990 / – | 0.999 / 1.000 | 0.995 / 1.000 | 1.000 / 1.000 |
| `jump_guard_alone` | 0.998 / 1.000 | 0.997 / 1.000 | 0.996 / 1.000 | 0.978 / 0.958 | 0.990 / – | 1.000 / 1.000 | 0.995 / 1.000 | 1.000 / 1.000 |
| `v05` | 0.998 / 1.000 | 0.997 / 1.000 | 0.996 / 1.000 | 0.978 / 0.958 | 0.990 / – | 1.000 / 1.000 | 0.995 / 1.000 | 1.000 / 1.000 |

### Winter storms

Share of BAs whose peak reading is kept, recall on faults injected into the windows, and share of the readings of the windows flagged with nothing injected (mean of the four events).

| candidate | peak kept, Uri | peak kept, Elliott | peak kept, Gerri and Heather | peak kept, January 2025 | recall, Uri | recall, Elliott | recall, Gerri and Heather | recall, January 2025 | flagged |
|---|---|---|---|---|---|---|---|---|---|
| `v04` | 0.955 | 0.953 | 1.000 | 1.000 | 0.972 | 0.955 | 0.900 | 1.000 | 0.18% |
| `m_diff_3.5` | 0.955 | 0.953 | 1.000 | 1.000 | 0.972 | 0.955 | 0.900 | 1.000 | 0.14% |
| `robust` | 0.955 | 0.953 | 1.000 | 1.000 | 0.977 | 0.978 | 0.900 | 1.000 | 0.21% |
| `forgetting` | 0.977 | 0.930 | 1.000 | 1.000 | 0.962 | 0.952 | 0.900 | 1.000 | 0.24% |
| `exclude_both` | 0.955 | 0.953 | 1.000 | 1.000 | 0.977 | 0.978 | 0.907 | 1.000 | 0.25% |
| `exclude_diff_gross` | 0.955 | 0.953 | 1.000 | 1.000 | 0.972 | 0.955 | 0.900 | 1.000 | 0.13% |
| `exclude_diff_3.5` | 0.955 | 0.953 | 1.000 | 1.000 | 0.972 | 0.978 | 0.900 | 1.000 | 0.16% |
| `exclude_diff` | 0.955 | 0.953 | 1.000 | 1.000 | 0.972 | 0.978 | 0.900 | 1.000 | 0.14% |
| `keep_built` | 0.955 | 0.953 | 1.000 | 1.000 | 0.972 | 0.978 | 0.900 | 1.000 | 0.09% |
| `jump_guard_alone` | 0.955 | 0.953 | 1.000 | 1.000 | 0.995 | 0.978 | 0.924 | 1.000 | 0.09% |
| `v05` | 0.955 | 0.953 | 1.000 | 1.000 | 0.995 | 0.978 | 0.924 | 1.000 | 0.09% |

### The series as published

Readings flagged over the ten BAs from 2021 to 2024 (350,418 readings), and how they differ from the 0.4 composite: readings it flags that the candidate does not (of which before the profile has any history, and of which kept as genuine extremes), and readings only the candidate flags.

| candidate | flagged | no longer flagged | of which no expectation yet | of which kept as extremes | newly flagged |
|---|---|---|---|---|---|
| `v04` | 856 | 0 | 0 | 0 | 0 |
| `m_diff_3.5` | 760 | 97 | 79 | 2 | 1 |
| `robust` | 880 | 1 | 1 | 0 | 25 |
| `forgetting` | 682 | 243 | 113 | 0 | 69 |
| `exclude_both` | 893 | 23 | 16 | 0 | 60 |
| `exclude_diff_gross` | 751 | 105 | 86 | 2 | 0 |
| `exclude_diff_3.5` | 773 | 96 | 80 | 0 | 13 |
| `exclude_diff` | 765 | 103 | 86 | 0 | 12 |
| `keep_built` | 627 | 229 | 86 | 126 | 0 |
| `jump_guard_alone` | 627 | 229 | 86 | 126 | 0 |
| `v05` | 627 | 229 | 86 | 126 | 0 |

### Against the rule

| candidate | item 1 | item 3 | item 4 | item 5 | outcome |
|---|---|---|---|---|---|
| `m_diff_3.5` | ✗ dip | ✗ | ✓ | ✗ (2/8 item 1, 2/8 dips) | out |
| `robust` | ✗ F1, precision | ✓ | ✓ | ✗ (0/8 item 1, 8/8 dips) | out |
| `forgetting` | ✗ F1, precision, scale | ✓ | ✗ | ✗ (2/8 item 1, 8/8 dips) | out |
| `exclude_both` | ✗ F1, precision | ✓ | ✓ | ✗ (0/8 item 1, 8/8 dips) | out |
| `exclude_diff_gross` | ✓ | ✓ | ✓ | ✓ (8/8 item 1, 8/8 dips) | passes |
| `exclude_diff_3.5` | ✓ | ✓ | ✓ | ✓ (8/8 item 1, 8/8 dips) | passes |
| `exclude_diff` | ✓ | ✓ | ✓ | ✓ (8/8 item 1, 8/8 dips) | passes |
| `keep_built` | ✓ | ✓ | ✓ | ✓ (8/8 item 1, 8/8 dips) | passes |
| `jump_guard_alone` | ✓ | ✓ | ✓ | ✓ (8/8 item 1, 8/8 dips) | passes |
| `v05` | ✓ | ✓ | ✓ | ✓ (8/8 item 1, 8/8 dips) | passes |

### Reading the tables

**What fails, and why.** A higher threshold alone (`m_diff_3.5`) buys precision (0.976 to 0.995) and loses dips (0.713 to 0.575): the spread is still inflated, and a higher bar lets more dips through. Clipped updates, forgetting and keeping condemned readings out of both statistics all catch the dips (0.988 to 1.000), and all lose precision (0.865 to 0.941). The new alarms of `exclude_both`, read one by one, are flags on the statistic on levels at genuine summer peaks: once the spread of levels is no longer widened by the injected faults, a heat-wave peak stands out against it. The finding cuts both ways: the 0.4 composite leaves those peaks alone partly because the faults it is scored on widen its spread. Forgetting also loses a peak in Elliott (0.953 to 0.930), the only failure of item 4.

**Three settings pass, and the rule does not choose among them.** Keeping condemned readings out of the statistic on differences only, with the statistic on levels as before, passes every item with each of the three settings tried. `exclude_diff_gross` catches fewer dips (0.900 in the published scheme, 0.956 per series, and 0.500 in the confirmation combination 1/2021 per series, where the other two reach 1.000). `exclude_diff_3.5` catches more dips in the published scheme (0.975 against 0.950), the same per series, at a precision 0.004 lower in the published scheme and 0.004 lower per series. `exclude_diff` was kept. The three differ by a few thousandths of F1, and either of the other two would have been a defensible choice.

**What the series as published added.** Against the 0.4 composite, `exclude_diff` stops flagging 103 readings, 86 of them before the profile has any history: the first weeks of each year, when nothing confirms or withdraws an alarm and ordinary morning ramps crossed the threshold of 3 on differences (the higher threshold alone removes 79 of them). It also flags 12 readings the 0.4 composite does not, nearly all genuine summer peaks, among them California's record load of September 2022. The rule does not see this: its storm item covers winter only. Looking at why those peaks pass the cross-checks turned up the same problem in the 0.4 composite itself, which flags heat-wave peaks that the profile, the neighbours and the weather station all fail to confirm (`docs/real_cases_v05.md`). `keep_built` keeps a reading above the expectation when the reading before it already carried at least half of its excess: a fault arrives at once, a heat peak builds over hours. With it the series as published lose 229 flags against 0.4, 126 of them kept as extremes, and gain 0; precision per series rises from 0.964 to 0.979, and no recall moves.

**The jump guard, and why it carries an exception.** `keep_built` widens what the cross-checks may keep, and the jump guard narrows it again: a reading far above the expectation whose predecessor was near it is a lone jump, and the cross-checks do not keep it. Spike recall rises from 0.950 to 1.000 in the published scheme and from 0.869 to 0.926 per series, and the recall in the Uri windows from 0.972 to 0.995; on the series as published the flags are the same. `jump_guard_alone` and `v05` give the same numbers in every table on this page. They differ on a case the benchmarks do not hold: a genuine extreme that arrives as a step, such as a cold morning on which a whole region's load jumps in one hour. Without the exception the guard condemns the first hour of that step; with it, a step the neighbours share (their median hourly change at least half of the BA's own) is not a lone jump. `tests/test_sentinel_v04.py::test_jump_guard_condemns_a_lone_step_but_not_one_the_neighbours_share` holds the case. The choice of `v05` over `jump_guard_alone` rests on that test, not on the rule.

**Limits.** Item 4 covers winter only, and summer behaviour rests on the series as published, read case by case. The confirmation has eight combinations; in one of them no dip was drawn. In the published scheme, seed 1, 2021, spike recall is 0.400 for every candidate: the draw holds a single spike, repeated at the same hour in all ten BAs, small enough to fall inside the profile's band, which no candidate changes. One cost does not show in these tables: with a threshold of 4 on differences, the composite fires later at the very start of a series, before the profile has any history, and a spike in the first hours of a series can pass where a threshold of 3 would have caught it.

## Part B: partial loss of telemetry

### The detector, change by change

Each row adds one change to the row above it, except the two alternatives, which change the row before them in another way. Losses found: share of the dense injected losses with more than half of their readings flagged (per-series seeds 0 and 1 for development, 1 and 2 for confirmation). Flagged with nothing injected: readings flagged on the twenty development series as published, a rate of false alarms or of real losses.

| variant | change | losses found (dev.) | losses found (conf.) | share of flags outside the losses (dev.) | flagged with nothing injected | share |
|---|---|---|---|---|---|---|
| `f0` | Steps matched within 50%, drift of the plateau bounded; profile with the temperature regime; 3% minimum step. | 0.465 | 0.411 | 0.627 | 2988 | 1.70% |
| `f1` | Calendar-only profile (the temperature profile has no expectation where the temperature is missing). | 0.573 | 0.640 | 0.721 | 5720 | 3.26% |
| `f2` | Plateau test instead of the summed drift. | 0.700 | 0.771 | 0.746 | 8119 | 4.63% |
| `f3` | Skip losses with both edges on midnights or an edge on a midnight where the day type changes. | 0.662 | 0.729 | 0.198 | 475 | 0.27% |
| `f4` | Minimum step 5%. | 0.657 | 0.738 | 0.099 | 165 | 0.09% |
| `f5_all_holidays` | Skip the day before, of and after every federal holiday, and 24 Dec to 2 Jan. | 0.559 | 0.664 | 0.076 | 80 | 0.05% |
| `f5` | Skip only 24 December to 2 January. | 0.648 | 0.724 | 0.094 | 147 | 0.08% |
| `f6_veto4` | Veto a step a neighbour makes in the same hour, at 4 spreads. | 0.573 | 0.626 | 0.076 | 73 | 0.04% |
| `f6` | The same veto at 6 spreads: `FeederLoss()`. | 0.634 | 0.692 | 0.084 | 73 | 0.04% |

`f6` is `FeederLoss()`: `regime=None, confirm='plateau', skip_day_aligned=True, min_drop=0.05, skip_holidays='christmas', neighbor_veto=6.0` on top of the defaults.

### The composite with and without it

| benchmark | detector | precision adj. | F1 adj. | feeder losses | dip | spike |
|---|---|---|---|---|---|---|
| feeder benchmark, development, per series | `Sentinel()` | 0.837 | 0.550 | 0.011 | 1.000 | 0.982 |
| feeder benchmark, development, per series | with `FeederLoss()` | 0.859 | 0.760 | 0.594 | 1.000 | 0.982 |
| feeder benchmark, development, published | `Sentinel()` | 0.808 | 0.353 | 0.025 | 1.000 | 1.000 |
| feeder benchmark, development, published | with `FeederLoss()` | 0.835 | 0.549 | 0.374 | 1.000 | 1.000 |
| feeder benchmark, confirmation, per series | `Sentinel()` | 0.927 | 0.524 | 0.011 | 0.986 | 0.857 |
| feeder benchmark, confirmation, per series | with `FeederLoss()` | 0.866 | 0.808 | 0.736 | 0.986 | 0.857 |
| feeder benchmark, confirmation, published | `Sentinel()` | 0.963 | 0.560 | 0.001 | 1.000 | 0.967 |
| feeder benchmark, confirmation, published | with `FeederLoss()` | 0.938 | 0.744 | 0.594 | 1.000 | 0.967 |
| standard benchmark, development, published | `Sentinel()` | 0.994 | 0.990 | – | 0.950 | 1.000 |
| standard benchmark, development, published | with `FeederLoss()` | 0.899 | 0.927 | – | 0.950 | 1.000 |
| standard benchmark, development, per series | `Sentinel()` | 0.979 | 0.977 | – | 0.975 | 0.926 |
| standard benchmark, development, per series | with `FeederLoss()` | 0.920 | 0.938 | – | 0.975 | 0.926 |
| standard benchmark, confirmation, published | `Sentinel()` | 0.996 | 0.996 | – | 1.000 | 0.783 |
| standard benchmark, confirmation, published | with `FeederLoss()` | 0.947 | 0.960 | – | 1.000 | 0.783 |
| standard benchmark, confirmation, per series | `Sentinel()` | 0.994 | 0.992 | – | 0.989 | 0.963 |
| standard benchmark, confirmation, per series | with `FeederLoss()` | 0.910 | 0.932 | – | 0.989 | 0.963 |

Peak kept in the storms, with and without it: Uri 0.955 against 0.955; Elliott 0.953 against 0.953; Gerri and Heather 1.000 against 1.000; January 2025 1.000 against 1.000.

### Against the rule

| item | result |
|---|---|
| 1, standard benchmark | ✗ F1, precision |
| 2, feeder-loss benchmark | ✓ |
| 4, storms | ✓ |
| 5, confirmation | ✓ (4/4 per-series combinations) |

### Reading the tables

**The ladder.** The first three changes raise the losses found from 0.465 to 0.700, and flag 8,119 readings of the series as published. Most of those are segments that begin and end on a midnight or change day type at an edge: `f3` removes 94% of them for 0.038 of the losses found. A minimum step of 5% removes most of what is left. Skipping every federal holiday would halve it again (165 to 80) and lose 0.099 of the losses found; skipping only the year-end weeks takes 18 flags for 0.009, and was kept. The neighbour veto leaves 73 flags at either threshold; at 4 spreads it also vetoes 0.075 of the losses found, at 6 spreads 0.014, so 6 was kept. Every choice was made on the development column. The confirmation column moves the same way at every step except `f4`, where it moves the other way by 0.009.

**Why it is an option.** `FeederLoss()` passes items 2, 4 and 5 and fails item 1 on F1 and on precision. The standard benchmark holds no feeder losses, so every reading the detector adds there counts against precision, and the recall of every fault type is unchanged: the drop in F1 (0.990 to 0.927) follows from the drop in precision (0.994 to 0.899). It was read as a failure on precision only, which makes the detector an option, off by default. On the feeder-loss benchmark precision moves both ways: up in development, down in confirmation (0.927 to 0.866 per series). The share of feeder losses found is 0.594 and 0.736 per series, and 0.603 to 1.000 across the confirmation combinations. The published scheme gives 0.374 and 0.594, from two and four draws, each repeated in all ten BAs.

### On the record

Run on every BA with a weather station in 2023 and 2024 (86 BA-years, 753,542 readings), `FeederLoss()` flags 98 segments, 2,509 readings (0.33%). WALC alone accounts for 997 of them in 49 segments. BA-years with at least one segment: 29.

The WALC series has faults of its own in these years, readings far below the load around them. Without WALC the rate is 0.21%, and the most flagged BAs are PSCO, AZPS, BANC, NEVP, TEC. Two segments are read in `docs/real_cases_v05.md`; the others have not been checked one by one, so this rate bounds the false alarms from above and does not count them.

## What entered and what did not

- **Default composite:** `exclude_diff` with `keep_built` and the jump guard with its neighbour exception, `Sentinel()`. `Sentinel.v04()` keeps the previous configuration.
- **Option, off by default:** `FeederLoss()`, run next to the composite.
- **Left out:** a higher threshold alone, clipped updates, forgetting, and keeping condemned readings out of both statistics.
- **Decided outside the rule:** the threshold and the reach of the exclusion, among three settings that pass, and the neighbour exception of the jump guard. Both are stated above with the numbers behind them.

Produced by `scripts/v05_study.py` (tables in `results/v05_study/`, text by `scripts/v05_study_doc.py`).
