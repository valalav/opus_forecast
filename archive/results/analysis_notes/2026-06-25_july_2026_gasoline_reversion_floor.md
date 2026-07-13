# July 2026 Gasoline Reversion Lower-Bound Scenario

Date saved: 2026-06-25

Forecast question: if gasoline prices in July 2026 return to the level observed
at the start of June 2026, how low can the total CPI index plausibly fall in
July?

## Short Answer

If gasoline in July returns to the 2026-06-01 level, the direct mechanical drag
on total CPI is about **-0.43 percentage points**. From the current working July
forecast of **99.9**, that implies a practical lower bound of about **99.47**,
rounded to **99.5**.

If diesel also returns to the early-June level, the direct motor-fuel drag is
about **-0.46 percentage points**, so a hard downside scenario is around
**99.4-99.5**.

Do not push the July floor materially below **99.4** from fuel alone unless
there is additional evidence of food or non-fuel nonfood deflation.

## Source Data

Primary weekly source:

- `data/Сравнение еженедельных цен_01.csv`

Weights / item references:

- `data/micro_sprav.csv`
- `data/ИПЦПолныйРегион.xlsx`

Related scenario artifacts:

- `archive/results/micro_tariff_scenario_july100_oct110/micro_tariff_scenario_summary.csv`
- `archive/results/tariff_scenario_july100_oct110/scenario_july100_oct110.csv`

## Weekly Fuel Prices Used

The weekly price file had duplicate rows, so the calculation used deduplicated
date-item observations.

| Item | 2026-06-01 price | 2026-06-22 price | July index if returns to 2026-06-01 | July MoM |
|---|---:|---:|---:|---:|
| Бензин автомобильный | 68.96 | 82.03 | 84.07 | -15.93% |
| Бензин АИ-92 | 64.71 | 76.81 | 84.25 | -15.75% |
| Бензин АИ-95 | 69.98 | 84.19 | 83.12 | -16.88% |
| Бензин АИ-98 и выше | 92.70 | 99.10 | 93.54 | -6.46% |
| Дизельное топливо | 73.69 | 83.83 | 87.90 | -12.10% |

## Weight-Based Contribution

Main aggregate gasoline calculation:

```text
Gasoline July MoM if reversion = 68.96 / 82.03 * 100 - 100 = -15.93%
Gasoline weight = 0.02683
Direct headline contribution = -15.93 * 0.02683 = -0.4275 p.p.
```

Detailed gasoline check by AИ grades:

| Item | Weight | July MoM if reversion | Headline contribution |
|---|---:|---:|---:|
| АИ-92 | 0.01023 | -15.75% | -0.1612 p.p. |
| АИ-95 | 0.01543 | -16.88% | -0.2604 p.p. |
| АИ-98 | 0.00117 | -6.46% | -0.0076 p.p. |
| **Detailed gasoline total** | 0.02683 | | **-0.4291 p.p.** |

Diesel adds about **-0.0346 p.p.** if it also returns to the early-June level.

```text
Gasoline + diesel direct contribution = -0.4637 p.p.
```

## Forecast Implication

Current working July forecast before this downside fuel scenario:

```text
July 2026 total CPI = 99.9
```

Gasoline-only reversion:

```text
99.9 - 0.4275 = 99.47
```

Gasoline plus diesel reversion:

```text
99.9 - 0.4637 = 99.44
```

Operational forecast floor:

| Scenario | July 2026 total CPI floor |
|---|---:|
| Gasoline returns to 2026-06-01 level | about 99.5 |
| Gasoline + diesel return to 2026-06-01 level | about 99.4-99.5 |

## How To Use Later

When revisiting the July-August 2026 forecast:

1. Check whether weekly gasoline prices after 2026-06-22 actually revert toward
   the 2026-06-01 level.
2. If reversion is visible, lower the July control point from the baseline
   **99.9** toward **99.5**.
3. Keep August separate: a July fuel correction lowers July directly, but August
   needs new weekly evidence rather than automatic carryover.
4. Combine this note with the tariff assumption: no July regulated-tariff
   indexation, October tariff shock moved later.

## Caveats

- This is a direct mechanical contribution analysis, not a full model rerun.
- It uses weekly operational prices, not official monthly CPI facts.
- It does not include second-round effects, food shocks, or non-fuel nonfood
  movements.
- The lower bound below **99.4** requires additional downside evidence beyond
  gasoline.
