# Micro_SM + weekly fact nowcast

Generated: 2026-06-08T14:37:00

## Setup

- Target month: `2026-05-01`
- Weekly fact window: `2026-04-27` -> `2026-05-25`
- Micro items used: `537`
- Micro weight sum: `0.99351`
- Matched weekly items: `90`
- Matched weight sum: `0.31684`

## Headline Result

| Variant | Index | MoM, p.p. |
|---|---:|---:|
| Direct Micro_SM item_code=1 | 100.134 | 0.134 |
| Micro_SM weighted micro aggregate | 100.260 | 0.260 |
| Micro_SM with weekly facts for matched items | 100.142 | 0.142 |

Hybrid minus weighted Micro_SM baseline: `-0.118` p.p.

## YoY Context

| Variant | YoY index |
|---|---:|
| Send form current | 104.86 |
| Weighted Micro_SM baseline for target month | 105.03 |
| Hybrid weekly-fact target month | 104.91 |

## Largest Upward Weekly Replacements

| Позиция | Вес | Модель | Факт weekly | Δ вклад, п.п. |
|---|---:|---:|---:|---:|
| Огурцы свежие, кг | 0.0047 | 59.30 | 68.48 | +0.0430 |
| Сосиски, сардельки, кг | 0.0055 | 99.98 | 106.41 | +0.0355 |
| Колбаса вареная, кг | 0.0046 | 100.45 | 107.86 | +0.0344 |
| Крупа гречневая-ядрица, кг | 0.0034 | 99.58 | 107.41 | +0.0264 |
| Масло сливочное, кг | 0.0087 | 97.76 | 100.79 | +0.0264 |
| Сметана, кг | 0.0075 | 99.69 | 102.85 | +0.0236 |
| Колбаса полукопченая и варено-копченая, кг | 0.0030 | 100.71 | 108.11 | +0.0220 |
| Яйца куриные, 10 шт. | 0.0054 | 80.32 | 83.85 | +0.0191 |
| Сахар-песок, кг | 0.0053 | 100.22 | 102.00 | +0.0095 |
| Печенье, кг | 0.0055 | 100.86 | 102.32 | +0.0080 |
| Капуста белокочанная свежая, кг | 0.0024 | 121.16 | 124.48 | +0.0078 |
| Сыры твердые, полутвердые и мягкие, кг | 0.0051 | 99.02 | 100.54 | +0.0078 |

## Largest Downward Weekly Replacements

| Позиция | Вес | Модель | Факт weekly | Δ вклад, п.п. |
|---|---:|---:|---:|---:|
| Картофель, кг | 0.0061 | 119.51 | 99.14 | -0.1247 |
| Говядина (кроме бескостного мяса), кг | 0.0163 | 102.18 | 98.90 | -0.0535 |
| Морковь, кг | 0.0021 | 119.25 | 103.75 | -0.0326 |
| Лук репчатый, кг | 0.0024 | 116.67 | 106.46 | -0.0241 |
| Помидоры свежие, кг | 0.0057 | 74.24 | 70.29 | -0.0227 |
| Яблоки, кг | 0.0052 | 106.06 | 101.90 | -0.0214 |
| Щетка зубная, шт. | 0.0017 | 100.65 | 92.43 | -0.0143 |
| Свинина (кроме бескостного мяса), кг | 0.0048 | 100.55 | 97.64 | -0.0140 |
| Свёкла столовая, кг | 0.0008 | 117.95 | 100.60 | -0.0134 |
| Мука пшеничная, кг | 0.0042 | 100.59 | 98.42 | -0.0092 |
| Электропылесос напольный, шт. | 0.0019 | 99.14 | 94.70 | -0.0082 |
| Стрижка модельная в женском зале, стрижка | 0.0055 | 101.16 | 100.00 | -0.0064 |

## 12-Month Micro Path

| Month | Micro_SM index | Hybrid index | Replaced items |
|---|---:|---:|---:|
| 2026-05-01 | 100.260 | 100.142 | 90 |
| 2026-06-01 | 99.986 | 99.986 | 0 |
| 2026-07-01 | 100.136 | 100.136 | 0 |
| 2026-08-01 | 99.934 | 99.934 | 0 |
| 2026-09-01 | 100.823 | 100.823 | 0 |
| 2026-10-01 | 100.845 | 100.845 | 0 |
| 2026-11-01 | 100.889 | 100.889 | 0 |
| 2026-12-01 | 100.940 | 100.940 | 0 |
| 2027-01-01 | 101.217 | 101.217 | 0 |
| 2027-02-01 | 100.871 | 100.871 | 0 |
| 2027-03-01 | 100.895 | 100.895 | 0 |
| 2027-04-01 | 100.731 | 100.731 | 0 |

## Notes

- Weekly replacements are exact normalized-name matches only.
- The weekly fact is a price-level bridge from the last available April week to
  the last available May week, not an official monthly CPI fact.
- Unmatched micro items remain at their Micro_SM statsmodels forecast.
- Aggregation uses active `Item_type=5` micro positions and `weight_vertical`;
  their latest weight sum is `0.99351`, so the weighted
  index is normalized by the included weight sum.
