# External Code Repository Deep Dive

Date: 2026-06-25

Source folder:

```text
experiments/code_repository_20260625/
```

This note records the first direct code review after unpacking the external
repository. It focuses on code that can plausibly improve SIRENA-KBR rather
than on catalog completeness.

## Bottom Line

Most useful for immediate work:

1. `weekCpiR` weekly-to-monthly mechanics.
2. Khabarovsk/Omsk 45-component bottom-up reconstruction.
3. Mordovia Python component `auto_arima` loop as a simple baseline only.
4. Volgograd variable-selection methodology as a diagnostic gate.

Not useful as direct production code:

- pre-trained external CatBoost/R objects;
- full-sample STL/X-13 smoothing without rolling vintages;
- ARIMAX with externally supplied future regressors unless those paths are
  deterministic and cutoff-safe;
- ML ensemble weighting that uses the same realized holdout errors being
  reported.

## 1. Weekly-to-Monthly Nowcast

Primary files:

- `experiments/code_repository_20260625/nested_extracted/a24cd898db7c9/5 Недельная инфляция/weekCpiR/R/scripts/main.R`
- `experiments/code_repository_20260625/nested_extracted/a24cd898db7c9/5 Недельная инфляция/weekCpiR/R/common/data_cpi_week_wow_nowcast.R`
- `experiments/code_repository_20260625/nested_extracted/a24cd898db7c9/5 Недельная инфляция/weekCpiR/R/common/data_cpi_week_bi_calculate.R`
- `experiments/code_repository_20260625/nested_extracted/a24cd898db7c9/5 Недельная инфляция/weekCpiR/R/common/data_cpi_week_bi_bias_correct.R`
- `experiments/code_repository_20260625/nested_extracted/a24cd898db7c9/5 Недельная инфляция/weekCpiR/R/common/data_cpi_month_mom_from_wow_calculate.R`
- `experiments/code_repository_20260625/nested_extracted/a24cd898db7c9/5 Недельная инфляция/weekCpiR/R/common/modified_laspeyres_by_sum.R`

### What The Code Does

The `weekCpiR` pipeline is much more complete than our current simple bridge:

- parses weekly prices, weekly CPI, monthly CPI, weights, dictionaries, OKATO
  and item codes;
- converts weekly item movements into daily index paths under a uniform
  within-week assumption;
- treats regulated utility registration dates specially;
- aggregates item-level weekly indices upward through CPI hierarchy levels 7 to
  1 using a modified Laspeyres formula;
- reconciles modelled weekly paths to monthly facts through a bias-correction
  layer;
- converts daily base indices back into monthly MoM using Rosstat registration
  days or fixed cutoffs.

The useful formula is in `modified_laspeyres_by_sum.R`:

```text
effective_weight_i = weight_i * previous_december_index_i /
                     sum(weight_i * previous_december_index_i)
group_index = sum(effective_weight_i * item_index_i)
```

This is materially better than the current simple within-component average in
`docs/NOWCASTING.md`.

### What To Port

Port these ideas to Python, not the whole R project:

1. Weighted item aggregation via modified Laspeyres.
2. Week-to-day expansion and monthly registration-calendar logic.
3. Separate outputs for:
   - raw weekly chain;
   - price-level bridge;
   - modified-Laspeyres item aggregation;
   - train-only calibration/bias correction.
4. Item contribution tables for fuel, fruit/vegetables, ЖКУ and other policy
   groups.

Best target location:

```text
sirena/data/weekly_bridge.py
```

or a new helper under:

```text
sirena/data/weekly_laspeyres.py
```

### What Not To Port Directly

- The pre-trained `model2_interregional.RDS` CatBoost model: its training
  vintage and leakage discipline are unknown.
- Full-sample bias correction as production evidence.
- Hardcoded `last_date`, `last_availible_weights_year`, and external paths.

### Leakage Risks

`data_cpi_week_bi_bias_correct.R` computes average historical model error and a
perfect historical fit against monthly facts. This is fine for diagnostics and
historical reconstruction, but in rolling nowcast evaluation it must be
estimated only from data available at the cutoff.

Exchange-rate features in `data_cpi_week_wow_nowcast.R` also need strict
availability checks by week.

### Immediate SIRENA Use

This should become a targeted nowcast upgrade, not a new ensemble model yet.

Acceptance check before promotion:

- rolling h=0/h=1 nowcast backtest versus current `weekly_bridge_v1`;
- cutoff-safe monthly facts and weekly publication dates;
- item-level contribution reconciliation;
- no use of official monthly CPI before it is available.

## 2. Micro / Subcomponent Forecasting

### Mordovia Python

File:

```text
experiments/code_repository_20260625/nested_extracted/a0268154c4591/ВВГУ_Мордовия_КСП инфляции Python/infl.py
```

What it does:

- loads `Data_cpi_mord.xlsx`, sheet `base_SA`;
- runs `pmdarima.auto_arima` on 7 non-seasonal and 40 seasonal component
  series;
- forecasts 12 periods and writes a matrix to Excel.

Useful part:

- simple Python loop over many component series;
- explicit split between seasonal and non-seasonal component groups.

Not enough for production:

- hardcoded local path;
- no weight aggregation back to headline CPI;
- no cutoff-safe rolling backtest;
- no fallback for short or unstable series.

This can inform a baseline for our existing
`data/external/micro_cpi_region_export/region_cpi_long.csv`, but it should not
replace `Micro_SM`.

### Khabarovsk 45-Component Model

File:

```text
experiments/code_repository_20260625/nested_extracted/a2363877985ef/Прогноз_ИПЦ_по компонентам_Хабаровск/khab_mod.prg
```

What it does:

- defines 45 components across food, non-food and services;
- wraps seasonal adjustment and ARIMA in a reusable `sa_arima` subroutine;
- uses X-13/TRAMO-SEATS where possible, with fallback behavior;
- multiplies component MoM forecasts by weights from `vesa.xlsx`;
- reconstructs non-food, food, services and headline CPI through explicit
  contribution sums.

This is the best external template for a medium-granularity bottom-up model.

What to port:

- the fixed 45-component taxonomy;
- the contribution arithmetic;
- special policy components such as `u_gkh` and fuel-like non-food groups;
- the idea of a single reusable component forecast routine.

What to change:

- use SIRENA-KBR weights, not fixed Habarovsk weights;
- use recursive/vintage-safe seasonal adjustment;
- avoid EViews `autoarma` as a black box;
- keep actual historical facts and forecast periods clearly separated.

### Omsk 45-Component Model

File:

```text
experiments/code_repository_20260625/nested_extracted/afa0cc7c1c88b/ARIMA-45 (Омск, СГУ)/Скрипт/arima_omsk.prg
```

What it adds:

- same 45-component idea, but implemented verbosely;
- explicit `_mom_vklad` and `_yoy_vklad` contribution construction;
- useful evidence that a 45-component level is operationally manageable.

Main weakness:

- repetitive code;
- STL decomposition likely uses full sample unless carefully rerun per cutoff;
- some manual interpolation in service components.

Use Omsk as a taxonomy and aggregation reference; use Khabarovsk as the cleaner
architecture reference.

## 3. Variable Selection / ARIMAX / ML

### Volgograd Variable Selection

File:

```text
experiments/code_repository_20260625/nested_extracted/a0ac088b0b1be/Методика отбора переменных, обеспечивающих надежные прогнозы/variable_models_code.prg
```

What it does:

- compares AR benchmark against ARX models;
- selects lags by Schwarz/BIC;
- uses recursive forecasts;
- reports relative RMSE and outperform counts.

This is useful as a diagnostic gate. We already ran a first KBR pilot under:

```text
archive/results/variable_selection_pilot_20260625/
```

Next step should be a Pythonized, reusable version that evaluates variable
blocks, not only individual series.

### ARIMAX

File:

```text
experiments/code_repository_20260625/nested_extracted/ae00677697d94/ARIMAX.R
```

What it does:

- estimates a fixed ARIMAX and an `auto.arima` ARIMAX;
- uses `IPC_mom` plus regressors such as USD and shock dummies;
- reads future regressor values from `Прогноз регрессоров.xlsx`.

Main risk:

- future regressor paths are external. In backtesting this is a large
  lookahead-bias risk unless every regressor forecast is deterministic or was
  genuinely available at the historical cutoff.

Useful part:

- transparent ARIMAX challenger idea with shock dummies.

Port only as a controlled `ARIMAXPolicy` challenger with:

- last-observed, AR, or documented scenario exogenous paths;
- h=1/h=2/h=12 evaluation;
- residual diagnostics and stability checks.

### ML R Pipeline

Files:

- `experiments/code_repository_20260625/nested_extracted/a9c66f72e388c/Комбинированный прогноз ИПЦ моделей машинного обучения/methodology.Rmd`
- `experiments/code_repository_20260625/nested_extracted/a9c66f72e388c/Комбинированный прогноз ИПЦ моделей машинного обучения/forecasts/XGB.Rmd`
- `experiments/code_repository_20260625/nested_extracted/a9c66f72e388c/Комбинированный прогноз ИПЦ моделей машинного обучения/forecasts/Ensembles.Rmd`

Findings:

- The supplied `main1.xlsx` currently has only `date` and `ipc`, so the ML
  example is effectively univariate in the unpacked material.
- `XGB.Rmd` uses rolling-origin resampling, which is directionally sound.
- `Ensembles.Rmd` computes weights using errors on the same holdout being
  reported; this is not a valid real-time forecast ensemble.

SIRENA already has `sirena/models/xgboost_model.py`, so there is little value
in porting R `tidymodels` code.

Useful idea:

- consider additional feature families mentioned in the methodology:
  agricultural producer prices, deposits, retail demand, USD.

But these should enter through:

```text
sirena/macro_features.py
```

and pass the same pseudo-OOS gate before inclusion.

## Recommended Next Development Tasks

### P0. Weekly Laspeyres Nowcast Prototype

Build a Python prototype that uses current KBR weekly data and KBR weights:

```text
experiments/weekly_laspeyres_nowcast/
```

Output:

- `weekly_laspeyres_nowcast.csv`
- `weekly_laspeyres_contributions.csv`
- `weekly_laspeyres_backtest.md`

Do not include it in production `Ensemble` until it beats current
`weekly_bridge_v1` in a cutoff-safe nowcast backtest.

### P0. 45-Component Taxonomy Mapping For KBR

Create a mapping table between:

- Habarovsk/Omsk 45 groups;
- SIRENA `SubcomponentMulti` codes;
- KBR micro item codes and weights;
- scenario groups: ЖКУ, fuel, fruit/vegetables.

Suggested output:

```text
experiments/kbr_45_component_mapping/
```

### P1. Python Variable-Selection Gate

Generalize the existing pilot into a reusable tool:

```text
experiments/variable_selection/
```

Evaluate by horizon and by blocks:

- monetary;
- demand;
- components;
- weekly/fuel;
- policy/tariff dummies.

### P1. ARIMAX Challenger Only After Gate

Do not start with `auto.arima` grid search. Start from a parsimonious
specification approved by the variable-selection gate and compare against
Huber/Ridge/SubcomponentMulti.

## Readiness Ranking

| Rank | External code block | Immediate value | Porting effort | Production risk |
|---:|---|---:|---:|---:|
| 1 | `weekCpiR` aggregation/calendar/bias concepts | Very high | Medium | Medium |
| 2 | Khabarovsk 45-component model | High | Medium-high | Medium |
| 3 | Omsk 45-component model | Medium-high | Medium-high | Medium-high |
| 4 | Mordovia Python `auto_arima` loop | Medium | Low | Medium |
| 5 | Volgograd variable-selection EViews | High as methodology | Low-medium | Low if Pythonized |
| 6 | ARIMAX.R | Medium as challenger | Low | High if future exog leaks |
| 7 | Volgograd ML Rmd | Low-medium | Medium | Medium-high |

## Verification Status

This is a code review and feasibility analysis. No external model was executed
as a SIRENA-KBR forecast, and no production forecast cache was changed.

Before any promotion, run the relevant SIRENA rolling backtest and document:

- h=1/h=2/h=12 metrics;
- residual and stability diagnostics;
- cutoff-safe feature availability;
- trajectory realism for 12-month paths.
