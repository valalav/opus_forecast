# Review Of Added Volgograd / R Methodology Experiments

Date saved: 2026-06-25

Reviewed folders:

- `experiments/Комбинированный прогноз ИПЦ_Волгоград/`
- `experiments/Методика отбора переменных, обеспечивающих надежные прогнозы/`
- `experiments/Преобразование данных из Excel в длинный панельный набор данных с использованием R/`

## Executive Summary

The most useful piece for SIRENA-KBR is not the ready Volgograd ML forecast
itself, but the **pseudo-out-of-sample variable selection methodology** from
the EViews package.

Recommended useful takeaways:

1. Build a Python variable-selection diagnostic for KBR macro and auxiliary
   features: AR benchmark vs one-variable ARX candidates, recursive
   pseudo-OOS, BIC lag choice, RMSE ratio, and outperform ratio.
2. Reuse the wide-Excel-to-long-panel idea as a Python import utility for
   future Rosstat regional Excel exports.
3. Treat the Volgograd ML scripts only as a rough template for rolling-origin
   tuning, not as production evidence.

Do not copy the Volgograd ensemble logic directly: its current weighting uses
same-holdout realized errors to weight same-holdout predictions, which is
leakage for real forecasting.

## Volgograd Combined ML Forecast

Files inspected:

- `methodology.Rmd`
- `results.Rmd`
- `forecasts/Random forest.Rmd`
- `forecasts/XGB.Rmd`
- `forecasts/SVM.Rmd`
- `forecasts/kNN.Rmd`
- `forecasts/Neural network.Rmd`
- `forecasts/Ensembles.Rmd`
- `data/main1.xlsx`
- `results/*_metrics.csv`

The documentation describes five supervised ML methods:

- SVR
- Random forest
- XGBoost
- MLP
- kNN

The intended predictors in the methodology are:

- `ipc`
- `agro_price`
- `deposit`
- `demand`
- `usd_cb`

But the actual `data/main1.xlsx` in the provided folder has only:

```text
date, ipc
```

So the included metrics do not demonstrate a proper multivariate CPI model.
Most model recipes remove `date`; for RF/XGB/SVM/MLP this leaves no genuine
economic predictors in the current file. kNN adds date signature features, so
it behaves more like a calendar-pattern learner.

Observed metrics from included CSV files:

| Horizon | Best included method by MAE | MAE |
|---:|---|---:|
| 3 | kNN | 0.732 |
| 9 | ML ensemble | 0.462 |
| 12 | ML ensemble | 0.348 |
| 24 | ML ensemble | 0.605 |

Important caveat: the ensemble weights are calculated from realized errors on
the same holdout period being combined. This is useful as an ex-post comparison,
but not as a real-time forecast weighting scheme.

Useful ideas to reuse safely:

- horizon-specific ML candidates;
- rolling-origin resampling for hyperparameter tuning;
- Latin hypercube hyperparameter search;
- MAE/MASE reporting by horizon;
- saving forecasts and metrics to reusable CSV artifacts.

What not to reuse directly:

- same-holdout ensemble weighting;
- results from `main1.xlsx` as evidence for KBR;
- ML models without explicit lag/exogenous features;
- a single final split as production backtest evidence.

## Variable Selection Methodology

Files inspected:

- `Методика отбора переменных.docx`
- `Описание кода.docx`
- `variable_models_code.prg`
- `example_data.xlsx`

This is based on:

```text
Andic, S. and F. Ogunc. 2015.
Variable selection for inflation: a pseudo out-of-sample approach.
Central Bank of Turkey Working Paper No. 15/06.
```

Core model structure:

```text
y = c + lags(y) + lags(x_i) + e
```

For each candidate variable:

- use recursive pseudo-out-of-sample evaluation;
- choose lag length by Schwarz/BIC;
- compare against an autoregressive benchmark;
- score variables by:
  - cumulative RMSE ratio versus benchmark;
  - frequency of outperforming benchmark;
  - frequency of outperforming benchmark by a required margin.

The document suggests a good variable should:

- improve cumulative RMSE versus benchmark, e.g. `RRMSESUM <= 0.95`;
- outperform the benchmark often enough, e.g. outperform ratio at least around
  `0.6`.

This is directly useful for SIRENA-KBR because it matches current model-work
needs:

- avoid adding all macro variables blindly;
- identify which predictors are genuinely useful for KBR h=1/h=2/h=12;
- make macro and auxiliary features pass a real-time cutoff-safe screen before
  entering Ridge/Huber/VAR/Factor models.

Recommended adaptation:

- implement under `experiments/variable_selection/`;
- source data from `data/inflation_data.csv`, `sirena/macro_features.py`, and
  selected auxiliary files;
- evaluate h=1, h=2, h=12 separately;
- include baseline comparisons against AR/seasonal naive and current Huber or
  RidgeShockDummies where appropriate;
- save selected variables, metrics, and rolling predictions as CSV.

## Excel To Long Panel R Script

Files inspected:

- `code.R`
- `data/Данные.xlsx`
- `regions_rosstat.csv`
- `Описание кода.docx`

The R script:

1. reads several sheets from one Excel workbook;
2. renames the first column to `region_rus`;
3. pivots each sheet from wide years to long format;
4. full-joins sheets into a regional panel;
5. writes `regions_rosstat.csv`.

The generated panel has columns:

```text
region_rus, year, invest, k_autonomy, shadow_economy
```

This is a simple but useful import pattern. In SIRENA-KBR, it is better to
implement this in Python rather than depend on R:

- `pandas.read_excel(sheet_name=None)`
- normalize first column as region name;
- `melt` each sheet to long format;
- merge on `region, period`;
- validate duplicates, missing region names, and period parsing.

This can be reused for future Rosstat wide Excel exports and regional panels,
but it is less important than the variable-selection methodology.

## Recommended Next Steps

Priority 1:

- Build a Python pseudo-OOS variable selection runner for KBR candidate
  features, using the Andic/Ogunc-style criteria.

Priority 2:

- Add a small panel-Excel loader utility for future regional Excel files.

Priority 3:

- Only after variable selection and proper lag features are in place, test ML
  candidates such as XGBoost/RF/SVR as KBR experimental models. Use true rolling
  backtests and no same-holdout weighting.

Not recommended immediately:

- promoting the Volgograd ML ensemble;
- copying the R scripts into production;
- treating the included Volgograd metrics as evidence for KBR.
