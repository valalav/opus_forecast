# Full factor proposal sweep evidence

Дата: 2026-06-09

## Итог

После проверки полного перечня предложений выбранная факторная модель по
forecast-score не изменилась:

`RobustFAVAR_lean_f2_l1_seasonal`

Она остается лучшим обязательным factor-family baseline по weighted score:

| Model | Score | h=1 MAE | h=2 MAE | h=12 MAE | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `RobustFAVAR_lean_f2_l1_seasonal` | 0.400 | 0.371 | 0.425 | 0.439 | Selected |

После дополнительного эконометрического review/DR для отчетной защиты добавлена
вторая спецификация:

`StationaryBlockFAVAR_dspread_l1`

Она зарегистрирована как `stationary_block_favar` и является diagnostics-aware
моделью для случаев, когда важнее пройти hard gates, чем минимально улучшить
h=1 MAE.

| Model | Score | h=1 MAE | h=2 MAE | h=12 MAE | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `StationaryBlockFAVAR_dspread_l1` | 0.401 | 0.378 | 0.418 | 0.432 | Selected for econometric report |

Source run:

- `experiments/factor_model_research/runs/factor_proposal_sweep/`
- DFM control: `experiments/factor_model_research/runs/factor_dfm_control/`

## Проверенные предложения

| Proposal | Best tested candidate | Score | h=1 MAE | h=2 MAE | h=12 MAE | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Current FAVAR baseline | `RobustFAVAR_lean_f2_l1_seasonal` | 0.400 | 0.371 | 0.425 | 0.439 | Selected |
| Transparent block bridge | `BlockBridge_pca_l1_huber_seasonal` | 0.407 | 0.370 | 0.442 | 0.446 | Not promoted |
| Sparse PCA | `BlockBridge_sparse_pca_l1_huber_seasonal` | 0.404 | 0.366 | 0.438 | 0.445 | Better h=1, worse h=2/h=12 |
| Weighted PCA | `BlockBridge_weighted_pca_l1_huber_seasonal` | 0.405 | 0.368 | 0.439 | 0.448 | Better h=1, worse h=2/h=12 |
| Quantile/median bridge | `BlockBridge_pca_l1_quantile_seasonal` | 0.406 | 0.372 | 0.438 | 0.444 | Not promoted |
| Rolling window | `RobustFAVAR_lean_f2_l1_seasonal_rolling120` | 0.426 | 0.395 | 0.455 | 0.462 | Worse than expanding |
| Publication-lag proxy | `RobustFAVAR_lean_f2_l1_seasonal_pubLag1` | 0.422 | 0.402 | 0.444 | 0.438 | h=1/h=2 degrade |
| Seasonal baseline | `SeasonalAR1` | 0.432 | 0.414 | 0.452 | 0.445 | Worse than FAVAR |
| Regime split proxy | `BlockBridge_pca_l1_huber_regimeMedian` | 0.471 | 0.444 | 0.507 | 0.485 | Rejected |
| DFM control | `DFM_components_f1_q1_seasonal` | 0.432 | 0.414 | 0.453 | 0.447 | Worse than FAVAR |

## Diagnostics

Selected FAVAR residual diagnostics from `residual_diagnostics.csv`:

| Horizon | Ljung-Box p | ARCH-LM p | DM abs-loss delta vs seasonal | DM p vs seasonal |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.604 | 0.999 | -0.077 | 0.008 |
| 2 | 0.599 | 0.175 | -0.023 | 0.404 |
| 12 | 0.145 | 0.251 | -0.008 | 0.181 |

Interpretation:

- h=1 selected FAVAR materially beats the seasonal mean baseline by MAE
  difference and DM-style paired absolute-loss check.
- h=2/h=12 improvements over seasonal baseline are weaker but still the selected
  FAVAR has the best weighted factor-family score.
- Residual diagnostics do not show obvious autocorrelation/ARCH failure for the
  selected FAVAR at the tested horizons.

Source-series autocorrelation diagnostics were also saved:

- `experiments/factor_model_research/runs/factor_proposal_sweep/series_autocorrelation_diagnostics.csv`
- `experiments/factor_model_research/runs/factor_proposal_sweep/series_autocorrelation_diagnostics.json`
- `experiments/factor_model_research/runs/factor_diagnostics_full/econometric_diagnostics_report.md`
- `experiments/factor_model_research/runs/factor_diagnostics_full/diagnostic_gate_summary.csv`

Ljung-Box at lag 12 flags autocorrelation in the source `CPI`, components, and
several macro/activity series. This is consistent with the decision to keep a
dynamic FAVAR layer rather than a static factor regression. The selected
FAVAR rolling forecast errors did not show the same obvious autocorrelation flag.

Important caveat from the full diagnostics pack: the in-sample FAVAR factor
equation residuals (`Factor_1`, `Factor_2`) still show Ljung-Box lag 12 flags,
and `Ruonia`/`Deposits` do not pass a clean ADF/KPSS stationarity gate. Therefore
the model is defensible as a mandatory factor benchmark/control, but not as a
fully clean structural econometric model.

## Diagnostics-aware follow-up

To address those warning gates, a stationary block FAVAR follow-up was run:

- runner: `experiments/factor_model_research/stationary_block_favar_research.py`
- artifacts: `experiments/factor_model_research/runs/stationary_block_favar/`
- implementation: `sirena/models/stationary_block_favar.py`

The selected `StationaryBlockFAVAR_dspread_l1` uses component and monetary PCA
blocks with `USD`, `Ki_i`, and `d_spread_Ruonia_Ki`, then estimates a compact
Huber FAVAR with lag 1. It is slightly worse than the original FAVAR on h=1
but better on h=2/h=12 and passes the hard econometric gates:

| Gate | Selected result |
| --- | --- |
| Transformed-source stationarity | all ADF/KPSS gates pass |
| Structural-break unit-root check | Zivot-Andrews rejects unit root with one break for all transformed inputs |
| Equation residual autocorrelation | Ljung-Box min p = 0.514; Breusch-Godfrey min p = 0.075 |
| Equation residual ARCH | min p = 0.084 |
| Equation residual mean/bias | min p = 0.127 |
| Rolling forecast-error autocorrelation | h=1/h=2/h=12 Ljung-Box p = 0.557 / 0.246 / 0.149 |
| Rolling forecast-error ARCH | h=1/h=2/h=12 ARCH-LM p = 0.999 / 0.274 / 0.234 |
| Normality | Jarque-Bera warning remains and must be disclosed |

Phillips-Perron was not run because the optional `arch` package is not installed.
Cointegration/VECM is not applicable to the selected stationary-transformation
specification; it becomes mandatory only if a future model returns to I(1)
level variables.

## Limitations

- Publication-lag check is a proxy: macro/financial series are shifted by one
  month inside each train window. It is not a real-time vintage test.
- Regime test is a simple train-only median split, not a full Markov-switching
  FAVAR.
- DFM was tested as a compact statsmodels control, not as a large Bayesian/Kalman
  research track.
- Quantile/median bridge was tested in a narrow representative setup because
  broad QuantileRegressor grids are slow.

## Decision

Keep `factor_policy` as the mandatory factor model by forecast-score.

Keep `factor_bridge` registered with default ensemble weight `0.0` as a tested
research challenger. Do not add it to forecast cache or dashboard unless a
future run improves h=2/h=12 or changes the promotion rule.

Use `stationary_block_favar` as the report-facing diagnostics-aware factor
model when the requirement is to show a genuine factor model with hard
econometric gates passed. Keep its default ensemble weight at `0.0` until a
separate production-promotion decision is made.
