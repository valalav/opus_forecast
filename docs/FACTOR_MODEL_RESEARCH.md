# Factor-family model research

This document summarizes the mandatory factor-model track completed on 2026-06-09.

## Goal

The project needed a defensible factor-family model in addition to the mandatory VAR model. The goal was not to beat the production ensemble, but to build a real leakage-free factor benchmark/control model, test several variants, and document the selected configuration honestly.

## Selected models

There are now two factor-family statuses:

- Use `FactorPolicyForecaster`, registered as `factor_policy`, as the primary
  factor-family benchmark by forecast score.
- Use `StationaryBlockFAVARForecaster`, registered as
  `stationary_block_favar`, as the diagnostics-aware report model when the
  presentation needs stricter econometric gates.

Implementation:

- `sirena/models/factor_policy.py`
- `sirena/models/stationary_block_favar.py`
- research runner: `experiments/factor_model_research/factor_policy_backtest.py`
- diagnostics-aware runner:
  `experiments/factor_model_research/stationary_block_favar_research.py`
- final run: `experiments/factor_model_research/runs/factor_policy_rolling/`
- diagnostics-aware run:
  `experiments/factor_model_research/runs/stationary_block_favar/`

Selected specification:

| Field | Value |
| --- | --- |
| Model family | Robust seasonal FAVAR |
| Information set | `Food`, `NonFood`, `Services`, `USD`, `Ki_i`, `Ruonia` |
| Factor extraction | PCA, train-only standardization |
| Factors | 2 |
| VAR/factor lag | 1 |
| Seasonality | train-only month-of-year residualization and reconstruction |
| Robustness | Huber equation estimation |
| Registry key | `factor_policy` |

Diagnostics-aware report specification:

| Field | Value |
| --- | --- |
| Model family | Stationary block FAVAR |
| Component block | `Food`, `NonFood`, `Services` |
| Monetary block | `USD`, `Ki_i`, `d_spread_Ruonia_Ki` |
| Factor extraction | two separate train-only PCA blocks |
| VAR/factor lag | 1 |
| Seasonality | train-only month-of-year residualization and reconstruction |
| Robustness | Huber equation estimation |
| Registry key | `stationary_block_favar` |
| Ensemble default weight | `0.0` |

## Economic logic

The selected model can be presented as a **Robust Seasonal FAVAR**:

> Робастная сезонная FAVAR-модель выделяет скрытые общие факторы из компонент
> инфляции и макрофинансовых показателей, после чего прогнозирует инфляцию с
> учетом ее инерции, сезонности и реакции на общие ценовые и финансовые импульсы.

Economic interpretation:

- The component block (`Food`, `NonFood`, `Services`) captures the internal
  structure of regional inflation: food shocks, services persistence, and
  pass-through into non-food goods.
- The macro-financial block (`USD`, `Ki_i`, `Ruonia`) captures external and
  monetary conditions: exchange-rate/import pressure, cost of money, and policy
  tightness.
- PCA factors compress these correlated series into latent common drivers rather
  than using each noisy monthly series as a separate unrestricted regressor.
- The compact VAR/FAVAR layer keeps inflation inertia and lagged factor dynamics.
- Train-only seasonal residualization addresses the strong calendar profile of
  regional inflation without using future months.
- Huber equation estimation reduces the influence of shock periods and large
  outliers, especially around food and monetary/financial stress episodes.

Use this wording in management-facing reports: the model is an interpretable
factor benchmark and policy-control signal, not a replacement for the stronger
production ensemble unless future validation improves materially.

For a stricter econometric presentation, use this name:

> **Стационарная блочная FAVAR-модель денежно-компонентных факторов**

Short report wording:

> Модель выделяет два интерпретируемых скрытых фактора: компонентный фактор
> региональной инфляции и денежно-финансовый фактор на стационарных
> трансформациях. Затем компактная робастная FAVAR-система прогнозирует ИПЦ с
> учетом инерции, сезонности и лагового влияния этих факторов.

## Tested variants

The research runner compared:

- FAVAR with component-only, macro-only, lean, and full information sets;
- 1, 2, and 3 PCA factors;
- lag 1 and lag 2 factor VAR systems;
- raw vs train-only seasonal residual specifications;
- OLS vs Huber equation estimation;
- BlockFactorBridge challengers with train-only component and macro factors,
  direct horizon equations, PCA/average factor construction, Huber/Ridge
  estimators, and lags 1..3;
- limited statsmodels DFM controls in the smoke run.

All rolling forecasts train only up to `target_date - horizon`. Future macro actuals are not used.

## Block bridge challenger result

Agent-reviewed alternatives were implemented as
`sirena.models.factor_bridge.FactorBridgeForecaster` and tested in
`experiments/factor_model_research/runs/factor_bridge_compact/`.

A broader proposal sweep was then run under
`experiments/factor_model_research/runs/factor_proposal_sweep/`, with a DFM
control under `experiments/factor_model_research/runs/factor_dfm_control/`.
The evidence packet is saved at
`experiments/factor_model_research/agent_reports/20260609_full_proposal_sweep_evidence.md`.

Best compact bridge challenger:

| Candidate | Score | h=1 MAE | h=2 MAE | h=12 MAE | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `BlockBridge_pca_l1_huber_seasonal` | 0.407 | 0.370 | 0.442 | 0.446 | Not promoted |

Interpretation: the transparent bridge design is useful as a defensible tested
alternative and is nearly tied on h=1, but it degrades h=2/h=12 relative to the
selected FAVAR baseline. The current `factor_policy` remains the recommended
mandatory factor model.

Full proposal-sweep conclusion: sparse PCA, weighted PCA, quantile/median
bridge, rolling-window estimation, publication-lag proxy, seasonal baselines,
simple regime split, residual diagnostics, DM-style checks, and DFM control were
all checked. Several bridge variants improve h=1 slightly, but all degrade h=2
and/or h=12 enough that they are not promoted.

## Final rolling metrics

Source: `experiments/factor_model_research/runs/factor_policy_rolling/factor_model_report.md`.

| Horizon | N | MAE | RMSE | KPI violations | Coverage <= 0.5 | Non-shock MAE | Bias |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| h=1 | 100 | 0.371 | 0.619 | 26 | 74.0% | 0.334 | 0.008 |
| h=2 | 100 | 0.425 | 0.710 | 27 | 73.0% | 0.355 | 0.003 |
| h=12 | 100 | 0.439 | 0.718 | 30 | 70.0% | 0.374 | -0.011 |

## h=12 trajectory diagnostics

| Diagnostic | Value |
| --- | ---: |
| Mean path MAE | 0.418 |
| Mean volatility ratio | 0.489 |
| Mean flatness share | 0.084 |
| Mean seasonal correlation | 0.971 |
| Max absolute jump | 0.679 |
| Explosive path rate | 0.0% |

Interpretation: the selected factor model is weaker than the best production/subcomponent models on h=1, but it is a real data-rich factor benchmark with stable seasonal trajectories and no explosive paths.

## Econometric diagnostics

Raw monthly series do have autocorrelation. This is one reason the model uses a
FAVAR/VAR-style dynamic layer instead of a purely static factor regression.
However, the extended diagnostics pack shows that not every econometric gate is
a clean pass.

Full diagnostics were saved under:

- `experiments/factor_model_research/runs/factor_diagnostics_full/econometric_diagnostics_report.md`
- `experiments/factor_model_research/runs/factor_diagnostics_full/diagnostic_gate_summary.csv`
- `experiments/factor_model_research/runs/factor_diagnostics_full/source_series_diagnostics.csv`
- `experiments/factor_model_research/runs/factor_diagnostics_full/selected_favar_in_sample_residual_diagnostics.csv`
- `experiments/factor_model_research/runs/factor_diagnostics_full/selected_favar_rolling_forecast_error_diagnostics.csv`

Gate summary:

| Gate | Status | Interpretation |
| --- | --- | --- |
| Source autocorrelation | Expected warning | Ljung-Box lag 12 flags `CPI`, `Food`, `NonFood`, `Services`, `USD`, `Ruonia`, `Deposits`, `RetailReal` |
| Source stationarity | Warning | ADF/KPSS is not clean for `Ruonia`, `Deposits` |
| In-sample FAVAR equation residual autocorrelation | Warning | Factor equations `Factor_1`, `Factor_2` have Ljung-Box lag 12 flags |
| Rolling forecast-error autocorrelation | Pass | h=1/h=2/h=12 rolling forecast errors have no Ljung-Box lag 12 flag |
| Source multicollinearity | Pass | Scaled condition number 2.29, max absolute pairwise correlation 0.49 |
| Factor stability | Pass | Rolling 84-month PCA median explained variance sum 0.618, max loading change 0.262 |
| Residual normality | Expected warning | Jarque-Bera rejects normality for CPI/factor equations; Huber estimation is used |

Rolling forecast-error diagnostics for selected FAVAR:

| Horizon | Ljung-Box p | ARCH-LM p | Interpretation |
| ---: | ---: | ---: | --- |
| h=1 | 0.604 | 0.999 | No clear residual autocorrelation/ARCH flag |
| h=2 | 0.599 | 0.175 | No clear residual autocorrelation/ARCH flag |
| h=12 | 0.145 | 0.251 | No clear residual autocorrelation/ARCH flag |

Interpretation: the model is acceptable as an interpretable mandatory
factor-family benchmark/control model. It should not be described as a fully
clean structural econometric model, because source series are strongly dynamic,
two in-sample factor equations retain autocorrelation, several source rows are
not cleanly stationary, and normality is rejected. The cleanest evidence is
forecast-oriented: rolling h=1/h=2/h=12 errors do not show a clear autocorrelation
or ARCH flag, and alternative factor designs did not beat the selected FAVAR
under the promotion rule.

## Diagnostics-aware stationary block FAVAR

Because the original selected FAVAR still had warning gates, a second
diagnostics-aware specification was tested after the DR/econometric review.
The final candidate is:

`StationaryBlockFAVAR_dspread_l1`

Registered implementation: `stationary_block_favar`.

Source run:

- `experiments/factor_model_research/runs/stationary_block_favar/stationary_block_favar_report.md`
- `experiments/factor_model_research/runs/stationary_block_favar/transformed_source_diagnostics.csv`
- `experiments/factor_model_research/runs/stationary_block_favar/in_sample_residual_diagnostics.csv`
- `experiments/factor_model_research/runs/stationary_block_favar/rolling_forecast_error_diagnostics.csv`

Rolling metrics:

| Horizon | N | MAE | RMSE | Coverage <= 0.5 | Bias |
| ---: | ---: | ---: | ---: | ---: | ---: |
| h=1 | 100 | 0.378 | 0.619 | 77.0% | 0.042 |
| h=2 | 100 | 0.418 | 0.704 | 75.0% | 0.044 |
| h=12 | 100 | 0.432 | 0.715 | 71.0% | 0.030 |

Candidate decision:

| Candidate | Score | h=1 MAE | h=2 MAE | h=12 MAE | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `StationaryBlockFAVAR_dspread_l2` | 0.399 | 0.367 | 0.431 | 0.430 | Rejected: ARCH-LM residual gate fails |
| `StationaryBlockFAVAR_dspread_l1` | 0.401 | 0.378 | 0.418 | 0.432 | Selected: only candidate passing hard gates |
| `StationaryBlockFAVAR_dspread_l3` | 0.406 | 0.379 | 0.433 | 0.430 | Rejected: ARCH-LM residual gate fails |

Hard diagnostics for selected candidate:

| Gate | Result |
| --- | --- |
| Source stationarity | Pass: all transformed inputs pass ADF p < 0.05 and KPSS p > 0.05 |
| Structural-break stationarity check | Pass: Zivot-Andrews rejects unit root with one break for all transformed inputs |
| In-sample equation autocorrelation | Pass: Ljung-Box p >= 0.514 and Breusch-Godfrey p >= 0.075 |
| In-sample equation ARCH | Pass: minimum ARCH-LM p = 0.084 |
| In-sample mean/bias | Pass: minimum t-test p = 0.127 |
| Rolling forecast-error autocorrelation | Pass: h=1/h=2/h=12 Ljung-Box p = 0.557 / 0.246 / 0.149 |
| Rolling forecast-error ARCH | Pass: h=1/h=2/h=12 ARCH-LM p = 0.999 / 0.274 / 0.234 |
| Normality | Expected warning: Jarque-Bera rejects normality for shock-heavy inflation errors/residuals |

Phillips-Perron was not run because the optional `arch` package is not installed
in this environment. Cointegration/VECM tests are not applicable to the final
stationary block specification because it intentionally avoids nonstationary
level variables; if a future model uses I(1) levels, Engle-Granger/Johansen
testing becomes mandatory.

Interpretation: `stationary_block_favar` is the cleanest model to present if the
priority is econometric defensibility rather than raw h=1 MAE. It gives up about
0.007 p.p. h=1 MAE versus `factor_policy`, but improves h=2/h=12 slightly and
passes the hard residual/stationarity gates. Do not claim literal Gaussian
errors; claim robust estimation with documented non-normality warning.

## Current status

- Production integration: registered model `factor_policy`.
- Research challenger integration: registered model `factor_bridge` with default
  ensemble weight `0.0`.
- Diagnostics-aware report integration: registered model
  `stationary_block_favar` with default ensemble weight `0.0`.
- Forecast cache integration: `FactorPolicy` key in `scripts/precompute_forecasts.py` with modest control-model weight `0.03`.
- Standard backtest integration: `FactorPolicy` column in `scripts/backtest_framework.py`.

Use it as an interpretable mandatory factor-family control signal, not as a replacement for the stronger production ensemble unless future validation improves materially.

## Future-agent handoff

When continuing factor-model or adjacent model work with the
`sirena-model-econometrics` skill, start from these artifacts:

- `docs/FACTOR_MODEL_AGENT_PRD.md` - project framing, agent roles, acceptance gates.
- `experiments/factor_model_research/agent_reports/20260609_agent_review_synthesis.md` - initial MiniMax/Qwen/Gemini/Nemotron synthesis.
- `experiments/factor_model_research/agent_reports/20260609_full_proposal_sweep_evidence.md` - final evidence packet for all tested proposals.
- `experiments/factor_model_research/runs/factor_proposal_sweep/` - main proposal-sweep artifacts.
- `experiments/factor_model_research/runs/factor_dfm_control/` - compact DFM control artifacts.
- `experiments/factor_model_research/runs/stationary_block_favar/` - diagnostics-aware stationary block FAVAR artifacts.
- `sirena/models/factor_bridge.py` - implemented but not promoted block-factor challenger.
- `sirena/models/stationary_block_favar.py` - registered diagnostics-aware report model.

Important preserved conclusions:

1. Do not replace `factor_policy` with `factor_bridge` based only on h=1:
   sparse/weighted/bridge variants can slightly improve h=1 but degrade h=2/h=12.
2. Keep `factor_bridge` at default ensemble weight `0.0` unless a future run passes
   the promotion rule: at least 1% weighted-score improvement with no h=1/h=12
   degradation and no explosive h=12 trajectory.
3. Treat publication-lag results as a proxy, not real-time-vintage evidence.
4. Treat the median-regime split and DFM control as compact checks, not full
   Markov-switching/Bayesian DFM research tracks.
5. If future work revisits the factor line, the most promising direction is not a
   broad grid search; it is a targeted h=1 challenger with explicit handling for
   h=2/h=12 degradation, or a separate horizon-specific factor component.
6. When the task is to report a factor model that passes hard econometric gates,
   prefer `stationary_block_favar` over `factor_policy`, and keep the Jarque-Bera
   warning explicit.
