# Mandatory VAR Model Next Task

## Purpose

The project has an external requirement: there must be a VAR-family model. The previous work showed that one specific seasonal-residual VAR configuration (`roll42_l5`) was overfit and should not be used. This does **not** mean the VAR requirement is abandoned.

The goal now is to build the **best defensible VAR-family model** for СИРЕНА-КБР:

- methodologically correct;
- no future leakage;
- useful enough to report honestly;
- preferably competitive on h=1;
- with clear diagnostics, limitations, and self-evaluation.

The model does not have to beat every production ML model. It must be the strongest credible VAR implementation we can justify.

## Repository

- Root: `/home/valalav/_projects/sirena-kbr`
- Research area: `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research`

## Start With Previous Findings

Read:

- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/task.md`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/opus48_review_report.md`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/codex_cli_nested_reselection_report.md`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/opus48_nested_reselection_report.md`
- `/home/valalav/_projects/sirena-kbr/docs/BACKTEST_METHODOLOGY.md`
- `/home/valalav/_projects/sirena-kbr/docs/MODEL_CATALOG.md`
- `/home/valalav/_projects/sirena-kbr/docs/BVAR_ANALYSIS.md`
- `/home/valalav/_projects/sirena-kbr/archive/docs/SELF_EVALUATION_BVAR.md`
- `/home/valalav/_projects/sirena-kbr/archive/docs/INSTRUCTION_SELF_EVALUATION.md`

Previous conclusions to preserve:

- `roll42_l5` is rejected as a standalone production model.
- Train-only seasonal residualization is leakage-free but not enough by itself.
- Direct revised SA files are not production evidence unless real-time vintages are available.
- Honest nested selection is required for any tuned VAR-family result.

## New Direction

Do **not** keep tuning the same seasonal-residual VAR grid. Try richer VAR-family alternatives.

Evaluate at least three of the following directions:

### 1. Plain VAR / BVAR Baseline, Properly Documented

Create the cleanest possible baseline VAR/BVAR:

- official data only;
- total + components;
- conservative lag selection;
- train-only transformations;
- robust diagnostics;
- compare to ARIMA/SARIMA and archived BVAR.

This may be the final fallback if advanced variants fail.

### 2. Bayesian VAR With Model Averaging

Instead of selecting one fragile config, average across VAR/BVAR configurations:

- BVAR Minnesota / IW prior;
- lags 1..4 or 1..6;
- variable sets;
- shrinkage grid;
- weights learned only from inner validation;
- optionally Bayesian model averaging or softmax weights by inner MAE.

The final forecast must still be a VAR-family forecast, not an external ML ensemble.

### 3. VARX / BVARX With Exogenous Scenario Paths

Use a VAR-family model with exogenous variables where appropriate:

- CPI/components as endogenous variables;
- key macro variables as exogenous or jointly forecasted;
- candidates: USD, RUONIA/key rate, Ki, oil/Brent if already available;
- exogenous paths must be cutoff-safe:
  - forecast exog with existing `sirena/exog/var.py` or conservative scenario;
  - never use future actual exog values.

Document exactly which data are endogenous, exogenous, and forecasted.

### 4. Factor-Augmented VAR (FAVAR)

Use dimensionality reduction inside a VAR-family framework:

- extract PCA/factors from components/subcomponents/macro series using train-only data;
- VAR on CPI/components plus factors;
- forecast factors dynamically or as part of the VAR;
- compare to plain VAR.

Avoid using future information in PCA/scaling.

### 5. Regime-Aware VAR

Use simple, transparent regimes:

- normal / shock / high inflation;
- regimes determined only from information available at cutoff;
- either:
  - separate VAR/BVAR per regime if enough data, or
  - shrinkage/lag/weight rules conditional on regime.

Do not overfit many regimes. Prefer two or three documented regimes.

### 6. Component-Constrained VAR

Model components with VAR and reconstruct total inflation using weights:

- Food, NonFood, Services as endogenous variables;
- total CPI as weighted reconstruction, not necessarily endogenous;
- weights from `data/micro_sprav.csv` or documented official weights;
- compare against total-included VAR to test whether identity leakage/instability is reduced.

### 7. Forecast Combination Within VAR Family

Combine multiple VAR-family forecasts:

- plain VAR;
- BVAR;
- VARX/BVARX;
- component-constrained VAR;
- regime-aware VAR;
- FAVAR.

Weights must be selected inside each cutoff or set by transparent fixed rules. Do not mix in Huber/Ridge/ML models if calling this the mandatory VAR model.

## Required Metrics

At minimum:

- h=1 MAE;
- h=1 RMSE;
- h=1 bias;
- h=1 KPI violations (`abs(error) > 0.5`);
- coverage (`abs(error) <= 0.5`);
- h=2 and h=12 if runtime allows.

Compare to:

- archived BVAR;
- plain VAR;
- ARIMA/SARIMA baseline if available;
- Huber/RidgeShockDummies only as external context, not necessarily as the acceptance bar.

## Acceptance Criteria

A result can be accepted as the mandatory VAR model if:

- no leakage is found;
- the model beats archived BVAR on h=1 under fair validation;
- it is not worse than trivial VAR/seasonal-naive baselines;
- metrics and limitations are clearly documented;
- it has a stable enough configuration or a defensible averaging/selection mechanism;
- the final report explains why this is the best available VAR-family option even if ML models are better.

If none of the advanced variants beats the clean baseline VAR/BVAR, accept the clean baseline as the mandatory VAR model and document failed alternatives.

## Artifacts

Use an `AGENT_ID` in all outputs.

Run directories:

- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/{AGENT_ID}_mandatory_var_<name>/`

Final report:

- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/{AGENT_ID}_mandatory_var_report.md`

Each run directory should include:

- `config.json`
- `metrics.csv`
- `predictions.csv`
- `selection_log.csv` if tuning/selection is used
- `comparison.csv`
- `notes.md`
- any scripts used for the run

## Final Report Must Include

- exact VAR-family variants tested;
- final recommended mandatory VAR model;
- whether it is plain VAR, BVAR, VARX/BVARX, FAVAR, regime VAR, component-constrained VAR, or VAR-family combination;
- metrics table;
- comparison to archived BVAR and simple baselines;
- leakage audit;
- why rejected variants failed;
- exact commands run;
- final status:
  - `recommended mandatory VAR`
  - `experimental mandatory VAR candidate`
  - `fallback clean VAR only`
  - `no adequate VAR found`

## Important Tone For Reporting

Do not oversell. The final model can be described as:

- mandatory VAR-family benchmark;
- interpretable macro/component model;
- secondary control model;
- scenario/diagnostic model;
- not necessarily the main production forecaster.

Avoid claiming it is the best overall model if it does not beat production ML models.
