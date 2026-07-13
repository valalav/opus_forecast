# Robust / Outlier-Aware VAR Task

## Purpose

Build the strongest defensible mandatory VAR-family model **with proper outlier and shock handling**.

Previous runs showed that plain VAR(1), FAVAR, and VARX have useful signal in normal periods but degrade badly in shock windows, especially 2022. This task fixes the obvious missing econometric layer: robust estimation, intervention dummies, winsorization, shock downweighting, and outlier diagnostics.

## Core Question

Can a robust / outlier-aware VAR-family model improve the mandatory VAR benchmark while staying leakage-free and economically defensible?

## Required Context

Read first:

- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/mandatory_var_next_task.md`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/codex_cli_mandatory_var_report.md`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/opus48_mandatory_var_report.md`
- `/home/valalav/_projects/sirena-kbr/docs/BACKTEST_METHODOLOGY.md`
- `/home/valalav/_projects/sirena-kbr/docs/MODEL_CATALOG.md`

Current benchmark:

- Recommended mandatory VAR: `PlainVAR_BIC`, effectively VAR(1) on `CPI, Food, NonFood, Services`.
- Strong h=1 challenger: `favar_macro_components_f2_l1`.
- Macro VARX works in non-shock periods:
  - `varx_last_exog_l1` h=1 out-of-selection non-shock MAE about `0.306`.
  - But shock 2022 MAE about `1.014`, so naive exog handling breaks.

## Non-Negotiable Rules

- No future leakage.
- No production data edits.
- No production integration unless explicitly requested later.
- Use official monthly data as the primary source: `/home/valalav/_projects/sirena-kbr/data/inflation_data.csv`.
- Any outlier rule must be computed using train-only information at each cutoff, unless it is an explicitly pre-declared calendar intervention dummy.
- Do not hide 2022. Either model it explicitly or report performance with and without shock windows.

## Required Model Directions

Evaluate at least five of the following.

### 1. VAR With Intervention / Shock Dummies

Use VAR-equation regressions with deterministic exogenous dummies:

- 2022 sanctions shock dummy;
- COVID dummy if justified;
- high-volatility month dummy;
- tariff/admin-price dummy if known from existing docs/data.

The dummy calendar must be declared before evaluation and not selected by target errors.

Implementation can be equation-by-equation OLS with lagged endogenous variables plus deterministic dummies, equivalent to VAR with exogenous deterministic terms.

### 2. Additive Outlier / Pulse Dummy Detection

Inside each cutoff:

- fit baseline VAR;
- identify additive outliers using train residual robust z-score / MAD;
- create pulse dummies for detected train outlier months;
- refit VARX with those dummies;
- forecast target with future pulse dummies set to zero unless target month is in a pre-declared intervention calendar.

Detection threshold examples: robust z > 3.5 or 4.0. Tune threshold only inside validation.

### 3. Winsorized / Huberized Training Data

Train VAR on transformed series:

- winsorize MoM values using train-only rolling/expanding quantiles;
- Huberize deviations from train-only seasonal/monthly center;
- compare component-wise and total/components variants.

The target actual must remain untransformed for evaluation.

### 4. Robust Equation-by-Equation VAR

Estimate each VAR equation with robust regression:

- HuberRegressor or statsmodels RLM;
- predictors are lagged endogenous variables and optional deterministic terms;
- multi-step forecast recursively.

This is still a VAR-family model because the dynamic system is lagged endogenous variables; robust loss only changes estimation.

### 5. Student-t / Heavy-Tailed BVAR Approximation

Approximate heavy-tailed errors:

- residual bootstrap with robust scale;
- Student-t predictive shocks;
- BVAR with outlier downweighting;
- or mixture shock distribution.

If full Bayesian implementation is too slow, keep it as a documented approximation.

### 6. Regime-Aware Macro VARX

Use macro exog only where it helps:

- normal / non-shock regime: VARX with `USD, Ruonia, Ki_i`;
- shock regime: robust VAR/BVAR or downweighted macro VARX;
- regime determined only by cutoff-available data or pre-declared calendar.

The goal is to keep the non-shock gain from macro variables without letting 2022 dominate errors.

### 7. Robust FAVAR

Extend the FAVAR result with robust preprocessing:

- train-only robust scaling;
- winsorized factors;
- PCA fit only on train;
- optional outlier dummies.

Compare to `favar_macro_components_f2_l1`.

## Baselines To Compare

Mandatory:

- `PlainVAR_BIC`
- `plain_var_tc_l1`
- `favar_macro_components_f2_l1`
- `varx_last_exog_l1`
- `Archived_BVAR`
- random walk
- seasonal naive

External context only:

- Huber
- RidgeShockDummies

## Evaluation

Primary:

- h=1 MAE over all windows;
- h=1 MAE out-of-selection;
- h=1 MAE out-of-selection non-shock;
- h=1 MAE on 2022 shock window;
- KPI violations.

If feasible:

- h=2 and h=12 for the best few candidates.

Use windows:

- `2018-01..2019-12`
- `2020-01..2021-12`
- `2022-01..2022-12`
- `2023-01..2023-12`
- `2024-01..2025-03`
- `2025-04..2026-03`

## Acceptance Criteria

A robust VAR candidate is stronger than the current mandatory VAR if it:

- remains leakage-free;
- beats `PlainVAR_BIC` on h=1 all or out-of-selection;
- does not sacrifice non-shock performance badly;
- materially improves the 2022 shock window or KPI violations;
- beats archived BVAR;
- is explainable as a VAR-family model.

If the best robust model improves shock handling but slightly loses non-shock accuracy, document it as an alternative shock-robust VAR, not necessarily the default mandatory VAR.

## Artifacts

Use `AGENT_ID`.

Run directories:

- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/{AGENT_ID}_robust_var_<name>/`

Final report:

- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/{AGENT_ID}_robust_var_report.md`

Each run directory must include:

- `config.json`
- `metrics.csv`
- `predictions.csv`
- `comparison.csv`
- `outlier_log.csv`
- `selection_log.csv` if any tuning/selection is used
- `leakage_checks.csv`
- `notes.md`
- script copy

## Final Report Must Include

- exact outlier/shock handling methods tested;
- final recommended robust VAR, if any;
- whether it replaces `PlainVAR_BIC` or is a shock-robust alternative;
- h=1 all/out-of-selection/non-shock/2022 metrics;
- h=2/h=12 if tested;
- outlier dates detected or dummies used;
- leakage audit;
- exact commands;
- final status:
  - `recommended robust mandatory VAR`
  - `shock-robust VAR alternative`
  - `experimental robust VAR`
  - `no robust improvement`

## Reporting Tone

Be blunt. If outlier handling helps only because it hard-codes 2022, say so. If it improves shock windows but hurts normal windows, say so. The goal is a defensible mandatory VAR model, not a cosmetic metric win.
