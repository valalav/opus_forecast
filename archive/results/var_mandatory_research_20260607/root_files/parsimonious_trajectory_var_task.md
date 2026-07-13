# Parsimonious Multi-Horizon VAR / Trajectory Task

## Purpose

Build a professional mandatory VAR-family model that is both:

- **parsimonious**: tests whether simpler variable sets beat larger macro-heavy specifications;
- **trajectory-realistic**: produces deterministic, economically plausible multi-month paths, not flat lines, explosive paths, or fake realism from random noise.

This task continues after robust/outlier-aware VAR work. The previous best h=1 model was `RegimeMacroVARX_l1`, but it degraded at h=12. This task must address variable subset selection and multi-horizon trajectory quality directly.

## Required Context

Read first:

- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/robust_outlier_var_task.md`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/codex_cli_robust_var_report.md`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/opus48_robust_var_report.md`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/codex_cli_mandatory_var_report.md`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/opus48_mandatory_var_report.md`
- `/home/valalav/_projects/sirena-kbr/docs/BACKTEST_METHODOLOGY.md`
- `/home/valalav/_projects/sirena-kbr/docs/MODEL_CATALOG.md`
- `/home/valalav/_projects/sirena-kbr/docs/FOOD_TARIFF_FORECAST_2026_2027.md` if tariff/seasonal path assumptions are needed.

Known state:

- `PlainVAR_BIC`: stable mandatory VAR baseline, good interpretability.
- `RegimeMacroVARX_l1`: best h=1 robust VAR so far, but h=12 deteriorates.
- `RobustFAVAR_f2_l1`: horizon-stable robust alternative.
- `varx_last_exog_l1`: strong non-shock h=1 but bad in 2022 shock.

## Non-Negotiable Rules

- No future leakage.
- No production data edits.
- No stochastic noise added to make trajectories look realistic.
- Forecast paths must be deterministic for a fixed input/config/seed.
- Do not optimize only h=1. Evaluate at least h=1, h=2, and h=12.
- Report trajectory realism separately from point-error metrics.
- Prefer simple models when their validation metrics are comparable to complex models.

## Core Questions

1. Which variable subset is best for each horizon and regime?
2. Does including all macro variables hurt compared with a smaller subset?
3. Can we keep the h=1 macro gain without degrading h=12?
4. Which VAR-family model gives the most realistic deterministic 12-month trajectory?
5. Should the mandatory VAR be horizon-specific or a single stable model?

## Candidate Variable Sets

Test with and without each macro block. At minimum:

### Endogenous / component sets

- `CPI`
- `CPI + Food + NonFood + Services`
- `Food + NonFood + Services`, with CPI reconstructed by weights
- `CPI + Food`
- `CPI + NonFood`
- `CPI + Services`
- `CPI + Food + NonFood`
- `CPI + Food + Services`
- `CPI + NonFood + Services`

### Macro/exog sets

Test all as cutoff-safe exogenous or jointly modeled alternatives:

- none
- `USD`
- `Ki_i`
- `Ruonia`
- `USD + Ki_i`
- `USD + Ruonia`
- `Ki_i + Ruonia`
- `USD + Ki_i + Ruonia`

Optional only if already available and cutoff-safe:

- Brent / oil features
- `fl_potrb_zad`, `fl_dep`, `all_real`
- selected production/activity factors from existing data

Do not assume “more variables is better”. Penalize over-parameterized models.

## Model Families To Evaluate

Use VAR-family only:

- Plain VAR / BIC-selected VAR
- BVAR with Minnesota shrinkage
- VARX / BVARX with deterministic exog scenarios
- component-constrained VAR
- robust FAVAR
- regime-aware VARX
- robust equation-by-equation VAR where appropriate
- VAR-family model averaging only if weights are selected inside cutoff

## Exogenous Path Rules

For VARX/BVARX, future exog paths must be explicit and deterministic.

Test at least:

- last observed value scenario;
- simple AR/VAR forecast of exog variables using only cutoff data;
- mean-reversion-to-recent-average scenario;
- announced/current key-rate path if available in existing data/docs and no future actual is used.

Report which path is used. Do not use future actual USD/Ruonia/Ki.

## Seasonality / Trajectory Rules

Multi-month trajectories must not be judged only by average MAE.

Add deterministic seasonality handling where appropriate:

- month-of-year deterministic terms;
- seasonal dummies;
- train-only seasonal residualization if used carefully;
- component-specific seasonality;
- tariff/admin-price seasonal dummies if documented;
- deterministic seasonal reconstruction.

No random noise. If uncertainty is shown, it must be a separate interval/diagnostic, not part of the point forecast.

## Trajectory Realism Diagnostics

For every h=12 candidate, compute and report:

- h=1, h=2, h=12 MAE/RMSE/KPI;
- path volatility: std of 12 monthly forecast values;
- historical-comparable volatility ratio;
- number of sign changes;
- max month-to-month jump in forecast path;
- seasonal amplitude by month;
- flatness metric: share of adjacent monthly changes below a small threshold;
- explosive-path flag;
- component consistency if components are forecasted;
- visual trajectory chart for best candidates.

Flag unrealistic paths:

- nearly straight line / excessive flatness;
- explosive paths;
- wrong seasonal shape;
- implausible repeated identical values;
- unrealistic jump not explained by deterministic scenario or seasonality.

## Evaluation Windows

At minimum:

- `2018-01..2019-12`
- `2020-01..2021-12`
- `2022-01..2022-12`
- `2023-01..2023-12`
- `2024-01..2025-03`
- `2025-04..2026-03`

Use h=1, h=2, h=12. If runtime becomes high, use the full grid for h=1/h=2 and a shortlist for h=12, but explain the shortlist rule.

## Baselines

Compare to:

- `PlainVAR_BIC`
- `RegimeMacroVARX_l1`
- `RobustFAVAR_f2_l1`
- `FAVAR_f2_l1`
- `varx_last_exog_l1`
- `Archived_BVAR`
- random walk
- seasonal naive

External context:

- Huber
- RidgeShockDummies
- best current production h=12 model if available.

## Acceptance Criteria

A candidate can replace the robust mandatory VAR if it:

- is leakage-free;
- improves h=1 or h=2 without materially degrading h=12 trajectory;
- or improves h=12/trajectory realism while keeping h=1 close to the current robust model;
- beats archived BVAR and simple baselines;
- has a parsimonious variable set or a validated subset-selection rule;
- produces deterministic realistic paths without added random noise.

If no single model works across horizons, recommend a horizon-specific VAR-family policy:

- h=1: best robust/macro candidate;
- h=2: best short-horizon candidate;
- h=12: best trajectory-stable candidate.

## Artifacts

Use `AGENT_ID`.

Run directories:

- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/{AGENT_ID}_trajectory_var_<name>/`

Final report:

- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/{AGENT_ID}_trajectory_var_report.md`

Required files:

- `config.json`
- `metrics.csv`
- `predictions.csv`
- `comparison.csv`
- `selection_log.csv`
- `trajectory_metrics.csv`
- `leakage_checks.csv`
- `notes.md`
- trajectory charts for best h=12 candidates
- script copy

## Final Report Must Include

- variable subsets tested;
- exogenous path assumptions tested;
- final recommended model or horizon-specific policy;
- h=1/h=2/h=12 metrics;
- trajectory realism diagnostics;
- examples of rejected unrealistic paths;
- leakage audit;
- exact commands;
- final status:
  - `recommended parsimonious trajectory VAR`
  - `recommended horizon-specific VAR policy`
  - `experimental trajectory VAR`
  - `no improvement over robust VAR`

## Reporting Tone

Be strict. A model with a slightly better point MAE but a flat, explosive, or economically implausible trajectory should not be recommended as the long-horizon VAR. Do not use random noise to make forecasts look realistic.
