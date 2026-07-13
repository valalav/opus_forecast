# Parallel Nested Re-Selection Task

## Purpose

Evaluate whether the **seasonal-residual VAR family** has deployable forecasting value when hyperparameters are selected honestly inside each cutoff, rather than chosen on the final 12-month evaluation window.

This is a common task file for parallel agents. Each agent must write its own report and run artifacts using the `AGENT_ID` given in its `/goal` prompt.

## Core Question

Does seasonal-residual VAR remain useful under nested rolling/expanding re-selection, or was the previous `fine_seasonal_resid_var_tc_roll42_l5` result only single-window grid-selection overfit?

## Context To Read First

- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/task.md`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/opus48_review_report.md`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/run_var_sa_backtests.py`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/run_fixed_config_robustness.py`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/opus48_robustness.py`
- `/home/valalav/_projects/sirena-kbr/docs/BACKTEST_METHODOLOGY.md`
- `/home/valalav/_projects/sirena-kbr/docs/MODEL_CATALOG.md`

Important previous findings:

- `fine_seasonal_resid_var_tc_roll42_l5` reproduced exactly on the selection window:
  - h=1 MAE `0.223115`
  - h=2 MAE `0.308349`
  - h=12 MAE `0.413011`
- No leakage was found in the train-only seasonal residualization path.
- The specific `roll42_l5` config failed out-of-selection robustness:
  - Codex full robustness h=1 MAE about `0.489487`
  - Opus non-selection non-shock h=1 average about `0.435`
- Current status: seasonal-residual VAR family is `experimental only`; specific `roll42_l5` is rejected for standalone production.

## Required Independence Rules

- Do not overwrite another agent's outputs.
- Use the `AGENT_ID` provided in the `/goal` prompt.
- Put run artifacts only under:
  - `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/{AGENT_ID}_nested_<name>/`
- Write final report only to:
  - `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/{AGENT_ID}_nested_reselection_report.md`
- Do not edit this shared task file during the run.
- Do not edit production data.
- Do not integrate any model into production.
- If code is needed, create research-only scripts under `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/` with `{AGENT_ID}` in the filename.

## Methodological Requirement

The key requirement is **nested selection**.

For each outer target date:

1. Define the forecast cutoff according to horizon.
2. Use only data available at that cutoff.
3. Inside that cutoff, run an inner historical validation to select hyperparameters.
4. Fit the selected configuration on all data available at the outer cutoff.
5. Forecast the outer target date.
6. Record the selected hyperparameters and prediction.

Never choose roll-window, lag count, variables, or model family using the outer target actual.

## Candidate Family

At minimum, evaluate seasonal-residual VAR variants using official data only:

- Data: `/home/valalav/_projects/sirena-kbr/data/inflation_data.csv`
- Variables:
  - total/components: `CPI`, `Food`, `NonFood`, `Services`
  - component-only bottom-up variant if feasible
- Seasonal factor:
  - trailing month-of-year means computed from train-only data
- Candidate grid:
  - seasonal roll windows: `24`, `30`, `36`, `42`, `48`, `60`, `72`
  - VAR lags: `1` to `6`
  - optional train modes: expanding and rolling120

Do not use revised SA files as production evidence unless the report clearly labels them as revised-history experiments.

## Outer Evaluation

Evaluate at least h=1. Add h=2 and h=12 if feasible.

Use multiple historical windows, not only the previous selection window:

- `2018-01..2019-12`
- `2020-01..2021-12`
- `2022-01..2022-12`
- `2023-01..2023-12`
- `2024-01..2025-03`
- previous selection/reference window `2025-04..2026-03`

If runtime is too high, prioritize h=1 across all windows and explain what was skipped.

## Baselines

Compare against at least simple baselines computed with the same outer dates:

- Random walk
- Seasonal naive / trailing month-of-year mean
- Plain VAR without seasonal residualization

If feasible, also compare against project models:

- Huber
- RidgeShockDummies
- archived-style BVAR
- Ridge_ProdProxy_Roll24
- SubcomponentMulti only if runtime is manageable with cutoff-safe data handling

## Metrics

For every evaluated model/window/horizon, compute:

- MAE
- RMSE
- bias / mean error
- max absolute error
- KPI violations: `abs(error) > 0.5`
- coverage: share with `abs(error) <= 0.5`
- number of observations

Also record:

- chosen roll-window per target date
- chosen lag per target date
- chosen family/variant per target date
- inner validation score per target date

## Required Artifacts

Each agent must create at least one run directory:

- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/{AGENT_ID}_nested_<name>/config.json`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/{AGENT_ID}_nested_<name>/predictions.csv`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/{AGENT_ID}_nested_<name>/metrics.csv`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/{AGENT_ID}_nested_<name>/selection_log.csv`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/{AGENT_ID}_nested_<name>/notes.md`

Final report:

- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/{AGENT_ID}_nested_reselection_report.md`

## Final Report Must Answer

- Was nested re-selection implemented correctly?
- What inner validation scheme was used?
- Did the seasonal-residual VAR family beat simple baselines out of selection?
- Did it beat project baselines, if tested?
- Are selected hyperparameters stable or noisy?
- Is there evidence of deployable signal?
- Final status:
  - `production candidate`
  - `ensemble candidate`
  - `experimental only`
  - `rejected`
- Exact commands run.
- What remains unchecked.

## Suggested Starting Commands

```bash
cd /home/valalav/_projects/sirena-kbr
python3 -m py_compile experiments/var_sa_research/run_var_sa_backtests.py
python3 -m py_compile experiments/var_sa_research/run_fixed_config_robustness.py
```

If a new script is created, compile it before running:

```bash
python3 -m py_compile experiments/var_sa_research/{AGENT_ID}_nested_reselection.py
python3 experiments/var_sa_research/{AGENT_ID}_nested_reselection.py
```
