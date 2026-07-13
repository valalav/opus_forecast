# Opus 4.8 Independent VAR/SA Review Task

## Purpose

Run an independent audit and robustness check of the VAR/SA research in `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research`.

Treat the current Codex CLI result as a hypothesis, not as verified truth.

## Primary Question

Is the reported best configuration `fine_seasonal_resid_var_tc_roll42_l5` a real robust improvement, or did it overfit the same 12-month evaluation window used for selection?

## Required Context

Read first:

- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/task.md`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/run_var_sa_backtests.py`
- `/home/valalav/_projects/sirena-kbr/docs/BACKTEST_METHODOLOGY.md`
- `/home/valalav/_projects/sirena-kbr/docs/MODEL_CATALOG.md`
- `/home/valalav/_projects/sirena-kbr/docs/BVAR_ANALYSIS.md`

Inspect existing run artifacts, especially:

- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter01_all_quick_v3`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter02_seasonal_residual`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter03_seasonal_fine`

Compare against archived metrics:

- `/home/valalav/_projects/sirena-kbr/archive/results/backtest_h1_metrics.csv`
- `/home/valalav/_projects/sirena-kbr/archive/results/backtest_h2_metrics.csv`
- `/home/valalav/_projects/sirena-kbr/archive/results/backtest_h12_metrics.csv`
- `/home/valalav/_projects/sirena-kbr/archive/results/backtest_h1_predictions.csv`
- `/home/valalav/_projects/sirena-kbr/archive/results/backtest_h2_predictions.csv`
- `/home/valalav/_projects/sirena-kbr/archive/results/backtest_h12_predictions.csv`

## Current Claim To Verify

Codex CLI reportedly found:

- Best config: `fine_seasonal_resid_var_tc_roll42_l5`
- Family: train-only seasonal residual statsmodels VAR
- Data: `/home/valalav/_projects/sirena-kbr/data/inflation_data.csv`
- Variables: CPI, Food, NonFood, Services
- Seasonal factor: trailing 42-month month-of-year mean, computed inside each cutoff
- Core model: VAR on residuals, `lags=5`
- Reconstruction: residual forecast plus target-month seasonal mean
- Reported metrics:
  - h=1 MAE `0.223115`
  - h=2 MAE `0.308349`
  - h=12 MAE `0.413011`
  - h=1 KPI violations `2`
- Reported status: `experimental`

Do not accept these numbers until independently reproduced from saved artifacts or rerun.

## Rules

- Do not change production code unless the audit explicitly requires a small bug fix.
- Do not edit production data.
- Do not integrate anything into `ModelRegistry`.
- Avoid conflicts with another running agent:
  - Put all new runs under `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/opus48_*`.
  - Put your final report at `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/opus48_review_report.md`.
  - Do not overwrite existing `iter01_*`, `iter02_*`, or `iter03_*` artifacts.
- If you update `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/task.md`, only append a short cross-reference to your report after all work is complete.
- Do not claim robustness unless tested on windows not used for config selection.
- Record negative findings clearly.

## Audit Tasks

1. Reproduce the reported best run.
   - Verify that `iter03_seasonal_fine` really contains the reported config and metrics.
   - Rerun the same preset with a new run name such as `opus48_reproduce_iter03`.
   - Check whether results are deterministic for the same seed.

2. Inspect the runner for leakage.
   - Check seasonal residualization.
   - Confirm trailing 42-month month-of-year means are computed using train-only data for each cutoff.
   - Confirm target-month actuals are not used in seasonal factors, lag matrices, transformations, or model selection.
   - Confirm h=2 and h=12 use the correct target forecast step.

3. Test robustness without retuning.
   - Freeze `fine_seasonal_resid_var_tc_roll42_l5`.
   - Evaluate on at least two additional historical windows if data length allows.
   - Suggested windows:
     - 2022-01 to 2022-12
     - 2023-01 to 2023-12
     - 2024-01 to 2024-12
     - 2025-01 to 2025-12 or the current standard 12-month window
   - Use expanding or rolling backtest with the same cutoff discipline.

4. Compare to baselines on the same windows.
   - Archived BVAR
   - Huber
   - RidgeShockDummies
   - ElasticNet
   - SubcomponentMulti
   - Ridge_ProdProxy_Roll24 if available in archived predictions

5. Stress-test the hypothesis.
   - Check nearby but not over-tuned variants:
     - roll windows around 30, 36, 42, 48, 60
     - lags around 1 to 6
   - Identify whether `roll42_l5` is part of a stable plateau or an isolated lucky point.
   - Do not choose a new best production config from the same small window without labeling it exploratory.

6. Assess usefulness.
   - Standalone h=1 specialist?
   - Ensemble candidate?
   - Experimental only?
   - Rejected due to leakage, instability, or weak out-of-window performance?

## Required Outputs

Create:

- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/opus48_review_report.md`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/opus48_<name>/config.json`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/opus48_<name>/metrics.csv`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/opus48_<name>/predictions.csv`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/opus48_<name>/notes.md`

The report must include:

- whether the Codex CLI metrics were reproduced;
- any leakage or methodology issues found;
- robustness metrics by window;
- comparison against baselines on the same windows;
- whether `roll42_l5` looks stable or overfit;
- exact commands run;
- final status: `production candidate`, `ensemble candidate`, `experimental only`, or `rejected`;
- remaining unchecked items.

## Suggested Commands

Start with:

```bash
cd /home/valalav/_projects/sirena-kbr
python3 -m py_compile experiments/var_sa_research/run_var_sa_backtests.py
python3 experiments/var_sa_research/run_var_sa_backtests.py --run-name opus48_reproduce_iter03 --preset seasonal_fine --n-draws 160 --seed 20260607
```

If the existing runner cannot evaluate separate historical windows, add a small research-only extension under `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/` and document it in the report.

## Final Answer Format

Return a concise summary:

- task file used;
- report path;
- reproduced or not;
- leakage verdict;
- best robustness result;
- final status;
- exact commands used.
