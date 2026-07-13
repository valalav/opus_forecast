# VAR/SA Research Task

## Goal

Implement and evaluate a high-quality VAR/BVAR model for KBR inflation forecasting, including variants that use seasonally adjusted (SA) data. Work iteratively: inspect existing work, run backtests, compare metrics, modify the approach, rerun, and document every material decision here.

This file is the single entry point for any agent continuing the task. Keep it updated after every meaningful iteration.

## Repository

- Root: `/home/valalav/_projects/sirena-kbr`
- Research project: `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research`
- Run artifacts: `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/<iteration_or_timestamp>/`

## Non-Negotiable Rules

- Do not claim the model works, improves quality, or is production-ready without running the exact relevant backtest.
- Do not treat "script ran" as "forecast quality verified".
- Do not modify production data under `/home/valalav/_projects/sirena-kbr/data/`.
- Do not replace existing workflows until the current code and archive are understood.
- Negative results are useful. Record them instead of hiding them.
- Avoid future leakage. Each rolling forecast must train only on data available at the cutoff date.
- Do not add the model to the production ensemble unless metrics justify it and verification is complete.

## Start Here

Read these first:

- `/home/valalav/_projects/sirena-kbr/docs/index.md`
- `/home/valalav/_projects/sirena-kbr/docs/BACKTEST_METHODOLOGY.md`
- `/home/valalav/_projects/sirena-kbr/docs/MODEL_CATALOG.md`
- `/home/valalav/_projects/sirena-kbr/docs/BVAR_ANALYSIS.md`

Then inspect existing VAR/BVAR and SA work:

- `/home/valalav/_projects/sirena-kbr/sirena/models/bvar.py`
- `/home/valalav/_projects/sirena-kbr/sirena/models/bvar_rate.py`
- `/home/valalav/_projects/sirena-kbr/sirena/exog/var.py`
- `/home/valalav/_projects/sirena-kbr/sirena/sa_data_loader.py`
- `/home/valalav/_projects/sirena-kbr/scripts/experiment_simple_var.py`
- `/home/valalav/_projects/sirena-kbr/archive/models/bvar_tuner.py`
- `/home/valalav/_projects/sirena-kbr/archive/scripts/run_bvar.py`
- `/home/valalav/_projects/sirena-kbr/archive/scripts/sirena_bvar.py`
- `/home/valalav/_projects/sirena-kbr/archive/scripts/tune_component_bvar.py`
- `/home/valalav/_projects/sirena-kbr/archive/docs/SELF_EVALUATION_BVAR.md`
- `/home/valalav/_projects/sirena-kbr/archive/docs/CHANGELOG_v32.md`

Existing result references:

- `/home/valalav/_projects/sirena-kbr/archive/results/bvar_tuning_results.json`
- `/home/valalav/_projects/sirena-kbr/archive/results/backtest_h1_metrics.csv`
- `/home/valalav/_projects/sirena-kbr/archive/results/backtest_h2_metrics.csv`
- `/home/valalav/_projects/sirena-kbr/archive/results/backtest_h12_metrics.csv`
- `/home/valalav/_projects/sirena-kbr/archive/results/backtest_h1_predictions.csv`
- `/home/valalav/_projects/sirena-kbr/archive/results/backtest_h2_predictions.csv`
- `/home/valalav/_projects/sirena-kbr/archive/results/backtest_h12_predictions.csv`

## Data Sources

Official monthly facts:

- `/home/valalav/_projects/sirena-kbr/data/inflation_data.csv`

Alternative inflation formats:

- `/home/valalav/_projects/sirena-kbr/data/infl_kbr.csv`
- `/home/valalav/_projects/sirena-kbr/data/infl_kbr_detailed.csv`

SA data:

- `/home/valalav/_projects/sirena-kbr/data/mom_sa_kbr.csv`
- `/home/valalav/_projects/sirena-kbr/data/sa_fl.csv`
- `/home/valalav/_projects/sirena-kbr/data/sa_hor.csv`
- `/home/valalav/_projects/sirena-kbr/data/sa_sub.csv`
- `/home/valalav/_projects/sirena-kbr/data/raw/sa.csv`
- `/home/valalav/_projects/sirena-kbr/data/raw/sa.xlsx`

Weights:

- `/home/valalav/_projects/sirena-kbr/data/micro_sprav.csv`
- `/home/valalav/_projects/sirena-kbr/data/access_weights.csv`

Optional auxiliary data, only with explicit no-leakage handling:

- `/home/valalav/_projects/sirena-kbr/data/kbr_weekly_prices_2008_2026.csv`
- `/home/valalav/_projects/sirena-kbr/data/kbr_weekly_cpi_2008_2026.csv`
- `/home/valalav/_projects/sirena-kbr/data/kbr_indices.csv`
- `/home/valalav/_projects/sirena-kbr/data/items_names.csv`

## Model Families To Try

Start by reproducing existing VAR/BVAR behavior before changing anything.

Minimum candidates:

- Classical `statsmodels` VAR on total + Prod + Nonprod + Serv.
- Current `BVARForecaster` from `/home/valalav/_projects/sirena-kbr/sirena/models/bvar.py`.
- `BVARForecaster` variants with `auto_lags`, `auto_lambda`, lag grids, shrinkage grids, and variable-set grids.
- VAR/BVAR on SA total + SA components.
- Bottom-up SA VAR: forecast SA components, aggregate by weights, then compare with official MoM.
- Component-only VAR/BVAR without total as endogenous target, to avoid mechanical leakage from total/component identity.
- Macro-augmented BVAR only if the cutoff discipline is clear.

Do not invent a parallel modeling framework unless existing patterns cannot support the experiment.

## Main Metrics

Primary metric:

- h=1 MAE, using `/home/valalav/_projects/sirena-kbr/docs/BACKTEST_METHODOLOGY.md`.

Also calculate:

- RMSE
- mean error / bias
- max absolute error
- KPI violations: count of `abs(error) > 0.5`
- coverage: share of months with `abs(error) <= 0.5`
- h=2 MAE
- h=12 MAE
- SIRENA Score if the candidate is evaluated across h=1, h=2, and h=12

Compare against at least:

- Huber
- RidgeShockDummies
- ElasticNet
- SubcomponentMulti
- Current archived BVAR metrics

Known historical warning: archived BVAR is not a clean success. Existing files show BVAR around h=1 MAE `0.4436`, h=2 MAE `0.6341`, and h=12 MAE `0.4802` in `/home/valalav/_projects/sirena-kbr/archive/results/backtest_h*_metrics.csv`. Treat this as a baseline to beat, not as proof of adequacy.

## Backtest Requirements

- Use rolling or expanding backtest with no future leakage.
- For each target month, fit only on data available before the target according to the horizon.
- Log train/test date ranges.
- Verify whether SA files contain revised history and document the implication. If SA data are revised using future information, report that limitation clearly.
- Prefer existing backtest scripts when possible:
  - `/home/valalav/_projects/sirena-kbr/scripts/run_backtest_h1.py`
  - `/home/valalav/_projects/sirena-kbr/scripts/run_backtest_h2.py`
  - `/home/valalav/_projects/sirena-kbr/scripts/run_backtest_h12.py`
- If a research runner is needed, place it under `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/` and document how it differs from production backtests.

## Artifact Requirements

Each iteration must create a subdirectory:

`/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/<iteration_or_timestamp>/`

Save, when applicable:

- `config.json`
- `metrics.csv`
- `predictions.csv`
- `comparison.csv`
- `notes.md`
- charts as `.png` or `.html`
- any temporary runner or notebook-like script used for the run

Update this `task.md` after each run.

## Iteration Loop

For every iteration:

1. State the hypothesis.
2. Implement the smallest useful change.
3. Run the relevant backtest.
4. Compare with previous run and production baselines.
5. Record whether the change is accepted, rejected, or inconclusive.
6. Add the next hypothesis.

Stop after either:

- a candidate is clearly good enough for experimental or production integration, or
- three consecutive meaningful iterations fail to improve the best candidate.

## Quality Targets

Desired, if achievable:

- h=1 MAE `<= 0.30`
- KPI violations no worse than strong production models
- no catastrophic deterioration on h=2 or h=12

If VAR/BVAR cannot compete, mark it as experimental or rejected. Do not force production integration.

## Production Integration Rules

Only if metrics justify it:

- Follow `/home/valalav/_projects/sirena-kbr/docs/ADDING_MODEL_GUIDE.md`.
- Register via `ModelRegistry`.
- Export in `/home/valalav/_projects/sirena-kbr/sirena/models/__init__.py`.
- Run `python3 scripts/add_model_checklist.py ModelName`.
- Run relevant h=1/h=2/h=12 backtests.
- Regenerate forecasts/charts only if production outputs changed.

## Final Deliverable

The final response must include:

- path to this task file;
- best configuration found;
- before/after metrics table;
- accepted and rejected hypotheses;
- exact commands used for verification;
- final status: `production-ready`, `experimental`, or `rejected`;
- what remains unchecked, if anything.

## Current Plan

- [x] Inspect required docs and existing VAR/BVAR/SA files.
- [x] Summarize previous BVAR results and failure modes.
- [x] Build or reuse a reproducible research backtest runner.
- [x] Reproduce current BVAR/VAR baseline.
- [x] Test SA VAR/BVAR variants.
- [x] Test bottom-up SA component aggregation.
- [x] Compare against production baselines.
- [x] Decide whether the best candidate deserves production integration.
- [x] Update final status and artifacts.
- [x] Freeze `fine_seasonal_resid_var_tc_roll42_l5` without further tuning.
- [x] Run out-of-selection historical robustness backtests.
- [x] Compare against Ridge_ProdProxy_Roll24, Huber, RidgeShockDummies, SubcomponentMulti, and archived BVAR.
- [x] Audit seasonal residualization leakage.
- [x] Evaluate standalone h=1 and ensemble-overlay usefulness.

## Iteration Log

### 2026-06-07 Initial Task Setup

- Created research project directory.
- Created this task file as the single handoff point for `/goal` mode and future agents.
- No model work has been verified yet.

### 2026-06-07 Iteration 1: Official VAR/BVAR Grid + Revised SA Variants

- Hypothesis:
  - A compact VAR/BVAR using total CPI + components, component-only bottom-up aggregation, or revised SA series might materially improve over archived BVAR and approach h=1 MAE `<= 0.30`.
- Implementation:
  - Added reproducible runner: `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/run_var_sa_backtests.py`.
  - Runner uses archived test dates from `archive/results/backtest_h*_predictions.csv` for direct comparison with current archived metrics.
  - For each target date, training data are restricted to `target_date - horizon` cutoff.
  - Evaluated 48 candidates across h=1, h=2, h=12:
    - classical `statsmodels` VAR on official CPI total/components;
    - official component-only VAR/BVAR with weighted bottom-up aggregation;
    - current `BVARForecaster` archive-style/default/auto-lambda/auto-lags variants;
    - BVAR lag/shrinkage grids on official total/components and total+Food+USD+Ruonia;
    - revised SA total/components VAR/BVAR and bottom-up SA variants.
- Commands:
  - `python3 -m py_compile experiments/var_sa_research/run_var_sa_backtests.py`
  - Failed artifact assembly run: `python3 experiments/var_sa_research/run_var_sa_backtests.py --run-name iter01_all_quick --preset all --n-draws 120 --seed 20260607`
  - Metadata-fixed but superseded run: `python3 experiments/var_sa_research/run_var_sa_backtests.py --run-name iter01_all_quick_v2 --preset all --n-draws 120 --seed 20260607`
  - Final iteration artifact run: `python3 experiments/var_sa_research/run_var_sa_backtests.py --run-name iter01_all_quick_v3 --preset all --n-draws 120 --seed 20260607`
- Artifacts:
  - `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter01_all_quick_v3/config.json`
  - `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter01_all_quick_v3/metrics.csv`
  - `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter01_all_quick_v3/predictions.csv`
  - `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter01_all_quick_v3/comparison.csv`
  - `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter01_all_quick_v3/notes.md`
- Results:
  - Best research candidate by h=1: `var_official_total_components_l3`, h=1 MAE `0.353971`, h=2 MAE `0.513577`, h=12 MAE `0.464564`, h=1 KPI violations `4`.
  - Best BVAR by h=1: `bvar_grid_tc_l2_lam0p5`, h=1 MAE `0.357545`, h=2 MAE `0.563157`, h=12 MAE `0.482139`, h=1 KPI violations `3`.
  - Best SA by h=1: `bvar_sa_components_bottomup_l1_lam0p5`, h=1 MAE `0.410530`; direct SA-to-NSA comparison did not help.
  - Archived comparison remains stronger: `Ridge_ProdProxy_Roll24` h=1 MAE `0.247688`; `Ridge_Shock` h=1 MAE `0.304220`; archived BVAR h=1 MAE `0.443584`.
- Decision:
  - Accepted: compact official VAR/BVAR variants improve over archived BVAR on h=1, so VAR/BVAR is worth one more targeted iteration.
  - Rejected: direct revised SA VAR/BVAR and bottom-up SA variants as production evidence; SA files are revised full-history estimates, not real-time vintages, and h=1 MAE stayed above `0.41`.
  - Rejected for production: no iteration-1 candidate meets h=1 target `<= 0.30` or beats current strong archived baselines.
- Next hypothesis:
  - The main error may come from forecasting raw MoM level through 2026 shocks. Try residualized/seasonal-reconstruction variants: VAR/BVAR on de-seasonalized residuals using only trailing historical month-of-year seasonality available at each cutoff, then add the seasonal factor back to compare with official MoM.

### 2026-06-07 Iteration 2: Train-Only Seasonal Residual VAR/BVAR

- Hypothesis:
  - VAR/BVAR failed because raw MoM contains strong calendar seasonality. Removing train-only month-of-year means before VAR/BVAR and adding target-month seasonality back may improve h=1 without using revised SA files.
- Implementation:
  - Extended runner with `seasonal` preset.
  - Added official-data-only seasonal residual candidates with expanding, 60-month, and 36-month month-of-year means.
  - Tested total/components residual VAR, component bottom-up residual VAR, and residual BVAR.
- Command:
  - `python3 -m py_compile experiments/var_sa_research/run_var_sa_backtests.py && python3 experiments/var_sa_research/run_var_sa_backtests.py --run-name iter02_seasonal_residual --preset seasonal --n-draws 160 --seed 20260607`
- Artifacts:
  - `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter02_seasonal_residual/`
- Results:
  - Best h=1: `seasonal_resid_var_tc_roll60_l3`, h=1 MAE `0.310918`, h=2 MAE `0.430910`, h=12 MAE `0.458566`, h=1 KPI violations not lower than production baselines.
  - Stronger balanced candidate: `seasonal_resid_var_comp_bottomup_roll36_l1`, h=1 MAE `0.315152`, h=2 MAE `0.362869`, h=12 MAE `0.321432`.
- Decision:
  - Accepted: train-only seasonal residualization is materially better than raw VAR/BVAR and worth fine tuning.
  - Rejected: iteration-2 configurations as production candidates because h=1 still stayed above `0.30`.
- Next hypothesis:
  - The seasonal window is the key hyperparameter. Fine tune rolling windows around 36-60 months and allow higher VAR lag orders.

### 2026-06-07 Iteration 3: Fine Seasonal Residual Grid

- Hypothesis:
  - A finer no-leakage grid around rolling seasonal windows and higher VAR lags can cross the h=1 target while preserving acceptable h=2/h=12 behavior.
- Implementation:
  - Extended runner with `seasonal_fine` preset.
  - Tested 113 candidates:
    - rolling seasonal windows 24, 30, 36, 42, 48, 60, 72 months;
    - total/components seasonal residual VAR with lags 2-6;
    - component bottom-up seasonal residual VAR with lags 1-4;
    - BVAR residual grid for selected windows, lags 1-2, lambda1 0.05-0.5.
  - Added charts for the best h=1 candidate.
- Command:
  - `python3 -m py_compile experiments/var_sa_research/run_var_sa_backtests.py && python3 experiments/var_sa_research/run_var_sa_backtests.py --run-name iter03_seasonal_fine --preset seasonal_fine --n-draws 160 --seed 20260607`
  - Chart command generated `h1_actual_vs_prediction.png` and `h1_abs_error.png` from saved `predictions.csv`.
- Artifacts:
  - `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter03_seasonal_fine/config.json`
  - `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter03_seasonal_fine/metrics.csv`
  - `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter03_seasonal_fine/predictions.csv`
  - `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter03_seasonal_fine/comparison.csv`
  - `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter03_seasonal_fine/notes.md`
  - `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter03_seasonal_fine/h1_actual_vs_prediction.png`
  - `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter03_seasonal_fine/h1_abs_error.png`
- Results:
  - Best h=1 candidate: `fine_seasonal_resid_var_tc_roll42_l5`.
    - h=1 MAE `0.223115`, RMSE `0.351414`, bias `-0.022378`, max abs error `0.848904`, KPI violations `2`, coverage `83.333333%`.
    - h=2 MAE `0.308349`, KPI violations `2`, coverage `83.333333%`.
    - h=12 MAE `0.413011`, KPI violations `4`, coverage `66.666667%`.
  - Best h=1 BVAR candidate: `fine_seasonal_resid_bvar_tc_roll42_l2_lam0p05`, h=1 MAE `0.244856`, h=2 MAE `0.408876`, h=12 MAE `0.402226`.
  - Archived comparison:
    - `Ridge_ProdProxy_Roll24`: h=1 MAE `0.247688`, h=2 `0.279325`, h=12 `0.303691`.
    - `Ridge_Shock`: h=1 MAE `0.304220`, h=2 `0.361755`, h=12 `0.381701`.
    - archived BVAR: h=1 MAE `0.443584`, h=2 `0.634056`, h=12 `0.480201`.
- Decision:
  - Accepted as experimental h=1 candidate: `fine_seasonal_resid_var_tc_roll42_l5`.
  - Rejected for production integration now:
    - h=12 is materially worse than the current strong baseline;
    - the best configuration was selected by a broad fine grid on the same 12-month evaluation window, so it needs a separate historical robustness/backtest period before production use;
    - it is not registered in `ModelRegistry` and no production checklist was run.
  - Stop condition reached: candidate is good enough for experimental status, but not production-ready.

### 2026-06-07 Completion Audit Notes

- Required archive files inspected during the task:
  - `archive/scripts/run_bvar.py`: old MCMC pipeline using `SirenaBVAR`, variables CPI/Food/USD/RUONIA, lags 2, prior tightness 0.5.
  - `archive/scripts/sirena_bvar.py`: analytical Minnesota-prior BVAR precursor with OLS covariance simplification.
  - `archive/scripts/tune_component_bvar.py`: component-level BVAR grid over component-specific variables and lags/lambda values.
  - `archive/docs/CHANGELOG_v32.md`: documents move from MCMC to analytical BVAR, lambda1 increase to 1.0, lags increase to 4, and prior data-leakage fixes around cutoffs.
- No archived workflow found a no-leakage train-only seasonal residual VAR with reconstruction; this is the new experimental path produced here.
- Production integration deliberately not performed because the best candidate has weak h=12 and needs robustness validation beyond the selection window.

### 2026-06-07 Robustness: Frozen `fine_seasonal_resid_var_tc_roll42_l5`

- Objective:
  - Freeze the selected configuration and test it outside the selection window `2025-04` through `2026-03`.
  - Do not retune lag count, seasonal window, variables, or ensemble weights.
- Frozen configuration:
  - Variables: `CPI`, `Food`, `NonFood`, `Services`.
  - Seasonal residualization: trailing `42` months, month-of-year means computed inside each cutoff only.
  - Core model: classical `statsmodels` VAR on residuals, lags `5`.
  - Forecast: h=1 residual forecast plus target-month seasonal mean.
- Implementation:
  - Added `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/run_fixed_config_robustness.py`.
  - Full historical profile uses h=1 target windows:
    - `2018-01..2019-12`
    - `2020-01..2021-12`
    - `2022-01..2022-12`
    - `2023-01..2023-12`
    - `2024-01..2025-03`
  - Train modes: expanding and rolling120.
  - Baselines in full run: `Ridge_ProdProxy_Roll24`, `Huber`, `RidgeShockDummies`, archived-style `BVAR`.
  - `SubcomponentMulti` full run was attempted but was too heavy because it retrains many Prophet/ML submodels at each cutoff; the long run was stopped and replaced by a compact representative-month run with a cutoff-safe wrapper.
  - `SubcomponentMulti_cutoff` wrapper cuts loaded subcomponent rows to each cutoff and uses `use_exog_forecast=False`.
- Commands:
  - `python3 -m py_compile experiments/var_sa_research/run_fixed_config_robustness.py`
  - Stopped impractical full SubcomponentMulti run:
    - `python3 experiments/var_sa_research/run_fixed_config_robustness.py --run-name robustness_fixed_roll42_l5_v1 --seed 20260607`
  - Full robustness without SubcomponentMulti:
    - `python3 experiments/var_sa_research/run_fixed_config_robustness.py --run-name robustness_fixed_roll42_l5_full_no_subcomp --seed 20260607 --skip-subcomponent --profile full`
  - Representative SubcomponentMulti cutoff-safe comparison:
    - `python3 experiments/var_sa_research/run_fixed_config_robustness.py --run-name robustness_fixed_roll42_l5_subcomp_representative --seed 20260607 --profile representative_months`
  - Added `comparison.csv` to both completed robustness run directories from `metrics.csv + ensemble_metrics.csv`.
- Artifacts:
  - `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/robustness_fixed_roll42_l5_full_no_subcomp/`
  - `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/robustness_fixed_roll42_l5_subcomp_representative/`
  - Each completed run includes `config.json`, `metrics.csv`, `predictions.csv`, `comparison.csv`, `ensemble_metrics.csv`, `leakage_checks.csv`, `notes.md`, runner copy, and PNG charts.
- Leakage audit:
  - Full run: `0` seasonal residualization violations across `174` fixed-VAR forecast rows.
  - Representative SubcomponentMulti run: `0` seasonal residualization violations across `10` fixed-VAR forecast rows.
  - Checks verify `train_end <= cutoff` and `seasonal_source_end < target_date`.
- Full robustness results, all historical pre-selection months:
  - Expanding:
    - `RidgeShockDummies`: MAE `0.355622`, KPI violations `17/87`.
    - `Huber`: MAE `0.365295`, KPI violations `21/87`.
    - `Archived_BVAR`: MAE `0.431308`, KPI violations `24/87`.
    - `Fixed_VAR_roll42_l5`: MAE `0.489487`, KPI violations `27/87`.
    - `Ridge_ProdProxy_Roll24`: MAE `0.578425`, KPI violations `21/69`; early rows fail minimum-train checks.
  - Rolling120:
    - `RidgeShockDummies`: MAE `0.375591`, KPI violations `20/87`.
    - `Huber`: MAE `0.379048`, KPI violations `22/87`.
    - `Archived_BVAR`: MAE `0.425865`, KPI violations `21/87`.
    - `Fixed_VAR_roll42_l5`: MAE `0.499687`, KPI violations `28/87`.
    - `Ridge_ProdProxy_Roll24`: MAE `0.578425`, KPI violations `21/69`.
- Fixed VAR by window:
  - Expanding `2020-2021`: MAE `0.294935`, acceptable.
  - Expanding `2018-2019`: MAE `0.422984`.
  - Expanding `2022`: MAE `0.692917`.
  - Expanding `2023`: MAE `0.613633`.
  - Expanding `2024-2025Q1`: MAE `0.645119`.
  - Rolling120 shows the same pattern and does not rescue the model.
- Representative SubcomponentMulti comparison:
  - Expanding representative months:
    - `RidgeShockDummies`: MAE `0.215981`.
    - `Archived_BVAR`: MAE `0.314285`.
    - `Huber`: MAE `0.320484`.
    - `SubcomponentMulti_cutoff`: MAE `0.353897`.
    - `Fixed_VAR_roll42_l5`: MAE `0.575812`.
  - Rolling120 representative months:
    - `Archived_BVAR`: MAE `0.263606`.
    - `Huber`: MAE `0.277078`.
    - `SubcomponentMulti_cutoff`: MAE `0.364533`.
    - `Fixed_VAR_roll42_l5`: MAE `0.522651`.
- Ensemble diagnostics:
  - Fixed overlays are diagnostic only; no optimized weights were fit.
  - `80% Huber + 20% FixedVAR` is nearly flat vs Huber in full expanding (`0.364336` vs Huber `0.365295`) and rolling120 (`0.375514` vs Huber `0.379048`), but it does not beat `RidgeShockDummies`.
  - RidgeProd overlays improve the weak historical RidgeProd run, but remain worse than Huber/RidgeShockDummies and are based on only `69` valid RidgeProd rows.
- Decision:
  - `fine_seasonal_resid_var_tc_roll42_l5` is not robust outside the selected window.
  - Reject as standalone h=1 specialist.
  - Do not promote to ensemble candidate; observed blend gains are too small and do not beat stronger baselines.
  - Final status for this robustness phase: `experimental only`.

## Run Table

| Run | Date | Hypothesis | Candidate | h=1 MAE | h=2 MAE | h=12 MAE | KPI Violations | Status | Artifacts |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| setup | 2026-06-07 | Task definition only | none | | | | | not run | `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/task.md` |
| iter01_all_quick | 2026-06-07 | Initial all-candidate run | runner artifact assembly | | | | | failed before saving metrics | `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter01_all_quick/config.json` |
| iter01_all_quick_v3 | 2026-06-07 | Official VAR/BVAR grid and revised SA variants can beat archived BVAR and approach 0.30 | `var_official_total_components_l3` | 0.353971 | 0.513577 | 0.464564 | 4 | best research candidate, not production-ready | `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter01_all_quick_v3/` |
| iter02_seasonal_residual | 2026-06-07 | Train-only seasonal residualization can improve raw VAR/BVAR | `seasonal_resid_var_tc_roll60_l3` | 0.310918 | 0.430910 | 0.458566 | | improved but still above h=1 target | `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter02_seasonal_residual/` |
| iter03_seasonal_fine | 2026-06-07 | Fine seasonal window/lag grid can cross h=1 target | `fine_seasonal_resid_var_tc_roll42_l5` | 0.223115 | 0.308349 | 0.413011 | 2 | accepted experimental, not production-ready | `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/iter03_seasonal_fine/` |
| robustness_fixed_roll42_l5_full_no_subcomp | 2026-06-07 | Frozen best config is robust outside selection window | `Fixed_VAR_roll42_l5` | 0.489487 | | | 27 | rejected as standalone/ensemble candidate; experimental only | `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/robustness_fixed_roll42_l5_full_no_subcomp/` |
| robustness_fixed_roll42_l5_subcomp_representative | 2026-06-07 | Representative cutoff-safe SubcomponentMulti comparison | `Fixed_VAR_roll42_l5` | 0.575812 | | | 2/5 | SubcomponentMulti sanity comparison; fixed VAR worse | `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/robustness_fixed_roll42_l5_subcomp_representative/` |

## Best Candidate So Far

`fine_seasonal_resid_var_tc_roll42_l5` from `iter03_seasonal_fine`.

- Configuration:
  - Family: train-only seasonal residual classical VAR.
  - Data source: official `data/inflation_data.csv`.
  - Variables: `CPI`, `Food`, `NonFood`, `Services`.
  - Seasonal factor: trailing 42-month month-of-year mean, computed inside each cutoff.
  - Core model: `statsmodels` VAR on residuals, lags `5`.
  - Forecast reconstruction: residual forecast plus target-month seasonal mean.
- h=1 MAE: `0.223115`
- h=2 MAE: `0.308349`
- h=12 MAE: `0.413011`
- h=1 KPI violations: `2`
- Robustness h=1 MAE outside selection window: `0.489487` expanding, `0.499687` rolling120.
- Status: experimental only; h=1 was strong on the selected window, but out-of-selection robustness rejects standalone and ensemble-candidate use.

## Open Questions For The Next Agent

- Do SA files represent real-time vintage data or revised full-history SA estimates? Current evidence: `data/mom_sa_kbr.csv` and related SA files are full-history revised estimates through 2026-04, so SA backtests have revision leakage risk unless vintage files are found.
- Should the best model optimize only h=1, or must h=12 remain competitive enough for annual paths?
- Should a successful VAR/BVAR enter production as a standalone model or only as an ensemble feature/component?

### 2026-06-07 Opus 4.8 Independent Audit (cross-reference)

- An independent re-run audited `fine_seasonal_resid_var_tc_roll42_l5`.
- Result: metrics reproduced exactly (max |Δ|=0.0), **no future leakage** (numerical probe +
  code review), but the h=1 MAE 0.223 is a **single-window grid-selection overfit**: frozen
  out-of-selection h=1 MAE averages ~0.435 over 6 other 12-month windows, and `roll42` is an
  isolated trough (neighbor `roll36` is the worst row), not a plateau.
- Final audit status: `experimental only` for the seasonal-residual-VAR family; the specific
  `roll42_l5` config is `rejected` as a standalone production model.
- Full report: `experiments/var_sa_research/opus48_review_report.md`
- Artifacts: `runs/opus48_reproduce_iter03/`, `runs/opus48_robustness/`, `opus48_robustness.py`.
