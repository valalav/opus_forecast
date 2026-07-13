# codex_cli Nested Re-Selection Report

Agent: `codex_cli`  
Run directory: `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/codex_cli_nested_h1_full/`  
Script: `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/codex_cli_nested_reselection.py`

## Question

Does the seasonal-residual VAR family retain deployable value when roll-window, lag count, and variant are selected honestly inside each cutoff, instead of being selected on the final 12-month evaluation window?

## Nested Method

Nested re-selection was implemented for h=1.

For each outer target date:

1. Set `cutoff = target_date - 1 month`.
2. Use only observations available at or before `cutoff`.
3. Use the last 24 target months inside that cutoff as inner validation.
4. For every inner validation target, use its own `inner_cutoff = inner_target - 1 month`.
5. Score every candidate by inner MAE, requiring at least 12 valid inner predictions.
6. Select the lowest inner-MAE candidate.
7. Forecast the outer target using the selected candidate and data available at the outer cutoff.

Candidate grid:

- Variants: `total_components`, `component_bottomup`
- Roll windows: `24, 30, 36, 42, 48, 60, 72`
- VAR lags: `1..6`
- Train modes: `expanding`, `rolling120`
- Official data only: `data/inflation_data.csv`

Outer windows:

- `2018-01..2019-12`
- `2020-01..2021-12`
- `2022-01..2022-12`
- `2023-01..2023-12`
- `2024-01..2025-03`
- selection/reference window `2025-04..2026-03`

The implementation precomputed cutoff-correct one-step forecasts for each grid candidate, then used only already-available inner errors for selection. This is computationally different from refitting inside each outer loop, but methodologically equivalent because each precomputed prediction used its own correct cutoff.

Leakage checks: `0` violations across `198` nested outer forecasts. The checks verify that inner validation ends at or before the outer cutoff and the final outer training data do not exceed the outer cutoff.

## Results

Overall h=1, all windows:

| train mode | model | n | MAE | RMSE | KPI violations | coverage |
|---|---:|---:|---:|---:|---:|---:|
| expanding | RidgeShockDummies | 99 | 0.348562 | 0.623843 | 19 | 80.81% |
| expanding | Huber | 99 | 0.360858 | 0.620911 | 24 | 75.76% |
| expanding | PlainVAR_l5 | 99 | 0.439669 | 0.729881 | 32 | 67.68% |
| expanding | Archived_BVAR | 99 | 0.451817 | 0.723489 | 28 | 71.72% |
| expanding | NestedSeasonalResidualVAR | 99 | 0.458159 | 0.717446 | 32 | 67.68% |
| rolling120 | RidgeShockDummies | 99 | 0.374176 | 0.645705 | 22 | 77.78% |
| rolling120 | Huber | 99 | 0.374867 | 0.631004 | 25 | 74.75% |
| rolling120 | NestedSeasonalResidualVAR | 99 | 0.467674 | 0.711892 | 32 | 67.68% |
| rolling120 | Archived_BVAR | 99 | 0.474281 | 0.750150 | 31 | 68.69% |
| rolling120 | PlainVAR_l5 | 99 | 0.490053 | 0.791182 | 35 | 64.65% |

Out-of-selection only, excluding `2025-04..2026-03`:

| train mode | model | n | MAE | RMSE | KPI violations | coverage |
|---|---:|---:|---:|---:|---:|---:|
| expanding | RidgeShockDummies | 87 | 0.355622 | 0.652310 | 17 | 80.46% |
| expanding | Huber | 87 | 0.365295 | 0.646526 | 21 | 75.86% |
| expanding | Archived_BVAR | 87 | 0.436817 | 0.726193 | 23 | 73.56% |
| expanding | PlainVAR_l5 | 87 | 0.454887 | 0.760399 | 28 | 67.82% |
| expanding | NestedSeasonalResidualVAR | 87 | 0.472825 | 0.743523 | 28 | 67.82% |
| rolling120 | RidgeShockDummies | 87 | 0.375591 | 0.669954 | 20 | 77.01% |
| rolling120 | Huber | 87 | 0.379048 | 0.653672 | 22 | 74.71% |
| rolling120 | Archived_BVAR | 87 | 0.445810 | 0.740162 | 25 | 71.26% |
| rolling120 | NestedSeasonalResidualVAR | 87 | 0.476976 | 0.733712 | 28 | 67.82% |
| rolling120 | PlainVAR_l5 | 87 | 0.495199 | 0.814007 | 29 | 66.67% |

Out-of-selection non-shock, excluding both selection and `2022`:

| train mode | model | n | MAE |
|---|---:|---:|---:|
| expanding | RidgeShockDummies | 75 | 0.322756 |
| expanding | Huber | 75 | 0.336133 |
| expanding | Archived_BVAR | 75 | 0.350593 |
| expanding | PlainVAR_l5 | 75 | 0.370424 |
| expanding | NestedSeasonalResidualVAR | 75 | 0.427159 |
| rolling120 | Huber | 75 | 0.339139 |
| rolling120 | RidgeShockDummies | 75 | 0.343695 |
| rolling120 | Archived_BVAR | 75 | 0.352090 |
| rolling120 | PlainVAR_l5 | 75 | 0.387684 |
| rolling120 | NestedSeasonalResidualVAR | 75 | 0.419596 |

On the previous selection/reference window only:

- Expanding nested MAE: `0.351833`
- Rolling120 nested MAE: `0.400238`
- Previous fixed `roll42_l5` MAE on the same window was `0.223115`

Nested selection therefore does not reproduce the headline `0.223` result when hyperparameters are selected inside each cutoff.

## Baseline Comparison

Against simple baselines:

- Expanding nested seasonal-residual VAR beats `RandomWalk` and `SeasonalNaive_roll42`, but loses to `PlainVAR_l5` overall and out-of-selection.
- Rolling120 nested seasonal-residual VAR beats `PlainVAR_l5` overall and out-of-selection, but only modestly, and still does not beat project baselines.

Against project baselines:

- It does not beat `Huber`.
- It does not beat `RidgeShockDummies`.
- It does not beat archived-style BVAR out-of-selection.
- `Ridge_ProdProxy_Roll24` had only 76 valid rows because early windows fail minimum-train or NaN checks; its comparison is incomplete and not decisive here.
- `SubcomponentMulti` was not run in the nested loop because cutoff-safe full retraining is too expensive for this task. Prior codex robustness used a representative cutoff-safe check; this run focuses on the nested seasonal-residual family.

## Hyperparameter Stability

Selected hyperparameters are noisy.

For expanding selection:

- `component_bottomup` was selected `69/99` times.
- `total_components` was selected `30/99` times.
- Most frequent exact config was `component_bottomup roll72 lag4`, selected only `12/99` times.
- The old fixed `total_components roll42 lag5` was selected `0/99` times.

For rolling120 selection:

- `component_bottomup` was selected `79/99` times.
- `total_components` was selected `20/99` times.
- Most frequent exact config was `component_bottomup roll72 lag3`, selected only `13/99` times.
- The old fixed `total_components roll42 lag5` was selected `1/99` times.

This is not a stable deployment pattern. Nested selection generally shifts away from the earlier `roll42_l5` trough toward component-bottom-up variants and longer roll windows, but the resulting forecasts remain uncompetitive.

## Deployable Signal

Evidence of deployable signal is weak.

The nested family has some signal relative to random walk and seasonal naive, especially in rolling120 mode, but it fails the more relevant tests:

- It does not beat robust project baselines.
- It does not improve over expanding plain VAR.
- It has high KPI violations: `28/87` out-of-selection in both expanding and rolling120.
- It does not recover the selected-window `0.223` result under honest nested selection.
- Hyperparameter choices are noisy and not concentrated enough for a stable operational rule.

## Final Status

`rejected`

This status is for deployment or ensemble promotion of the seasonal-residual VAR family under the tested nested re-selection protocol. The family remains a valid research idea, but this run does not show sufficient standalone or ensemble value.

## Commands Run

```bash
cd /home/valalav/_projects/sirena-kbr

sed -n '1,260p' experiments/var_sa_research/parallel_nested_reselection_task.md
sed -n '1,280p' experiments/var_sa_research/opus48_review_report.md
sed -n '1,300p' experiments/var_sa_research/opus48_robustness.py
sed -n '1,220p' docs/BACKTEST_METHODOLOGY.md
sed -n '1,180p' docs/MODEL_CATALOG.md

python3 -m py_compile \
  experiments/var_sa_research/codex_cli_nested_reselection.py \
  experiments/var_sa_research/run_var_sa_backtests.py \
  experiments/var_sa_research/run_fixed_config_robustness.py

python3 experiments/var_sa_research/codex_cli_nested_reselection.py \
  --run-name codex_cli_nested_h1_full \
  --seed 20260607

# The heavy run completed all model calculations and wrote CSV artifacts, then
# failed only while formatting notes.md due to a column-name typo. The script was
# fixed and notes/charts were regenerated from the saved CSVs:
python3 -m py_compile experiments/var_sa_research/codex_cli_nested_reselection.py
python3 - <<'PY'
from pathlib import Path
import pandas as pd
from experiments.var_sa_research import codex_cli_nested_reselection as mod
base=Path('experiments/var_sa_research/runs/codex_cli_nested_h1_full')
metrics=pd.read_csv(base/'metrics.csv')
selection_log=pd.read_csv(base/'selection_log.csv')
leakage=pd.read_csv(base/'leakage_checks.csv')
stability=pd.read_csv(base/'selection_stability.csv')
mod.write_notes(base, metrics, selection_log, leakage, stability)
mod.make_charts(base, metrics, selection_log)
(base/'codex_cli_nested_reselection.py').write_text(Path('experiments/var_sa_research/codex_cli_nested_reselection.py').read_text(encoding='utf-8'), encoding='utf-8')
PY

# Extra subset summaries for out-of-selection reporting:
python3 - <<'PY'
from pathlib import Path
import pandas as pd
import numpy as np
base=Path('experiments/var_sa_research/runs/codex_cli_nested_h1_full')
pred=pd.read_csv(base/'predictions.csv', parse_dates=['target_date'])
subsets={
    'all_windows': pred,
    'out_of_selection': pred[pred['window']!='selection_reference_2025q2_2026q1'],
    'out_of_selection_nonshock': pred[~pred['window'].isin(['selection_reference_2025q2_2026q1','sanctions_2022'])],
    'selection_reference_only': pred[pred['window']=='selection_reference_2025q2_2026q1'],
}
rows=[]
for subset, df in subsets.items():
    ok=df.dropna(subset=['error']).copy()
    for keys,g in ok.groupby(['horizon','train_mode','model'], sort=False):
        h,tm,model=keys
        e=g['error'].astype(float); ae=e.abs()
        rows.append({'subset':subset,'horizon':h,'train_mode':tm,'model':model,'n':len(g),'MAE':ae.mean(),'RMSE':np.sqrt((e**2).mean()),'Mean_Error':e.mean(),'Max_Error':ae.max(),'KPI_Violations':int((ae>0.5).sum()),'Coverage_50pct':float((ae<=0.5).mean()*100)})
out=pd.DataFrame(rows).sort_values(['subset','train_mode','MAE','model'])
out.to_csv(base/'subset_metrics.csv', index=False)
pd.read_csv(base/'metrics.csv').assign(kind='window_metrics').to_csv(base/'comparison.csv', index=False)
PY
```

## Artifacts

Run directory:

- `runs/codex_cli_nested_h1_full/config.json`
- `runs/codex_cli_nested_h1_full/candidate_grid_predictions.csv`
- `runs/codex_cli_nested_h1_full/predictions.csv`
- `runs/codex_cli_nested_h1_full/metrics.csv`
- `runs/codex_cli_nested_h1_full/subset_metrics.csv`
- `runs/codex_cli_nested_h1_full/comparison.csv`
- `runs/codex_cli_nested_h1_full/selection_log.csv`
- `runs/codex_cli_nested_h1_full/selection_stability.csv`
- `runs/codex_cli_nested_h1_full/leakage_checks.csv`
- `runs/codex_cli_nested_h1_full/notes.md`
- `runs/codex_cli_nested_h1_full/overall_expanding_mae.png`
- `runs/codex_cli_nested_h1_full/selected_roll_windows_expanding.png`
- `runs/codex_cli_nested_h1_full/selected_roll_windows_rolling120.png`
- `runs/codex_cli_nested_h1_full/codex_cli_nested_reselection.py`

## Remaining Unchecked

- h=2 and h=12 nested re-selection were skipped for runtime; h=1 was prioritized as required.
- SubcomponentMulti was not included in the nested loop because full cutoff-safe retraining is too expensive.
- No optimized ensemble weights were fit; given poor standalone/project-baseline comparison, this was not pursued.
- SA revised-history experiments were not run because the task asks for production-relevant official-data seasonal residual VAR.
