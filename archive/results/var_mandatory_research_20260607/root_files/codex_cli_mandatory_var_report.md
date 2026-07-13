# codex_cli Mandatory VAR Report

## Scope

Research-only h=1 search for the best defensible mandatory VAR-family model. No production data, model registry, dashboard, or shared task file was modified.

## Variants Tested

| direction                 |   candidate_count |
|:--------------------------|------------------:|
| bvar_model_averaging_pool |                18 |
| component_constrained_var |                 4 |
| factor_augmented_var      |                 6 |
| plain_var_bvar_baseline   |                 4 |
| regime_aware_var          |                 2 |
| varx_cutoff_safe_exog     |                 3 |

Implemented families: plain VAR, Minnesota BVAR posterior-mean grid, BVAR inner-validation averaging, VARX with last-observed exogenous scenario paths, train-only FAVAR, two transparent regime VAR/BVAR rules, component-constrained bottom-up VAR, and a VAR-family forecast combination.

## Recommendation

- Final recommended mandatory VAR model: `favar_macro_components_f2_l1`
- Final status: `recommended mandatory VAR`
- External Huber/Ridge rows are context only and are not eligible for the mandatory VAR recommendation.
- Use as mandatory VAR-family benchmark / secondary diagnostic model, not as a claim that VAR beats production ML.

## h=1 Metrics

|   horizon | train_mode   | window      | model                                 | role                   |   n |      MAE |     RMSE |         Bias |   Max_Error |   KPI_Violations |   Coverage_50pct |
|----------:|:-------------|:------------|:--------------------------------------|:-----------------------|----:|---------:|---------:|-------------:|------------:|-----------------:|-----------------:|
|         1 | expanding    | all_windows | RidgeShockDummies                     | external_context       |  99 | 0.348562 | 0.623843 |  0.0286837   |     4.7108  |               19 |          80.8081 |
|         1 | expanding    | all_windows | Huber                                 | external_context       |  99 | 0.360858 | 0.620911 |  0.0698464   |     4.54912 |               24 |          75.7576 |
|         1 | expanding    | all_windows | favar_macro_components_f2_l1          | var_family_candidate   |  99 | 0.387324 | 0.672507 | -0.0049063   |     4.55126 |               23 |          76.7677 |
|         1 | expanding    | all_windows | component_constrained_var_l1          | var_family_candidate   |  99 | 0.396669 | 0.677676 |  0.00678446  |     4.83315 |               27 |          72.7273 |
|         1 | expanding    | all_windows | bvar_det_comp_l1_lam1p0               | var_family_candidate   |  99 | 0.396756 | 0.677364 |  0.00644662  |     4.83126 |               28 |          71.7172 |
|         1 | expanding    | all_windows | favar_macro_components_f2_l2          | var_family_candidate   |  99 | 0.397284 | 0.70863  |  0.00117622  |     4.45864 |               25 |          74.7475 |
|         1 | expanding    | all_windows | bvar_det_tc_l1_lam1p0                 | var_family_candidate   |  99 | 0.39752  | 0.679446 |  0.00572765  |     4.81662 |               28 |          71.7172 |
|         1 | expanding    | all_windows | bvar_det_comp_l1_lam0p3               | var_family_candidate   |  99 | 0.399595 | 0.677085 |  0.00319795  |     4.81633 |               26 |          73.7374 |
|         1 | expanding    | all_windows | bvar_det_comp_l2_lam0p3               | var_family_candidate   |  99 | 0.400568 | 0.678977 | -0.00144055  |     4.82754 |               29 |          70.7071 |
|         1 | expanding    | all_windows | plain_var_tc_l1                       | var_family_candidate   |  99 | 0.400753 | 0.68345  |  0.00417521  |     4.819   |               28 |          71.7172 |
|         1 | expanding    | all_windows | varx_last_exog_l1                     | var_family_candidate   |  99 | 0.401564 | 0.75573  | -0.0147275   |     4.73301 |               19 |          80.8081 |
|         1 | expanding    | all_windows | bvar_det_comp_l2_lam1p0               | var_family_candidate   |  99 | 0.401758 | 0.679647 | -0.00758728  |     4.79633 |               26 |          73.7374 |
|         1 | expanding    | all_windows | bvar_det_tc_l1_lam0p3                 | var_family_candidate   |  99 | 0.402852 | 0.679797 |  0.00128207  |     4.80069 |               28 |          71.7172 |
|         1 | expanding    | all_windows | bvar_det_tc_l2_lam1p0                 | var_family_candidate   |  99 | 0.403376 | 0.680542 | -0.00865207  |     4.78417 |               29 |          70.7071 |
|         1 | expanding    | all_windows | bvar_det_tc_l2_lam0p3                 | var_family_candidate   |  99 | 0.403551 | 0.680273 | -0.00349349  |     4.79723 |               29 |          70.7071 |
|         1 | expanding    | all_windows | regime_var_shock_component_else_plain | var_family_candidate   |  99 | 0.404026 | 0.677751 | -0.000687812 |     4.76702 |               28 |          71.7172 |
|         1 | expanding    | all_windows | component_constrained_var_l2          | var_family_candidate   |  99 | 0.404682 | 0.681945 | -0.0106811   |     4.78394 |               26 |          73.7374 |
|         1 | expanding    | all_windows | component_constrained_var_l3          | var_family_candidate   |  99 | 0.406457 | 0.691318 | -0.0101424   |     4.84105 |               29 |          70.7071 |
|         1 | expanding    | all_windows | BVAR_ModelAverage                     | var_family_combination |  99 | 0.409305 | 0.675246 | -0.00859449  |     4.76012 |               28 |          71.7172 |
|         1 | expanding    | all_windows | bvar_det_tfu_l1_lam1p0                | var_family_candidate   |  99 | 0.410555 | 0.668225 | -0.0427063   |     4.69862 |               29 |          70.7071 |
|         1 | expanding    | all_windows | plain_var_tc_l2                       | var_family_candidate   |  99 | 0.412604 | 0.69511  | -0.0137577   |     4.76702 |               30 |          69.697  |
|         1 | expanding    | all_windows | bvar_det_tfu_l2_lam0p3                | var_family_candidate   |  99 | 0.412842 | 0.66202  | -0.0216769   |     4.59188 |               28 |          71.7172 |
|         1 | expanding    | all_windows | favar_macro_components_f2_l3          | var_family_candidate   |  99 | 0.412953 | 0.717287 | -0.00451615  |     4.51462 |               25 |          74.7475 |
|         1 | expanding    | all_windows | favar_macro_components_f1_l1          | var_family_candidate   |  99 | 0.41606  | 0.698565 | -0.0691775   |     4.5842  |               27 |          72.7273 |
|         1 | expanding    | all_windows | varx_last_exog_l3                     | var_family_candidate   |  99 | 0.416118 | 0.784469 | -0.0544127   |     4.69175 |               25 |          74.7475 |
|         1 | expanding    | all_windows | varx_last_exog_l2                     | var_family_candidate   |  99 | 0.416424 | 0.785031 | -0.0434501   |     4.65303 |               24 |          75.7576 |
|         1 | expanding    | all_windows | component_constrained_var_l4          | var_family_candidate   |  99 | 0.416722 | 0.69609  | -0.0162749   |     4.7834  |               26 |          73.7374 |
|         1 | expanding    | all_windows | bvar_det_tfu_l1_lam0p3                | var_family_candidate   |  99 | 0.4168   | 0.67168  | -0.0283176   |     4.71278 |               27 |          72.7273 |
|         1 | expanding    | all_windows | plain_var_tc_l3                       | var_family_candidate   |  99 | 0.417716 | 0.708471 | -0.0195782   |     4.79205 |               32 |          67.6768 |
|         1 | expanding    | all_windows | bvar_det_tfu_l2_lam1p0                | var_family_candidate   |  99 | 0.420602 | 0.679797 | -0.0414413   |     4.41877 |               26 |          73.7374 |
|         1 | expanding    | all_windows | bvar_det_comp_l2_lam0p1               | var_family_candidate   |  99 | 0.426902 | 0.69251  | -0.00515219  |     4.78519 |               26 |          73.7374 |
|         1 | expanding    | all_windows | bvar_det_comp_l1_lam0p1               | var_family_candidate   |  99 | 0.427205 | 0.69263  | -0.00387546  |     4.76723 |               25 |          74.7475 |
|         1 | expanding    | all_windows | ARIMA_101                             | simple_baseline        |  99 | 0.428925 | 0.692452 | -0.0286038   |     4.76139 |               29 |          70.7071 |
|         1 | expanding    | all_windows | regime_bvar_high_shrink_else_tfu      | var_family_candidate   |  99 | 0.430771 | 0.692434 | -0.0272574   |     4.71278 |               29 |          70.7071 |
|         1 | expanding    | all_windows | plain_var_tc_l4                       | var_family_candidate   |  99 | 0.431257 | 0.713341 | -0.0233969   |     4.68293 |               34 |          65.6566 |

## Leakage Audit

- Outer forecast training cutoff: target month minus one month for every prediction.
- Inner model averaging selection: trailing `24` target months ending at cutoff; no outer actual enters weights.
- FAVAR scaling/PCA is fit separately on each training cutoff only.
- VARX target exogenous path uses the last observed cutoff value, not future actual exogenous values.
- Nested selection leakage violations recorded: `0`.

## Rejected / Weaker Variants

- Individual VARX and FAVAR candidates are retained in metrics but were not automatically preferred unless their cutoff-only h=1 errors won.
- Regime rules are intentionally simple; they did not get extra regime tuning beyond fixed thresholds.
- Rejected `roll42_l5` seasonal-residual tuning was not continued in this run.

## Commands Run

```bash
python3 experiments/var_sa_research/codex_cli_mandatory_var.py --run-name codex_cli_mandatory_var_h1_full_v2
```

## Artifacts

- Run directory: `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/codex_cli_mandatory_var_h1_full_v2`
- `config.json`, `metrics.csv`, `predictions.csv`, `selection_log.csv`, `comparison.csv`, `leakage_checks.csv`, `candidate_predictions.csv`, `notes.md`, and script copy are saved there.
