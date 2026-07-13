# codex_cli Robust VAR Report

## Scope

Research-only robust/outlier-aware h=1 VAR-family evaluation. No production data, registry, dashboard, or shared task file was modified.

## Methods Tested

| direction                  |   candidate_count |
|:---------------------------|------------------:|
| additive_outlier_detection |                 2 |
| baseline                   |                 7 |
| context                    |                 2 |
| intervention_dummies       |                 2 |
| regime_macro_varx          |                 2 |
| robust_equation_var        |                 2 |
| robust_favar               |                 1 |
| student_t_bvar_approx      |                 1 |
| winsorized_training        |                 1 |

Robust directions include deterministic intervention dummies, train-only additive outlier pulse detection, winsorized VAR/BVAR, equation-by-equation Huber VAR, regime-aware macro VARX, and robust FAVAR.

## Recommendation

- Final robust VAR model: `robust_favar_mad_f2_l1`
- Final status: `recommended robust mandatory VAR`
- Decision: Replaces `PlainVAR_BIC` as the robust mandatory VAR-family benchmark, but not production ML.
- `PlainVAR_BIC` remains the explicit incumbent acceptance bar from the task.
- Huber and RidgeShockDummies are external production context only, not VAR candidates.

## h=1 Metrics

| model                            | role               | family             |   all_windows_MAE |   out_of_selection_MAE |   out_of_selection_non_shock_MAE |   shock_2022_MAE |   all_windows_KPI |   shock_2022_KPI |
|:---------------------------------|:-------------------|:-------------------|------------------:|-----------------------:|---------------------------------:|-----------------:|------------------:|-----------------:|
| RidgeShockDummies                | external_context   | ML_context         |          0.348562 |               0.355622 |                         0.322756 |         0.561035 |                19 |                2 |
| Huber                            | external_context   | ML_context         |          0.360858 |               0.365295 |                         0.336133 |         0.547562 |                24 |                1 |
| favar_macro_components_f2_l1     | mandatory_baseline | FAVAR              |          0.387324 |               0.38746  |                         0.315293 |         0.838501 |                23 |                3 |
| robust_favar_mad_f2_l1           | robust_candidate   | Robust_FAVAR       |          0.388305 |               0.383858 |                         0.314525 |         0.817189 |                27 |                5 |
| huber_macro_varx_l1              | robust_candidate   | Robust_VARX        |          0.395792 |               0.396001 |                         0.319714 |         0.872799 |                24 |                4 |
| huber_var_l1                     | robust_candidate   | Robust_VAR         |          0.3969   |               0.398603 |                         0.325951 |         0.852677 |                25 |                5 |
| additive_outlier_var_l1_z35      | robust_candidate   | VARX_pulse_outlier |          0.398024 |               0.400803 |                         0.326119 |         0.867582 |                23 |                5 |
| additive_outlier_var_l1_z40      | robust_candidate   | VARX_pulse_outlier |          0.398288 |               0.401284 |                         0.324973 |         0.878228 |                24 |                5 |
| PlainVAR_BIC                     | mandatory_baseline | VAR                |          0.400753 |               0.401016 |                         0.318285 |         0.918084 |                28 |                6 |
| plain_var_tc_l1                  | mandatory_baseline | VAR                |          0.400753 |               0.401016 |                         0.318285 |         0.918084 |                28 |                6 |
| varx_last_exog_l1                | mandatory_baseline | VARX_OLS           |          0.401564 |               0.403954 |                         0.306384 |         1.01377  |                19 |                3 |
| winsorized_var_l1_q05            | robust_candidate   | VAR_winsorized     |          0.414202 |               0.415276 |                         0.322758 |         0.993514 |                26 |                6 |
| winsorized_bvar_l1_q05           | robust_candidate   | BVAR_winsorized    |          0.417263 |               0.418236 |                         0.324102 |         1.00658  |                29 |                6 |
| intervention_varx_macro_l1       | robust_candidate   | VARX_deterministic |          0.425994 |               0.427222 |                         0.317173 |         1.11503  |                23 |                5 |
| SeasonalNaive                    | simple_baseline    | Naive              |          0.439718 |               0.44692  |                         0.373116 |         0.908194 |                33 |                5 |
| regime_macro_varx_shock_guard_l1 | robust_candidate   | Regime_VARX        |          0.444611 |               0.451719 |                         0.311466 |         1.3283   |                23 |                6 |
| intervention_var_l1              | robust_candidate   | VARX_deterministic |          0.457365 |               0.460485 |                         0.32268  |         1.32177  |                27 |                6 |
| huber_intervention_var_l1        | robust_candidate   | Robust_VARX        |          0.462942 |               0.469734 |                         0.332364 |         1.3283   |                26 |                6 |
| Archived_BVAR                    | mandatory_baseline | BVAR               |          0.480103 |               0.47193  |                         0.383074 |         1.02728  |                32 |                6 |
| RandomWalk                       | simple_baseline    | Naive              |          0.524242 |               0.514828 |                         0.405733 |         1.19667  |                34 |                6 |

## Interpretation

- Versus `PlainVAR_BIC`: all-window MAE 0.388305 vs 0.400753; 2022 MAE 0.817189 vs 0.918084.
- Non-shock out-of-selection MAE 0.314525 vs 0.318285; all-window KPI violations 27 vs 28.
- Versus non-robust FAVAR: all-window MAE is 0.388305 vs 0.387324; robust preprocessing helps 2022 slightly but is not a broad accuracy breakthrough.
- Versus macro VARX: non-shock MAE is 0.314525 vs 0.306384; 2022 MAE is 0.817189 vs 1.013766.
- Deterministic intervention dummies and the simple regime guard did not fix 2022; in this validation they often made the shock slice worse.
- Additive pulse dummies and Huber VAR reduce some damage relative to plain VAR, but they do not beat robust FAVAR.

## Outlier / Dummy Evidence

- Predeclared deterministic dummies: 2022 year, March-April 2022, COVID Q2 2020, and July tariff/admin seasonality.
- Additive outlier pulse dummies are detected separately inside each cutoff from baseline VAR(1) train residual MAD scores; target-month pulse dummies are zero.

| model                       | detected_date       |   count |   mean_max_z | max_variable   |
|:----------------------------|:--------------------|--------:|-------------:|:---------------|
| additive_outlier_var_l1_z35 | 2010-09-01 00:00:00 |      99 |      5.45773 | Food           |
| additive_outlier_var_l1_z35 | 2011-01-01 00:00:00 |      99 |      4.44343 | Services       |
| additive_outlier_var_l1_z35 | 2012-07-01 00:00:00 |      99 |     10.7989  | Services       |
| additive_outlier_var_l1_z35 | 2013-07-01 00:00:00 |      99 |     10.1831  | Services       |
| additive_outlier_var_l1_z35 | 2014-10-01 00:00:00 |      99 |      4.50644 | Services       |
| additive_outlier_var_l1_z35 | 2015-01-01 00:00:00 |      99 |      7.67509 | NonFood        |
| additive_outlier_var_l1_z35 | 2015-04-01 00:00:00 |      99 |      7.76911 | Services       |
| additive_outlier_var_l1_z35 | 2015-07-01 00:00:00 |      99 |      7.38154 | Services       |
| additive_outlier_var_l1_z35 | 2016-12-01 00:00:00 |      99 |      5.58991 | NonFood        |
| additive_outlier_var_l1_z35 | 2014-12-01 00:00:00 |      85 |      3.98453 | NonFood        |
| additive_outlier_var_l1_z35 | 2022-03-01 00:00:00 |      48 |     18.9001  | NonFood        |
| additive_outlier_var_l1_z35 | 2010-06-01 00:00:00 |      47 |      3.64094 | Services       |
| additive_outlier_var_l1_z35 | 2015-03-01 00:00:00 |      47 |      4.17906 | NonFood        |
| additive_outlier_var_l1_z35 | 2022-04-01 00:00:00 |      47 |      4.27743 | NonFood        |
| additive_outlier_var_l1_z35 | 2022-07-01 00:00:00 |      44 |      5.88072 | Services       |
| additive_outlier_var_l1_z35 | 2016-07-01 00:00:00 |      42 |      3.61686 | Services       |
| additive_outlier_var_l1_z35 | 2022-12-01 00:00:00 |      39 |      4.57114 | Services       |
| additive_outlier_var_l1_z35 | 2024-07-01 00:00:00 |      20 |      5.57314 | Services       |
| additive_outlier_var_l1_z35 | 2016-09-01 00:00:00 |      12 |      3.59968 | Services       |
| additive_outlier_var_l1_z35 | 2025-07-01 00:00:00 |       8 |      6.8237  | Services       |
| additive_outlier_var_l1_z35 | 2015-02-01 00:00:00 |       2 |      3.53723 | NonFood        |
| additive_outlier_var_l1_z35 | 2023-09-01 00:00:00 |       1 |      3.50323 | CPI            |
| additive_outlier_var_l1_z40 | 2010-09-01 00:00:00 |      99 |      5.45773 | Food           |
| additive_outlier_var_l1_z40 | 2012-07-01 00:00:00 |      99 |     10.7989  | Services       |
| additive_outlier_var_l1_z40 | 2013-07-01 00:00:00 |      99 |     10.1831  | Services       |
| additive_outlier_var_l1_z40 | 2014-10-01 00:00:00 |      99 |      4.50644 | Services       |
| additive_outlier_var_l1_z40 | 2015-01-01 00:00:00 |      99 |      7.67509 | NonFood        |
| additive_outlier_var_l1_z40 | 2015-04-01 00:00:00 |      99 |      7.76911 | Services       |
| additive_outlier_var_l1_z40 | 2015-07-01 00:00:00 |      99 |      7.38154 | Services       |
| additive_outlier_var_l1_z40 | 2016-12-01 00:00:00 |      99 |      5.58991 | NonFood        |

## Leakage Audit

- Every target uses `cutoff = target_date - 1 month`.
- Outlier detection uses only residuals from rows at or before the cutoff.
- VARX macro target scenarios use last observed cutoff values for USD/Ruonia/Ki_i.
- Predeclared calendar dummies are allowed for target dates but are not selected from forecast errors.
- Leakage violations in `leakage_checks.csv`: `0`.

## h=2 / h=12

Not tested. The robust task is h=1-primary; extending regime Macro VARX to h=2/h=12 needs explicit recursive exogenous scenario paths rather than reusing one-step last-observed paths.

## Commands Run

```bash
python3 experiments/var_sa_research/codex_cli_robust_var.py --run-name codex_cli_robust_var_h1_full_v2
```

## Artifacts

- Run directory: `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/codex_cli_robust_var_h1_full_v2`
- Required files: `config.json`, `metrics.csv`, `predictions.csv`, `comparison.csv`, `outlier_log.csv`, `selection_log.csv`, `leakage_checks.csv`, `notes.md`, script copy.
