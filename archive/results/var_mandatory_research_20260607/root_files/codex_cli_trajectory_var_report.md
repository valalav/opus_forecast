# codex_cli Parsimonious Trajectory VAR Report

## Scope

Research-only deterministic VAR-family evaluation for h=1, h=2, and h=12. No production data, model registry, dashboard, or shared task file was modified. No random noise was added to point paths.

## Variable Subsets Tested

| set                  | variables                 |
|:---------------------|:--------------------------|
| cpi                  | CPI                       |
| tc                   | CPI,Food,NonFood,Services |
| comp                 | Food,NonFood,Services     |
| cpi_food             | CPI,Food                  |
| cpi_nonfood          | CPI,NonFood               |
| cpi_services         | CPI,Services              |
| cpi_food_nonfood     | CPI,Food,NonFood          |
| cpi_food_services    | CPI,Food,Services         |
| cpi_nonfood_services | CPI,NonFood,Services      |

## Macro / Exogenous Sets Tested

| set             | variables       |
|:----------------|:----------------|
| none            | none            |
| USD             | USD             |
| Ki_i            | Ki_i            |
| Ruonia          | Ruonia          |
| USD_Ki_i        | USD,Ki_i        |
| USD_Ruonia      | USD,Ruonia      |
| Ki_i_Ruonia     | Ki_i,Ruonia     |
| USD_Ki_i_Ruonia | USD,Ki_i,Ruonia |

Exogenous paths tested: `last`, `ar1`, `mean_revert`, and `rate_hold_usd_ar`. All are fit or declared from cutoff data only.

h=1 and h=2 use the full candidate grid. h=12 uses a transparent shortlist after the full h=12 grid proved too slow: all baselines, all no-macro endogenous/component sets, all macro subsets/path rules on total-components, and all BVAR/FAVAR candidates.

## Recommendation

- Final status: `recommended horizon-specific VAR policy`
- Decision: Horizon-specific policy is preferred because short-horizon and h=12 trajectory criteria select different parsimonious models.

|   horizon | selected_model              | selected_family   |      MAE |   rank_score |
|----------:|:----------------------------|:------------------|---------:|-------------:|
|         1 | RegimeMacroVARX_l1          | Regime_VARX       | 0.378618 |     0.378618 |
|         2 | seasonal_var_cpi_nonfood_l1 | Seasonal_VAR      | 0.43998  |     0.43998  |
|        12 | seasonal_var_comp_l1        | Seasonal_VAR      | 0.4391   |     0.446713 |

## h=1 / h=2 / h=12 Metrics And Trajectory Diagnostics

| model                                    | role            | family         |   h1_MAE |   h2_MAE |   h12_MAE |   h12_path_std_mean |   h12_vol_ratio_mean |   h12_flatness_mean |   h12_explosive_rate |   h12_flat_path_rate |
|:-----------------------------------------|:----------------|:---------------|---------:|---------:|----------:|--------------------:|---------------------:|--------------------:|---------------------:|---------------------:|
| seasonal_var_cpi_nonfood_l1              | candidate       | Seasonal_VAR   | 0.38756  | 0.43998  |  0.439044 |           0.349024  |             0.731535 |           0.0780533 |               0      |               0      |
| seasonal_var_comp_l1                     | candidate       | Seasonal_VAR   | 0.397033 | 0.440463 |  0.4391   |           0.338536  |             0.709319 |           0.0449954 |               0      |               0      |
| seasonal_var_cpi_nonfood_services_l1     | candidate       | Seasonal_VAR   | 0.400109 | 0.442929 |  0.441148 |           0.351635  |             0.736585 |           0.0826446 |               0      |               0      |
| seasonal_var_cpi_food_nonfood_l1         | candidate       | Seasonal_VAR   | 0.400369 | 0.442725 |  0.441306 |           0.352106  |             0.738221 |           0.0817264 |               0      |               0      |
| seasonal_var_cpi_food_services_l1        | candidate       | Seasonal_VAR   | 0.400799 | 0.443935 |  0.441008 |           0.351501  |             0.73596  |           0.0863177 |               0      |               0      |
| seasonal_var_tc_l1                       | candidate       | Seasonal_VAR   | 0.40131  | 0.443824 |  0.441281 |           0.351964  |             0.737537 |           0.0835629 |               0      |               0      |
| seasonal_var_cpi_l1                      | candidate       | Seasonal_VAR   | 0.407952 | 0.445298 |  0.438064 |           0.349808  |             0.734955 |           0.0881543 |               0      |               0      |
| seasonal_var_cpi_services_l1             | candidate       | Seasonal_VAR   | 0.412611 | 0.443334 |  0.440846 |           0.352695  |             0.740422 |           0.0918274 |               0      |               0      |
| seasonal_var_cpi_food_l1                 | candidate       | Seasonal_VAR   | 0.413801 | 0.45192  |  0.438958 |           0.350654  |             0.736951 |           0.0909091 |               0      |               0      |
| SeasonalNaive                            | simple_baseline | Naive          | 0.439718 | 0.439718 |  0.439718 |           0.352197  |             0.749111 |           0.0863177 |               0      |               0      |
| seasonal_varx_tc_Ki_i_last_l1            | candidate       | Seasonal_VARX  | 0.392661 | 0.484086 |  0.601433 |           0.300079  |             0.639778 |           0.15978   |               1.0101 |               0      |
| seasonal_varx_tc_Ruonia_last_l1          | candidate       | Seasonal_VARX  | 0.43809  | 0.511294 |  0.524594 |           0.328602  |             0.693472 |           0.103765  |               0      |               0      |
| seasonal_varx_tc_Ki_i_Ruonia_last_l1     | candidate       | Seasonal_VARX  | 0.401891 | 0.500225 |  0.595748 |           0.295846  |             0.636535 |           0.156107  |               1.0101 |               0      |
| Archived_BVAR                            | baseline        | BVAR           | 0.435738 | 0.501708 |  0.473063 |           0.146677  |             0.246728 |           0.632691  |               1.0101 |              19.1919 |
| FAVAR_f2_l1                              | baseline        | FAVAR          | 0.387324 | 0.485414 |  0.487202 |           0.104688  |             0.190186 |           0.747475  |               1.0101 |              43.4343 |
| RobustFAVAR_f2_l1                        | baseline        | Robust_FAVAR   | 0.388305 | 0.486735 |  0.486537 |           0.0993567 |             0.196555 |           0.730028  |               0      |              44.4444 |
| RegimeMacroVARX_l1                       | baseline        | Regime_VARX    | 0.378618 | 0.474414 |  0.545928 |           0.104952  |             0.214304 |           0.764922  |               0      |              49.4949 |
| seasonal_varx_tc_USD_Ki_i_Ruonia_last_l1 | candidate       | Seasonal_VARX  | 0.389415 | 0.521169 |  0.677738 |           0.322981  |             0.686251 |           0.161616  |               1.0101 |               0      |
| varx_tc_USD_Ruonia_mean_revert_l1        | candidate       | VARX           | 0.399076 | 0.489356 |  0.525336 |           0.11933   |             0.21697  |           0.685032  |               1.0101 |              42.4242 |
| varx_tc_USD_mean_revert_l1               | candidate       | VARX           | 0.403182 | 0.483342 |  0.530629 |           0.124972  |             0.215095 |           0.640955  |               1.0101 |              42.4242 |
| BVAR_comp_bottomup_l1_lam0p3             | candidate       | BVAR_component | 0.399595 | 0.480074 |  0.481891 |           0.0784818 |             0.146081 |           0.814509  |               0      |              62.6263 |
| var_cpi_food_services_l1                 | candidate       | VAR            | 0.397937 | 0.482922 |  0.482836 |           0.0781697 |             0.14282  |           0.816345  |               0      |              63.6364 |
| seasonal_varx_tc_USD_Ruonia_last_l1      | candidate       | Seasonal_VARX  | 0.397818 | 0.529205 |  0.686272 |           0.345876  |             0.726309 |           0.118457  |               1.0101 |               0      |
| varx_tc_USD_Ki_i_Ruonia_ar1_l1           | candidate       | VARX           | 0.40953  | 0.499928 |  0.494133 |           0.0951956 |             0.172043 |           0.766758  |               1.0101 |              50.5051 |
| var_cpi_nonfood_services_l1              | candidate       | VAR            | 0.397563 | 0.482527 |  0.483111 |           0.0786719 |             0.144606 |           0.815427  |               0      |              64.6465 |
| var_comp_l1                              | candidate       | VAR            | 0.39666  | 0.481239 |  0.482833 |           0.076729  |             0.141883 |           0.820937  |               0      |              65.6566 |
| var_cpi_food_nonfood_l1                  | candidate       | VAR            | 0.398231 | 0.48219  |  0.483233 |           0.0791792 |             0.146896 |           0.818182  |               0      |              64.6465 |
| seasonal_varx_tc_USD_Ki_i_last_l1        | candidate       | Seasonal_VARX  | 0.3876   | 0.51599  |  0.721756 |           0.332015  |             0.690879 |           0.161616  |               1.0101 |               0      |
| BVAR_tc_l1_lam0p3                        | candidate       | BVAR           | 0.402852 | 0.48093  |  0.481828 |           0.0793821 |             0.146452 |           0.816345  |               0      |              63.6364 |
| var_cpi_services_l1                      | candidate       | VAR            | 0.399509 | 0.487406 |  0.48192  |           0.0780616 |             0.14529  |           0.8191    |               0      |              63.6364 |
| var_tc_l1                                | candidate       | VAR            | 0.399138 | 0.483213 |  0.483131 |           0.0781962 |             0.143355 |           0.817264  |               0      |              64.6465 |
| PlainVAR_BIC                             | baseline        | VAR_BIC        | 0.400753 | 0.484539 |  0.483059 |           0.0774714 |             0.141063 |           0.814509  |               0      |              65.6566 |
| varx_tc_USD_Ki_i_mean_revert_l1          | candidate       | VARX           | 0.398532 | 0.500341 |  0.531319 |           0.118187  |             0.201279 |           0.701561  |               1.0101 |              49.4949 |
| varx_tc_USD_Ki_i_ar1_l1                  | candidate       | VARX           | 0.406547 | 0.495321 |  0.483726 |           0.0909625 |             0.162749 |           0.79247   |               1.0101 |              60.6061 |
| varx_tc_USD_ar1_l1                       | candidate       | VARX           | 0.411595 | 0.500121 |  0.483569 |           0.095058  |             0.170461 |           0.774105  |               1.0101 |              57.5758 |

## Rejected Unrealistic Paths

| model                                    | role      | family         |   h1_MAE |   h2_MAE |   h12_MAE |   h12_path_std_mean |   h12_vol_ratio_mean |   h12_flatness_mean |   h12_explosive_rate |   h12_flat_path_rate |
|:-----------------------------------------|:----------|:---------------|---------:|---------:|----------:|--------------------:|---------------------:|--------------------:|---------------------:|---------------------:|
| seasonal_varx_tc_Ki_i_last_l1            | candidate | Seasonal_VARX  | 0.392661 | 0.484086 |  0.601433 |           0.300079  |             0.639778 |            0.15978  |               1.0101 |               0      |
| seasonal_varx_tc_Ki_i_Ruonia_last_l1     | candidate | Seasonal_VARX  | 0.401891 | 0.500225 |  0.595748 |           0.295846  |             0.636535 |            0.156107 |               1.0101 |               0      |
| Archived_BVAR                            | baseline  | BVAR           | 0.435738 | 0.501708 |  0.473063 |           0.146677  |             0.246728 |            0.632691 |               1.0101 |              19.1919 |
| FAVAR_f2_l1                              | baseline  | FAVAR          | 0.387324 | 0.485414 |  0.487202 |           0.104688  |             0.190186 |            0.747475 |               1.0101 |              43.4343 |
| seasonal_varx_tc_USD_Ki_i_Ruonia_last_l1 | candidate | Seasonal_VARX  | 0.389415 | 0.521169 |  0.677738 |           0.322981  |             0.686251 |            0.161616 |               1.0101 |               0      |
| varx_tc_USD_Ruonia_mean_revert_l1        | candidate | VARX           | 0.399076 | 0.489356 |  0.525336 |           0.11933   |             0.21697  |            0.685032 |               1.0101 |              42.4242 |
| varx_tc_USD_mean_revert_l1               | candidate | VARX           | 0.403182 | 0.483342 |  0.530629 |           0.124972  |             0.215095 |            0.640955 |               1.0101 |              42.4242 |
| BVAR_comp_bottomup_l1_lam0p3             | candidate | BVAR_component | 0.399595 | 0.480074 |  0.481891 |           0.0784818 |             0.146081 |            0.814509 |               0      |              62.6263 |
| var_cpi_food_services_l1                 | candidate | VAR            | 0.397937 | 0.482922 |  0.482836 |           0.0781697 |             0.14282  |            0.816345 |               0      |              63.6364 |
| seasonal_varx_tc_USD_Ruonia_last_l1      | candidate | Seasonal_VARX  | 0.397818 | 0.529205 |  0.686272 |           0.345876  |             0.726309 |            0.118457 |               1.0101 |               0      |
| varx_tc_USD_Ki_i_Ruonia_ar1_l1           | candidate | VARX           | 0.40953  | 0.499928 |  0.494133 |           0.0951956 |             0.172043 |            0.766758 |               1.0101 |              50.5051 |
| var_cpi_nonfood_services_l1              | candidate | VAR            | 0.397563 | 0.482527 |  0.483111 |           0.0786719 |             0.144606 |            0.815427 |               0      |              64.6465 |
| var_comp_l1                              | candidate | VAR            | 0.39666  | 0.481239 |  0.482833 |           0.076729  |             0.141883 |            0.820937 |               0      |              65.6566 |
| var_cpi_food_nonfood_l1                  | candidate | VAR            | 0.398231 | 0.48219  |  0.483233 |           0.0791792 |             0.146896 |            0.818182 |               0      |              64.6465 |
| seasonal_varx_tc_USD_Ki_i_last_l1        | candidate | Seasonal_VARX  | 0.3876   | 0.51599  |  0.721756 |           0.332015  |             0.690879 |            0.161616 |               1.0101 |               0      |
| BVAR_tc_l1_lam0p3                        | candidate | BVAR           | 0.402852 | 0.48093  |  0.481828 |           0.0793821 |             0.146452 |            0.816345 |               0      |              63.6364 |
| var_cpi_services_l1                      | candidate | VAR            | 0.399509 | 0.487406 |  0.48192  |           0.0780616 |             0.14529  |            0.8191   |               0      |              63.6364 |
| var_tc_l1                                | candidate | VAR            | 0.399138 | 0.483213 |  0.483131 |           0.0781962 |             0.143355 |            0.817264 |               0      |              64.6465 |
| PlainVAR_BIC                             | baseline  | VAR_BIC        | 0.400753 | 0.484539 |  0.483059 |           0.0774714 |             0.141063 |            0.814509 |               0      |              65.6566 |
| varx_tc_USD_Ki_i_mean_revert_l1          | candidate | VARX           | 0.398532 | 0.500341 |  0.531319 |           0.118187  |             0.201279 |            0.701561 |               1.0101 |              49.4949 |

## Leakage And Determinism Audit

- Every target uses `cutoff = target_date - horizon months`.
- Exogenous paths are deterministic and use only cutoff-observed history.
- BVAR candidates use posterior mean recursion only; no Monte Carlo draws or shock noise are used.
- Month and tariff dummies are deterministic date functions.
- Leakage/random-noise violations in `leakage_checks.csv`: `0`.

## Charts

- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/codex_cli_trajectory_var_h1_h2_h12_full_v2/trajectory_2026_03_top_h12.png`
- `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/codex_cli_trajectory_var_h1_h2_h12_full_v2/top_candidates_horizon_mae.png`

## Commands Run

```bash
python3 experiments/var_sa_research/codex_cli_trajectory_var.py --run-name codex_cli_trajectory_var_h1_h2_h12_full_v2
```

## Artifacts

- Run directory: `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/codex_cli_trajectory_var_h1_h2_h12_full_v2`
- Required files: `config.json`, `metrics.csv`, `predictions.csv`, `comparison.csv`, `selection_log.csv`, `trajectory_metrics.csv`, `leakage_checks.csv`, `notes.md`, trajectory charts, script copy.
