# Final Mandatory VAR Policy Rolling Backtest

Run directory: `/home/valalav/_projects/sirena-kbr/experiments/var_sa_research/runs/final_var_policy_rolling`

## Models

- `SeasonalVAR_CPI_F_NF_S`: deterministic expanding month-of-year seasonal reconstruction plus VAR(1) on residuals.
- `RegimeMacroVARX_l1`: cutoff-only normal/shock regime; normal uses VARX with `USD`, `Ruonia`, `Ki_i`; shock uses robust Huber VAR without macro exog.
- `Hybrid_VAR_Policy`: h=1 uses `RegimeMacroVARX_l1`; h=12 uses `SeasonalVAR_CPI_F_NF_S`.

No random noise is added to point forecasts.

## Summary Metrics

|   horizon | model                  |   all_MAE |   all_KPI |   oos_MAE |   nonshock_MAE |   shock2022_MAE |
|----------:|:-----------------------|----------:|----------:|----------:|---------------:|----------------:|
|         1 | SeasonalVAR_CPI_F_NF_S |  0.396047 |        29 |  0.403061 |       0.346955 |        0.753721 |
|         1 | RegimeMacroVARX_l1     |  0.379328 |        24 |  0.376304 |       0.303609 |        0.830648 |
|         1 | Hybrid_VAR_Policy      |  0.379328 |        24 |  0.376304 |       0.303609 |        0.830648 |
|        12 | SeasonalVAR_CPI_F_NF_S |  0.440175 |        32 |  0.447508 |       0.373748 |        0.908504 |
|        12 | RegimeMacroVARX_l1     |  0.546615 |        39 |  0.543625 |       0.432888 |        1.23573  |
|        12 | Hybrid_VAR_Policy      |  0.440175 |        32 |  0.447508 |       0.373748 |        0.908504 |

## Recommendation

- Best h=1 by all-window MAE: `RegimeMacroVARX_l1` (0.379).
- Best h=12 by all-window MAE: `SeasonalVAR_CPI_F_NF_S` (0.440).
- Recommended reporting policy: use `Hybrid_VAR_Policy` when a horizon-specific mandatory VAR is allowed; use `SeasonalVAR_CPI_F_NF_S` as the single-model trajectory fallback.

## Verification

- Leakage probe passed: `True`.
- Rolling h=1 and h=12 predictions, metrics, trajectory paths, and charts are saved in the run directory.

## Files

- `predictions.csv`
- `metrics.csv`
- `comparison.csv`
- `trajectory_paths_and_metrics.csv`
- `leakage_checks.csv`
- `rolling_h1_predictions.png`
- `rolling_h12_predictions.png`
