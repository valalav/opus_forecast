# Factor model agent review synthesis

Дата: 2026-06-09

## Запущенные агенты

| Agent | Channel/model | Status | Notes |
| --- | --- | --- | --- |
| Qwen local | `http://127.0.0.1:8082`, `qwen3.6-27b-mtp-q5` | completed | Fast enough on compact context; useful for idea generation. |
| MiniMax | `minimax/MiniMax-M3` via direct `pi` | completed | Useful matrix, but contains hallucinated commands/scripts and some weak variable suggestions. |
| Gemini | `omniroute/gemini-cli/gemini-3.1-pro-preview` via direct `pi` | completed | Strong methodology and leakage review. |
| Nemotron | `omniroute/nvidia/nemotron-3-super-120b-a12b` via direct `pi` | completed | Strong econometric critique and final-model preference. |

All calls were run without write tools. Pi `/goals-set` was not used because the
previous smoke test showed auto-continue/token-control issues.

## Common recommendations

The agents broadly agree on four points:

1. The current robust seasonal FAVAR is a defensible baseline, but it is too
   smooth at h=12 and should be benchmarked against simpler seasonal baselines.
2. The most valuable next variant is not a larger black-box model, but a more
   interpretable factor design:
   - component factor from `Food`, `NonFood`, `Services`;
   - macro/financial factor from `USD`, `Ki_i`, `Ruonia` and optionally other
     already available macro columns if leakage-safe;
   - direct bridge/Huber/Ridge equation on CPI rather than only VAR propagation.
3. Evaluation must be strengthened:
   - relative MAE versus seasonal naive or seasonal AR;
   - residual autocorrelation diagnostics;
   - h=12 volatility ratio;
   - directional/turning-point accuracy;
   - Diebold-Mariano style comparison only if implemented carefully.
4. Selection must be fixed before running new variants. Recommended weighted
   score:
   - 50% h=1 MAE relative to current factor baseline;
   - 30% h=2 MAE relative to current factor baseline;
   - 20% h=12 MAE relative to current factor baseline;
   - disqualify any variant with explosive h=12 paths or clear leakage risk.

## Useful candidate variants

Ranked for expected value and implementation fit:

1. **Transparent block factor bridge**
   - Predefined component factor and macro/financial factor.
   - Forecast CPI with lagged CPI, lagged block factors, seasonal terms, and
     Huber/Ridge/quantile loss.
   - Best reportability if metrics are close.

2. **Factor-Augmented Huber/Ridge direct equations**
   - Keep train-only PCA factors, but use direct h=1/h=2/h=12 equations instead
     of forcing all horizons through the same VAR dynamics.
   - Likely helps h=12 smoothness and reduces VAR parameter risk.

3. **Sparse/regularized factor extraction**
   - Sparse PCA or constrained group factors.
   - Goal: improve interpretability and factor stability.
   - Must tune sparsity inside the rolling train window only.

4. **Lag/window robustness screen**
   - Test lag 1/2/3 and expanding versus rolling windows.
   - Use as diagnostic first, not as an unlimited grid search.

5. **Quantile/median factor bridge**
   - Qwen suggested Quantile FAVAR to address distributional misspecification.
   - In this project, the immediate value is robust median/quantile direct
     equations, not interval coverage, because current "coverage <= 0.5" is a
     KPI error threshold rather than a prediction-interval metric.

## Risks and rejected/low-priority ideas

- Full Markov-switching FAVAR and Bayesian/time-varying DFM are econometrically
  interesting but too heavy for the next compact implementation slice.
- MiniMax suggested verification commands and scripts that do not exist in this
  repository (`scripts/check_no_lookahead.py`, `scripts/dm_test.py`, several
  `uv run` examples). Treat them as conceptual gates, not executable commands.
- Qwen interpreted KPI coverage as prediction-interval coverage. The idea of
  quantile modeling is still useful, but the diagnosis needs correction.
- Adding new data sources should be deferred. The next iteration should use
  existing `data/inflation_data.csv` columns and current loaders.

## Recommended next implementation slice

Implement one compact challenger in the existing factor research framework:

**BlockFactorBridge**

- Inputs:
  - target: `CPI`;
  - component block: `Food`, `NonFood`, `Services`;
  - macro block: `USD`, `Ki_i`, `Ruonia`;
  - optional lags: CPI lag 1, block factors lag 1-3;
  - train-only month seasonality.
- Factor construction:
  - component factor: train-only PCA(1) or weighted average of components;
  - macro factor: train-only PCA(1);
  - factors recomputed inside each rolling train window.
- Estimators:
  - Huber;
  - Ridge;
  - optional QuantileRegressor median if dependency/performance is acceptable.
- Evaluation:
  - h=1/h=2/h=12 rolling MAE/RMSE/KPI coverage;
  - h=12 trajectory diagnostics;
  - relative MAE versus current `RobustFAVAR_lean_f2_l1_seasonal`;
  - relative MAE versus a simple seasonal naive baseline if already available
    or easy to add without disturbing production code.

Promotion rule:

- Promote only if h=1 MAE does not regress by more than 2% versus current
  factor baseline and weighted relative score improves by at least 1%;
- otherwise keep current `factor_policy` as the mandatory factor model and
  document BlockFactorBridge as a tested challenger.
