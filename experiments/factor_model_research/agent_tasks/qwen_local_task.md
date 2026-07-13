# Qwen local task: compact factor-model critique

You are reviewing a regional inflation factor-model baseline.

Context:

- Target: monthly CPI MoM percentage points for KBR.
- Current baseline: robust seasonal FAVAR.
- Information set: `Food`, `NonFood`, `Services`, `USD`, `Ki_i`, `Ruonia`.
- Factor extraction: train-only PCA standardization.
- Factors: 2.
- Lag: 1.
- Seasonality: train-only month-of-year residualization and reconstruction.
- Estimator: Huber equation estimation.
- Rolling metrics:
  - h=1 MAE 0.371, RMSE 0.619, coverage <= 0.5 is 74.0%.
  - h=2 MAE 0.425, RMSE 0.710, coverage <= 0.5 is 73.0%.
  - h=12 MAE 0.439, RMSE 0.718, coverage <= 0.5 is 70.0%.
- h=12 trajectory diagnostics:
  - mean path MAE 0.418;
  - volatility ratio 0.489;
  - flatness share 0.084;
  - seasonal correlation 0.971;
  - explosive path rate 0%.

Task:

Give a compact critique for a local Qwen agent. Do not write code. Suggest 3-5
high-value factor-model variants that could be tested next, with one sentence
for why each might help and one leakage risk to avoid.

Output format:

1. Short verdict.
2. Candidate variants.
3. Risks.
4. Best next experiment.
