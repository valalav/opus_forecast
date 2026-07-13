# Nemotron task: econometric factor-model alternatives

You are an econometric reviewer. The project needs a defensible factor-family
model for monthly regional inflation forecasting.

Baseline:

- Robust seasonal FAVAR.
- Inputs: components `Food`, `NonFood`, `Services` plus macro/financial
  `USD`, `Ki_i`, `Ruonia`.
- PCA factors: 2.
- Lag: 1.
- Estimator: Huber equations.
- h=1/h=2/h=12 rolling MAE: 0.371 / 0.425 / 0.439.
- h=12 trajectory is stable but slightly too smooth: volatility ratio 0.489.

Task:

Propose factor-family alternatives that are econometrically defensible with
small regional monthly data. Avoid large black-box models. Prioritize
leakage-safe and reportable designs.

Output format:

1. Econometric assessment of baseline.
2. Alternatives ranked by expected value.
3. Diagnostic tests to require.
4. Preferred final model family if results are close.
