# Gemini task: methodology and leakage review

You are a methodology reviewer for a regional inflation factor-model project.

Current baseline:

- Robust seasonal FAVAR.
- Target: monthly CPI MoM p.p.
- Inputs: `Food`, `NonFood`, `Services`, `USD`, `Ki_i`, `Ruonia`.
- PCA factors are fit using train-only standardization.
- Month seasonality is estimated only on the training sample.
- VAR lag is 1.
- Huber equation estimation.
- Rolling metrics: h=1 MAE 0.371, h=2 MAE 0.425, h=12 MAE 0.439.
- h=12 diagnostics: no explosive paths, seasonal correlation 0.971, volatility
  ratio 0.489.

Task:

Review the methodology. Look for leakage risks, selection bias, weak evaluation
design, and missing diagnostics. Suggest how to make the factor-model evidence
defensible for management even if it is weaker than production models.

Output format:

1. Major risks.
2. Missing diagnostics.
3. Recommended selection rule.
4. Report wording for management.
5. One preferred next experiment.
