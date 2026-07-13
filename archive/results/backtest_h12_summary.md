# Backtest h=12 Summary

**Period:** 2025-04-01 to 2026-03-01 (12 months)
**Horizon:** 12 month(s) ahead
**Generated:** 2026-05-13 16:55:20

## Top 5 Models

13. **Ridge_ProdProxy_Roll24** — MAE 0.304 (2 KPI violations)
12. **Ridge_ProdProxy** — MAE 0.326 (3 KPI violations)
23. **Subcomp_Multi** — MAE 0.331 (4 KPI violations)
18. **NGBoost_Shock** — MAE 0.344 (4 KPI violations)
1. **Ridge** — MAE 0.346 (4 KPI violations)

## KPI Violations (|error| > 0.5)

- Ridge_ProdProxy_Roll24: 2/12 (16.7%)
- Ridge_ProdProxy: 3/12 (25.0%)
- Subcomp_Multi: 4/12 (33.3%)
- NGBoost_Shock: 4/12 (33.3%)
- Ridge: 4/12 (33.3%)
- EBM: 5/12 (41.7%)
- Rolling_Ridge: 3/12 (25.0%)
- Huber_ProdProxy: 4/12 (33.3%)
- Ridge_Shock_Roll24: 3/12 (25.0%)
- Ridge_AsymERPT: 3/12 (25.0%)

## Metrics Table

| Model | MAE | RMSE | KPI Violations | Coverage 50% |
|-------|-----|------|----------------|-------------|
| Ridge_ProdProxy_Roll24 | 0.304 | 0.367 | 2 | 83.3% |
| Ridge_ProdProxy | 0.326 | 0.381 | 3 | 75.0% |
| Subcomp_Multi | 0.331 | 0.421 | 4 | 66.7% |
| NGBoost_Shock | 0.344 | 0.397 | 4 | 66.7% |
| Ridge | 0.346 | 0.394 | 4 | 66.7% |
| EBM | 0.351 | 0.405 | 5 | 58.3% |
| Rolling_Ridge | 0.354 | 0.408 | 3 | 75.0% |
| Huber_ProdProxy | 0.354 | 0.408 | 4 | 66.7% |
| Ridge_Shock_Roll24 | 0.362 | 0.418 | 3 | 75.0% |
| Ridge_AsymERPT | 0.372 | 0.414 | 3 | 75.0% |
