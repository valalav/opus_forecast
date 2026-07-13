# Backtest h=2 Summary

**Period:** 2025-07-01 to 2026-06-01 (12 months)
**Horizon:** 2 month(s) ahead
**Generated:** 2026-07-13 16:41:23

## Top 5 Models

13. **Ridge_ProdProxy_Roll24** — MAE 0.294 (0 KPI violations)
9. **Huber_ProdProxy** — MAE 0.336 (1 KPI violations)
12. **Ridge_ProdProxy** — MAE 0.350 (1 KPI violations)
16. **Rolling_Ridge** — MAE 0.423 (3 KPI violations)
24. **EBM** — MAE 0.434 (6 KPI violations)

## KPI Violations (|error| > 0.5)

- Ridge_ProdProxy_Roll24: 0/12 (0.0%)
- Huber_ProdProxy: 1/12 (8.3%)
- Ridge_ProdProxy: 1/12 (8.3%)
- Rolling_Ridge: 3/12 (25.0%)
- EBM: 6/12 (50.0%)
- Ridge_Ext_Roll24: 4/12 (33.3%)
- Subcomp_Multi: 5/12 (41.7%)
- CatBoost: 5/12 (41.7%)
- Ridge_Shock_Roll24: 4/12 (33.3%)
- Ridge_Ext_ProdProxy: 4/12 (33.3%)

## Metrics Table

| Model | MAE | RMSE | KPI Violations | Coverage 50% |
|-------|-----|------|----------------|-------------|
| Ridge_ProdProxy_Roll24 | 0.294 | 0.333 | 0 | 100.0% |
| Huber_ProdProxy | 0.336 | 0.393 | 1 | 80.0% |
| Ridge_ProdProxy | 0.350 | 0.397 | 1 | 80.0% |
| Rolling_Ridge | 0.423 | 0.516 | 3 | 75.0% |
| EBM | 0.434 | 0.511 | 6 | 50.0% |
| Ridge_Ext_Roll24 | 0.439 | 0.556 | 4 | 66.7% |
| Subcomp_Multi | 0.448 | 0.535 | 5 | 58.3% |
| CatBoost | 0.455 | 0.565 | 5 | 58.3% |
| Ridge_Shock_Roll24 | 0.458 | 0.539 | 4 | 66.7% |
| Ridge_Ext_ProdProxy | 0.459 | 0.582 | 4 | 66.7% |
