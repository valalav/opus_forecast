# Backtest h=1 Summary

**Period:** 2025-08-01 to 2026-07-01 (12 months)
**Horizon:** 1 month(s) ahead
**Generated:** 2026-09-08 15:54:51

## Top 5 Models

13. **Ridge_ProdProxy_Roll24** — MAE 0.261 (1 KPI violations)
12. **Ridge_ProdProxy** — MAE 0.338 (1 KPI violations)
8. **Huber_Roll24** — MAE 0.370 (3 KPI violations)
16. **Rolling_Ridge** — MAE 0.373 (3 KPI violations)
11. **Ridge_Shock_Roll24** — MAE 0.382 (4 KPI violations)

## KPI Violations (|error| > 0.5)

- Ridge_ProdProxy_Roll24: 1/12 (8.3%)
- Ridge_ProdProxy: 1/12 (8.3%)
- Huber_Roll24: 3/12 (25.0%)
- Rolling_Ridge: 3/12 (25.0%)
- Ridge_Shock_Roll24: 4/12 (33.3%)
- Ridge_Ext_ProdProxy: 2/12 (16.7%)
- Bayes_Ridge: 4/12 (33.3%)
- Ridge_Ext_Roll24: 3/12 (25.0%)
- NGBoost_Shock: 5/12 (41.7%)
- ElasticNet: 4/12 (33.3%)

## Metrics Table

| Model | MAE | RMSE | KPI Violations | Coverage 50% |
|-------|-----|------|----------------|-------------|
| Ridge_ProdProxy_Roll24 | 0.261 | 0.320 | 1 | 66.7% |
| Ridge_ProdProxy | 0.338 | 0.399 | 1 | 66.7% |
| Huber_Roll24 | 0.370 | 0.481 | 3 | 75.0% |
| Rolling_Ridge | 0.373 | 0.468 | 3 | 75.0% |
| Ridge_Shock_Roll24 | 0.382 | 0.481 | 4 | 66.7% |
| Ridge_Ext_ProdProxy | 0.382 | 0.454 | 2 | 33.3% |
| Bayes_Ridge | 0.386 | 0.485 | 4 | 66.7% |
| Ridge_Ext_Roll24 | 0.386 | 0.484 | 3 | 75.0% |
| NGBoost_Shock | 0.397 | 0.509 | 5 | 58.3% |
| ElasticNet | 0.397 | 0.496 | 4 | 66.7% |
