# Backtest h=2 Summary

**Period:** 2025-04-01 to 2026-03-01 (12 months)
**Horizon:** 2 month(s) ahead
**Generated:** 2026-05-13 16:55:07

## Top 5 Models

13. **Ridge_ProdProxy_Roll24** — MAE 0.279 (0 KPI violations)
4. **Ridge_Ext_Roll24** — MAE 0.289 (2 KPI violations)
16. **Rolling_Ridge** — MAE 0.294 (1 KPI violations)
12. **Ridge_ProdProxy** — MAE 0.312 (2 KPI violations)
3. **Ridge_Ext_ProdProxy** — MAE 0.328 (3 KPI violations)

## KPI Violations (|error| > 0.5)

- Ridge_ProdProxy_Roll24: 0/12 (0.0%)
- Ridge_Ext_Roll24: 2/12 (16.7%)
- Rolling_Ridge: 1/12 (8.3%)
- Ridge_ProdProxy: 2/12 (16.7%)
- Ridge_Ext_ProdProxy: 3/12 (25.0%)
- Ridge_Ext: 4/12 (33.3%)
- Ridge_Shock_Roll24: 2/12 (16.7%)
- Huber: 4/12 (33.3%)
- Huber_Roll24: 2/12 (16.7%)
- Huber_ProdProxy: 2/12 (16.7%)

## Metrics Table

| Model | MAE | RMSE | KPI Violations | Coverage 50% |
|-------|-----|------|----------------|-------------|
| Ridge_ProdProxy_Roll24 | 0.279 | 0.318 | 0 | 100.0% |
| Ridge_Ext_Roll24 | 0.289 | 0.352 | 2 | 83.3% |
| Rolling_Ridge | 0.294 | 0.342 | 1 | 91.7% |
| Ridge_ProdProxy | 0.312 | 0.378 | 2 | 75.0% |
| Ridge_Ext_ProdProxy | 0.328 | 0.384 | 3 | 75.0% |
| Ridge_Ext | 0.334 | 0.394 | 4 | 66.7% |
| Ridge_Shock_Roll24 | 0.340 | 0.397 | 2 | 83.3% |
| Huber | 0.343 | 0.398 | 4 | 66.7% |
| Huber_Roll24 | 0.345 | 0.408 | 2 | 83.3% |
| Huber_ProdProxy | 0.347 | 0.390 | 2 | 75.0% |
