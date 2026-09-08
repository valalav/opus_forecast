# Backtest h=2 Summary

**Period:** 2025-08-01 to 2026-07-01 (12 months)
**Horizon:** 2 month(s) ahead
**Generated:** 2026-09-08 15:55:29

## Top 5 Models

13. **Ridge_ProdProxy_Roll24** — MAE 0.289 (0 KPI violations)
3. **Ridge_Ext_ProdProxy** — MAE 0.347 (1 KPI violations)
12. **Ridge_ProdProxy** — MAE 0.354 (1 KPI violations)
9. **Huber_ProdProxy** — MAE 0.373 (1 KPI violations)
16. **Rolling_Ridge** — MAE 0.457 (4 KPI violations)

## KPI Violations (|error| > 0.5)

- Ridge_ProdProxy_Roll24: 0/12 (0.0%)
- Ridge_Ext_ProdProxy: 1/12 (8.3%)
- Ridge_ProdProxy: 1/12 (8.3%)
- Huber_ProdProxy: 1/12 (8.3%)
- Rolling_Ridge: 4/12 (33.3%)
- Ridge_Ext_Roll24: 5/12 (41.7%)
- Huber_Roll24: 5/12 (41.7%)
- EBM: 6/12 (50.0%)
- Ridge_Shock_Roll24: 5/12 (41.7%)
- Ridge_Ext: 6/12 (50.0%)

## Metrics Table

| Model | MAE | RMSE | KPI Violations | Coverage 50% |
|-------|-----|------|----------------|-------------|
| Ridge_ProdProxy_Roll24 | 0.289 | 0.338 | 0 | 100.0% |
| Ridge_Ext_ProdProxy | 0.347 | 0.402 | 1 | 75.0% |
| Ridge_ProdProxy | 0.354 | 0.411 | 1 | 75.0% |
| Huber_ProdProxy | 0.373 | 0.430 | 1 | 75.0% |
| Rolling_Ridge | 0.457 | 0.555 | 4 | 66.7% |
| Ridge_Ext_Roll24 | 0.471 | 0.582 | 5 | 58.3% |
| Huber_Roll24 | 0.499 | 0.612 | 5 | 58.3% |
| EBM | 0.499 | 0.568 | 6 | 50.0% |
| Ridge_Shock_Roll24 | 0.500 | 0.580 | 5 | 58.3% |
| Ridge_Ext | 0.505 | 0.595 | 6 | 50.0% |
