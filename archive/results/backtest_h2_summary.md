# Backtest h=2 Summary

**Period:** 2025-08-01 to 2026-07-01 (12 months)
**Horizon:** 2 month(s) ahead
**Generated:** 2026-08-17 11:42:23

## Top 5 Models

13. **Ridge_ProdProxy_Roll24** — MAE 0.289 (0 KPI violations)
12. **Ridge_ProdProxy** — MAE 0.354 (1 KPI violations)
9. **Huber_ProdProxy** — MAE 0.374 (1 KPI violations)
16. **Rolling_Ridge** — MAE 0.455 (4 KPI violations)
24. **EBM** — MAE 0.460 (6 KPI violations)

## KPI Violations (|error| > 0.5)

- Ridge_ProdProxy_Roll24: 0/12 (0.0%)
- Ridge_ProdProxy: 1/12 (8.3%)
- Huber_ProdProxy: 1/12 (8.3%)
- Rolling_Ridge: 4/12 (33.3%)
- EBM: 6/12 (50.0%)
- Subcomp_Multi: 5/12 (41.7%)
- Prophet: 5/12 (41.7%)
- Huber_Roll24: 5/12 (41.7%)
- Ridge_Ext_Roll24: 5/12 (41.7%)
- Ridge_Ext_ProdProxy: 4/12 (33.3%)

## Metrics Table

| Model | MAE | RMSE | KPI Violations | Coverage 50% |
|-------|-----|------|----------------|-------------|
| Ridge_ProdProxy_Roll24 | 0.289 | 0.338 | 0 | 100.0% |
| Ridge_ProdProxy | 0.354 | 0.411 | 1 | 75.0% |
| Huber_ProdProxy | 0.374 | 0.430 | 1 | 75.0% |
| Rolling_Ridge | 0.455 | 0.550 | 4 | 66.7% |
| EBM | 0.460 | 0.522 | 6 | 50.0% |
| Subcomp_Multi | 0.464 | 0.544 | 5 | 58.3% |
| Prophet | 0.466 | 0.515 | 5 | 58.3% |
| Huber_Roll24 | 0.490 | 0.592 | 5 | 58.3% |
| Ridge_Ext_Roll24 | 0.492 | 0.604 | 5 | 58.3% |
| Ridge_Ext_ProdProxy | 0.492 | 0.599 | 4 | 66.7% |
