# Backtest h=1 Summary

**Period:** 2025-08-01 to 2026-07-01 (12 months)
**Horizon:** 1 month(s) ahead
**Generated:** 2026-08-17 10:31:59

## Top 5 Models

13. **Ridge_ProdProxy_Roll24** — MAE 0.261 (1 KPI violations)
12. **Ridge_ProdProxy** — MAE 0.338 (1 KPI violations)
11. **Ridge_Shock_Roll24** — MAE 0.418 (5 KPI violations)
8. **Huber_Roll24** — MAE 0.422 (4 KPI violations)
9. **Huber_ProdProxy** — MAE 0.424 (2 KPI violations)

## KPI Violations (|error| > 0.5)

- Ridge_ProdProxy_Roll24: 1/12 (8.3%)
- Ridge_ProdProxy: 1/12 (8.3%)
- Ridge_Shock_Roll24: 5/12 (41.7%)
- Huber_Roll24: 4/12 (33.3%)
- Huber_ProdProxy: 2/12 (16.7%)
- Ridge_AsymERPT: 4/12 (33.3%)
- Prophet: 3/12 (25.0%)
- EBM: 6/12 (50.0%)
- Rolling_Ridge: 4/12 (33.3%)
- Ridge_Shock: 4/12 (33.3%)

## Metrics Table

| Model | MAE | RMSE | KPI Violations | Coverage 50% |
|-------|-----|------|----------------|-------------|
| Ridge_ProdProxy_Roll24 | 0.261 | 0.320 | 1 | 66.7% |
| Ridge_ProdProxy | 0.338 | 0.399 | 1 | 66.7% |
| Ridge_Shock_Roll24 | 0.418 | 0.510 | 5 | 58.3% |
| Huber_Roll24 | 0.422 | 0.519 | 4 | 66.7% |
| Huber_ProdProxy | 0.424 | 0.514 | 2 | 33.3% |
| Ridge_AsymERPT | 0.429 | 0.509 | 4 | 66.7% |
| Prophet | 0.430 | 0.495 | 3 | 75.0% |
| EBM | 0.435 | 0.516 | 6 | 50.0% |
| Rolling_Ridge | 0.435 | 0.522 | 4 | 66.7% |
| Ridge_Shock | 0.441 | 0.517 | 4 | 66.7% |
