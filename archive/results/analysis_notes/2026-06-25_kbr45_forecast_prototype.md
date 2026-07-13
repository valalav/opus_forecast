# KBR45 Forecast Prototype

Date: 2026-06-25

Status: experimental prototype, not production.

## Purpose

Create the first runnable 45-component forecast/scenario layer on top of the
canonical KBR45 mapping. This is meant to make July/August, fuel, ЖКУ and other
component scenarios transparent before any production integration.

## Artifacts

- Runner:
  `experiments/kbr_45_forecast_prototype/run_kbr45_forecast_prototype.py`
- Output folder:
  `archive/results/kbr45_forecast_prototype_20260625/`
- Report:
  `archive/results/kbr45_forecast_prototype_20260625/kbr45_forecast_report.md`
- Component forecasts:
  `archive/results/kbr45_forecast_prototype_20260625/kbr45_component_forecast.csv`
- Latest-weight headline forecast:
  `archive/results/kbr45_forecast_prototype_20260625/kbr45_headline_forecast_latest_weights.csv`
- Backtest metrics:
  `archive/results/kbr45_forecast_prototype_20260625/kbr45_backtest_metrics.csv`

## Method

- Last official fact used: May 2026.
- Component history source:
  `data/external/micro_cpi_region_export/region_cpi_long.csv`.
- For each of 45 components, forecast MoM p.p. as a robust blend of
  same-calendar-month median and recent median.
- Aggregate by latest May 2026 regional weights for the current path and by
  canonical weights for backtest compatibility.
- Apply tariff scenario overrides after the baseline:
  July ЖКУ=100.0 and October ЖКУ=110.0.

## Current Output

Latest-weight baseline:

| Month | Index | MoM p.p. |
|---|---:|---:|
| 2026-06 | 100.136 | 0.136 |
| 2026-07 | 100.288 | 0.288 |
| 2026-08 | 99.838 | -0.162 |

Tariff-shift scenario:

| Month | Baseline MoM p.p. | Scenario MoM p.p. | Delta p.p. |
|---|---:|---:|---:|
| 2026-07 | 0.288 | 0.040 | -0.248 |
| 2026-10 | 0.577 | 1.464 | 0.887 |

Diagnostic backtest:

| Horizon | Observations | MAE p.p. | RMSE p.p. | Bias p.p. |
|---:|---:|---:|---:|---:|
| 1 | 125 | 0.417 | 0.681 | -0.103 |
| 2 | 124 | 0.424 | 0.699 | -0.103 |
| 12 | 114 | 0.453 | 0.716 | -0.108 |

## Important Interpretation

- The June 2026 baseline is mechanical and does not include the separate weekly
  gasoline nowcast. It must not replace the weekly Laspeyres June signal.
- The July tariff effect is useful immediately: under this prototype, setting
  July ЖКУ to 100 reduces July headline by about 0.25 p.p.
- The October ЖКУ=110 scenario adds about 0.89 p.p. to October headline under
  latest May 2026 weights.
- The prototype is a transparent scenario layer, not a promoted model.

## Verification

Executed:

```bash
python3 experiments/kbr_45_forecast_prototype/run_kbr45_forecast_prototype.py
python3 -m py_compile experiments/kbr_45_forecast_prototype/run_kbr45_forecast_prototype.py
```

## Next Step

Build a comparison package:

- KBR45 versus `SubcomponentMulti`, `Micro_SM`, Huber and Ensemble;
- h=1/h=2/h=12 errors side by side;
- June 2026 weekly overlay for fuel;
- July/August control-point table with baseline, tariff-shift and gasoline
  downside/upside scenarios.
