# KBR45 Forecast Comparison Package

Date: 2026-06-25

Status: diagnostic comparison, not production.

## Purpose

Answer the immediate "what next?" after building the KBR45 map and forecast
prototype: place KBR45 next to the current production forecast cache and add a
transparent June weekly fuel overlay.

## Artifacts

- Builder:
  `experiments/kbr45_forecast_comparison/build_kbr45_comparison.py`
- Output folder:
  `archive/results/kbr45_forecast_comparison_20260625/`
- Report:
  `archive/results/kbr45_forecast_comparison_20260625/kbr45_forecast_comparison_report.md`
- Control-point table:
  `archive/results/kbr45_forecast_comparison_20260625/control_point_forecasts_wide.csv`
- Weekly fuel overlay summary:
  `archive/results/kbr45_forecast_comparison_20260625/weekly_fuel_overlay_summary.csv`
- Weekly fuel overlay items:
  `archive/results/kbr45_forecast_comparison_20260625/weekly_fuel_overlay_items.csv`

## Key Numbers

| Source / model | 2026-06 | 2026-07 | 2026-08 |
|---|---:|---:|---:|
| Production cache Ensemble | 100.484 | 100.649 | 100.340 |
| Production cache Nowcast | 100.810 | n/a | n/a |
| KBR45 baseline | 100.136 | 100.288 | 99.838 |
| KBR45 baseline + June weekly fuel overlay | 100.682 | 100.288 | 99.838 |
| KBR45 tariff scenario + June weekly fuel overlay | 100.682 | 100.040 | 99.838 |

Weekly fuel overlay:

- latest weekly observation: 2026-06-22;
- weekly matched-basket contribution: 0.610 p.p.;
- weekly fuel contribution: 0.561 p.p.;
- KBR45 mechanical June fuel contribution: 0.015 p.p.;
- fuel overlay delta: 0.546 p.p.

## Interpretation

- Production `Ensemble` remains the current cache baseline.
- Cache `Nowcast` is higher for June because short-term weekly pressure is
  already visible there.
- Mechanical KBR45 is too calm for June without the weekly fuel overlay.
- Tariff shift is the clean KBR45 policy insight: July ЖКУ=100 lowers July by
  about 0.25 p.p.; October ЖКУ=110 adds about 0.89 p.p. in October in the
  KBR45 prototype.

## Next Step

Do not promote KBR45 yet. Build a challenger backtest against
`SubcomponentMulti`, `Micro_SM`, Huber and Ensemble, and then decide whether the
KBR45 layer should remain diagnostic or become a registered model.
