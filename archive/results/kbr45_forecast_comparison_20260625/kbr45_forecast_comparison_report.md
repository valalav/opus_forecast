# KBR45 Forecast Comparison Package

Date: 2026-06-25

Status: diagnostic comparison, not a production forecast update.

## What This Answers

This package places the new KBR45 scenario layer next to the existing production
forecast cache and the June weekly gasoline signal. It is the next step after
building the 45-component map and prototype.

## Control Points

| source                      | model                                           | status                              |   2026-06 |   2026-07 |   2026-08 |
|:----------------------------|:------------------------------------------------|:------------------------------------|----------:|----------:|----------:|
| kbr45_plus_weekly_laspeyres | KBR45_baseline_weekly_fuel_overlay              | experimental_overlay_not_production |   100.682 |   100.288 |    99.838 |
| kbr45_plus_weekly_laspeyres | KBR45_tariff_july100_oct110_weekly_fuel_overlay | experimental_overlay_not_production |   100.682 |   100.040 |    99.838 |
| kbr45_prototype             | KBR45_baseline                                  | experimental_not_production         |   100.136 |   100.288 |    99.838 |
| kbr45_prototype             | KBR45_tariff_july100_oct110                     | experimental_not_production         |   100.136 |   100.040 |    99.838 |
| production_cache            | Ensemble                                        | production_or_existing_diagnostic   |   100.484 |   100.649 |   100.340 |
| production_cache            | Micro_SM                                        | production_or_existing_diagnostic   |    99.758 |    99.845 |    99.597 |
| production_cache            | Nowcast                                         | production_or_existing_diagnostic   |   100.810 |   nan     |   nan     |

## Weekly Fuel Overlay

- Latest weekly observation used: `2026-06-22`
- Weekly matched-basket contribution: `0.610` p.p.
- Weekly fuel contribution: `0.561` p.p.
- KBR45 mechanical June fuel contribution: `0.015` p.p.
- June fuel overlay delta: `0.546` p.p.

## Reading The Numbers

- Production `Ensemble` remains the current cache baseline.
- `Nowcast` is available only for June in the cache and is higher because it
  already reflects short-term weekly pressure.
- Mechanical KBR45 baseline is too calm for June unless the weekly fuel overlay
  is added.
- The tariff-shift result is the clean policy insight: setting July ЖКУ to 100
  lowers July KBR45 by about 0.25 p.p.; moving the 110 indexation to October
  pushes the October point by about 0.89 p.p. in the prototype.

## Next Operational Step

Build a proper challenger backtest against `SubcomponentMulti`, `Micro_SM`,
Huber and Ensemble, then decide whether KBR45 should remain a diagnostic layer
or be registered as a model.
