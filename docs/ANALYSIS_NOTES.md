# Scenario Analysis Notes

Use this page to find compact saved forecast analyses that should carry across
agent sessions.

## Stable Location

All short scenario notes and forecast-memory memos should live in:

```text
archive/results/analysis_notes/
```

The registry is:

```text
archive/results/analysis_notes/analysis_index.csv
```

Each saved note should have:

- a Markdown file with the assumptions, source files, calculation, and forecast
  implication;
- one row in `analysis_index.csv`;
- links to any related CSV, JSON, XLSX, DOCX, or package artifacts.

## When To Check This Registry

Check `analysis_index.csv` before continuing discussions about:

- control-point forecasts for the next months;
- gasoline, fuel, or other item-level downside/upside scenarios;
- tariff timing and regulated-price assumptions;
- microcomponent overrides;
- weekly nowcast interpretation;
- any previously discussed expert adjustment that may not be in the main model
  cache.

## Current Notes

| Date | Topic | Note |
|---|---|---|
| 2026-09-01 | September live-nowcast after fixing August at 100.0 | [2026-09-01_september_live_nowcast_protocol.md](../archive/results/analysis_notes/2026-09-01_september_live_nowcast_protocol.md) |
| 2026-09-01 | PR3 revision after preliminary August near zero | [2026-09-01_august_preliminary_pr3_revision.md](../archive/results/analysis_notes/2026-09-01_august_preliminary_pr3_revision.md) |
| 2026-08-27 | August 2026 nowcast through 24 August | [2026-08-27_august_nowcast_week4.md](../archive/results/analysis_notes/2026-08-27_august_nowcast_week4.md) |
| 2026-08-21 | August 2026 nowcast through 17 August | [2026-08-21_august_nowcast_week3.md](../archive/results/analysis_notes/2026-08-21_august_nowcast_week3.md) |
| 2026-06-25 | July 2026 gasoline reversion lower-bound scenario | [2026-06-25_july_2026_gasoline_reversion_floor.md](../archive/results/analysis_notes/2026-06-25_july_2026_gasoline_reversion_floor.md) |
| 2026-06-25 | Review of added Volgograd ML / variable-selection / Excel-panel experiments | [2026-06-25_experiments_volgograd_methodology_review.md](../archive/results/analysis_notes/2026-06-25_experiments_volgograd_methodology_review.md) |
| 2026-06-25 | KBR pseudo-OOS variable-selection pilot on current monthly data | [variable_selection_pilot_report.md](../archive/results/variable_selection_pilot_20260625/variable_selection_pilot_report.md) |
| 2026-06-25 | Roadmap of external model ideas for SIRENA-KBR | [EXTERNAL_MODEL_ROADMAP.md](EXTERNAL_MODEL_ROADMAP.md) |
| 2026-06-25 | External code repository unpacking and study inventory | [2026-06-25_code_repository_inventory.md](../archive/results/analysis_notes/2026-06-25_code_repository_inventory.md) |
| 2026-06-25 | Direct code review of external repository model candidates | [2026-06-25_external_code_deep_dive.md](../archive/results/analysis_notes/2026-06-25_external_code_deep_dive.md) |
| 2026-06-25 | Weekly Laspeyres nowcast prototype for June 2026 | [2026-06-25_weekly_laspeyres_nowcast_prototype.md](../archive/results/analysis_notes/2026-06-25_weekly_laspeyres_nowcast_prototype.md) |
| 2026-06-25 | External code integration plan for SIRENA-KBR | [EXTERNAL_CODE_INTEGRATION_PLAN.md](EXTERNAL_CODE_INTEGRATION_PLAN.md) |
| 2026-06-25 | KBR 45-component mapping for microcomponent forecast/scenarios | [2026-06-25_kbr_45_component_mapping.md](../archive/results/analysis_notes/2026-06-25_kbr_45_component_mapping.md) |
| 2026-06-25 | KBR45 forecast prototype and tariff-shift scenario layer | [2026-06-25_kbr45_forecast_prototype.md](../archive/results/analysis_notes/2026-06-25_kbr45_forecast_prototype.md) |
| 2026-06-25 | KBR45 comparison package for June-August control points | [2026-06-25_kbr45_forecast_comparison.md](../archive/results/analysis_notes/2026-06-25_kbr45_forecast_comparison.md) |
| 2026-06-25 | July-August 2026 management control points | [2026-06-25_july_august_control_points.md](../archive/results/analysis_notes/2026-06-25_july_august_control_points.md) |

## Maintenance Rule

Do not create a new one-off folder for a short scenario note without adding it
to `archive/results/analysis_notes/analysis_index.csv`. If a full production
package is created elsewhere under `archive/results/`, add a registry row that
points to it.
