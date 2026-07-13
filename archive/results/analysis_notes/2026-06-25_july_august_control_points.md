# July-August 2026 Control Points

Date: 2026-06-25

Status: management decision artifact, not production cache update.

## Purpose

Create a compact July-August control-point fork after reviewing the production
forecast cache, KBR45 tariff adjustment, and the saved gasoline reversion
scenario.

## Artifacts

- Builder:
  `experiments/july_august_control_points/build_july_august_control_points.py`
- Output folder:
  `archive/results/july_august_control_points_20260625/`
- Report:
  `archive/results/july_august_control_points_20260625/july_august_control_points_report.md`
- Scenario table:
  `archive/results/july_august_control_points_20260625/july_august_control_points.csv`
- Assumption bridge:
  `archive/results/july_august_control_points_20260625/assumption_bridge.csv`

## Key Numbers

| Scenario | July 2026 | August 2026 |
|---|---:|---:|
| Production cache Ensemble | 100.649 | 100.340 |
| Policy-adjusted baseline | 100.400 | 100.340 |
| Policy-adjusted + gasoline reversion | 99.973 | 100.340 |
| Policy-adjusted + gasoline/diesel reversion | 99.936 | 100.340 |

## Assumptions

- Production reference comes from `data/precomputed_forecasts.json`.
- July tariff adjustment uses KBR45 scenario delta from
  `archive/results/kbr45_forecast_comparison_20260625/control_point_forecasts_wide.csv`.
- Gasoline reversion drag uses the saved scenario note
  `archive/results/analysis_notes/2026-06-25_july_2026_gasoline_reversion_floor.md`.
- August is not mechanically lowered by the July fuel downside. It should be
  updated only after July weekly data appear.

## Operational Rule

For current July-August discussion, carry three numbers:

1. Production reference: July 100.65, August 100.34.
2. Working policy-adjusted baseline: July 100.40, August 100.34.
3. Downside fuel-risk case: July 99.97, August 100.34.

The working baseline should be `policy_adjusted` unless management explicitly
requires the raw production cache without expert tariff correction.

## Verification

Executed:

```bash
python3 experiments/july_august_control_points/build_july_august_control_points.py
python3 -m py_compile experiments/july_august_control_points/build_july_august_control_points.py
```
