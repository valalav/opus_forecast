# July–August 2026 Fuel Trajectory

Date: 2026-07-13

## Decision

Central control points for KBR CPI:

- July 2026: **101.40** (+1.40% MoM); KPI tolerance is ±0.5 p.p., not a confidence interval.
- August 2026: **100.50** (+0.50% MoM)

The July point rounds the fresh weekly bridge (`+1.41%`). The reported 120 rub./litre market price remains a conditional fuel-upside signal, not the central input.

## Fuel Scenarios

| Scenario | July | August | Rule |
|---|---:|---:|---|
| Fast correction to the 29 June official level | 100.85 | 99.95 | The direct fuel increment to June is removed. |
| Expert central (KPI) | 101.40 | 100.50 | Rounded fresh weekly bridge; working KPI band is 100.90–101.90. |
| Fuel upside: partial 120 | 101.70 | 100.50 | Mechanical ceiling with embedded-fuel double-count risk; not central. |
| Stress: 120 rub./litre | 102.30 | 100.75 | Direct July fuel contribution is +1.192 p.p.; August contains no repeat direct contribution, only a separately marked expectations/logistics risk. |

## Evidence

- `data/inflation_data.csv`: May `100.09`, June official `101.06`.
- June direct fuel contribution from AI-92/95/98 and diesel, using `data/micro_sprav.csv` weights and fresh ACCDB indices: `+0.709` p.p.
- Production Ensemble after the June fact: July `+1.14%`; auxiliary weekly bridge Nowcast: `+1.41%`.
- Item-weighted weekly Laspeyres diagnostic for 6 July: component-scaled `+1.03%`, headline partial `+0.31%`, matched basket weight `34.687%`.

## Artifacts

- Full report: `archive/results/july_2026_fuel_trajectory_20260713/july_august_fuel_trajectory_report.md`
- Scenario calculation: `archive/results/july_2026_fuel_trajectory_20260713/fuel_scenario_contributions.csv`
- Central path: `archive/results/july_2026_fuel_trajectory_20260713/central_policy_trajectory.csv`
- Weighted weekly diagnostic: `archive/results/july_2026_fuel_trajectory_20260713/weekly_laspeyres/`
- OPR-form backup: `archive/results/july_2026_fuel_trajectory_20260713/06_2026_02_Прогноз.before_kpi_revision.xlsx`

## Limits

Weekly observations remain a nowcast signal, not an official monthly CPI fact. The inflation-expectations/logistics overlay is a transparent scenario because no separately identified cutoff-safe coefficient has been validated.
