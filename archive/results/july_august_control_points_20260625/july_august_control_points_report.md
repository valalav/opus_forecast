# July-August 2026 Control Points

Date: 2026-06-25

Status: management decision artifact, not production cache update.

## Recommendation

Use **policy-adjusted baseline** as the working July-August control point until
the official June fact and additional July weekly fuel data arrive.

- July working point: `100.400`.
- August working point: `100.340`.
- July downside if gasoline reverts: `99.973`.
- Hard July downside if gasoline and diesel revert: `99.936`.

## Scenario Table

| scenario_id                           | scenario_name                               |   july_index |   august_index | decision_use                                                  |
|:--------------------------------------|:--------------------------------------------|-------------:|---------------:|:--------------------------------------------------------------|
| production_cache                      | Production cache Ensemble                   |      100.649 |        100.340 | Reference only unless no expert override is allowed           |
| policy_adjusted                       | Policy-adjusted baseline                    |      100.400 |        100.340 | Recommended working baseline before final June fact           |
| policy_plus_gasoline_reversion        | Policy-adjusted + gasoline reversion        |       99.973 |        100.340 | Lower-risk bound if July gasoline returns to early-June level |
| policy_plus_gasoline_diesel_reversion | Policy-adjusted + gasoline/diesel reversion |       99.936 |        100.340 | Hard downside sensitivity, not central                        |

## Assumption Bridge

| item                             |   value_pp | source                                                                              | comment                                                  |
|:---------------------------------|-----------:|:------------------------------------------------------------------------------------|:---------------------------------------------------------|
| Production Ensemble July         |     0.6487 | data/precomputed_forecasts.json                                                     | Current production cache before expert policy correction |
| Production Ensemble August       |     0.3398 | data/precomputed_forecasts.json                                                     | Current production cache                                 |
| KBR45 July tariff delta          |    -0.2485 | archive/results/kbr45_forecast_comparison_20260625/control_point_forecasts_wide.csv | KBR45 baseline minus KBR45 July ЖКУ=100 scenario         |
| Gasoline reversion drag          |    -0.4275 | archive/results/analysis_notes/2026-06-25_july_2026_gasoline_reversion_floor.md     | Direct July drag if gasoline returns to 2026-06-01 level |
| Gasoline + diesel reversion drag |    -0.4637 | archive/results/analysis_notes/2026-06-25_july_2026_gasoline_reversion_floor.md     | Hard downside sensitivity                                |

## Interpretation

- The production Ensemble remains the reference cache path: July 100.649,
  August 100.340.
- Because July 2026 regulated-tariff indexation is assumed absent, the raw
  Ensemble July point is too high for the working policy path.
- Applying the KBR45 tariff delta lowers July by about 0.248 p.p., giving a
  working July point around 100.400.
- If gasoline returns to early-June levels in July, the direct drag can push the
  July point to about 99.97. Diesel reversion makes the hard downside about
  99.94.
- August should not mechanically inherit the July fuel downside. Keep August at
  the policy-adjusted baseline until new weekly July data appear.

## Operational Rule

For current discussion, carry three numbers:

1. Production reference: July 100.65, August 100.34.
2. Working policy-adjusted baseline: July 100.40, August 100.34.
3. Downside fuel-risk case: July 99.97, August 100.34.
