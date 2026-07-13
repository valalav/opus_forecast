# KBR45 Forecast Prototype

Date: 2026-06-25

Status: experimental forecast prototype, not production.

Last official fact in source data: `2026-05-01`

## Inputs

- 45-component map:
  `experiments/kbr_45_component_mapping/kbr_45_component_mapping.csv`
- Official regional long data:
  `data/external/micro_cpi_region_export/region_cpi_long.csv`

## Method

- For each of 45 components, forecast monthly MoM p.p. by a robust blend of
  same-calendar-month median and recent median.
- Aggregate by `latest_region_weight` for the current forecast and by
  `canonical_weight` for compatibility checks.
- Apply transparent scenario overrides only after the baseline forecast.
- Weekly nowcast signals are not yet integrated into this prototype. June 2026
  fuel pressure should therefore be read from the separate weekly Laspeyres
  nowcast until a proper KBR45 weekly overlay is added.

## Current Forecast: Latest Weights

| scenario   | date                |   headline_index |   headline_mom_pp |
|:-----------|:--------------------|-----------------:|------------------:|
| baseline   | 2026-06-01 00:00:00 |          100.136 |             0.136 |
| baseline   | 2026-07-01 00:00:00 |          100.288 |             0.288 |
| baseline   | 2026-08-01 00:00:00 |           99.838 |            -0.162 |

## Tariff Scenario Delta: Latest Weights

| date                |   headline_mom_pp_baseline |   headline_mom_pp_tariff |   delta_pp |
|:--------------------|---------------------------:|-------------------------:|-----------:|
| 2026-06-01 00:00:00 |                      0.136 |                    0.136 |      0.000 |
| 2026-07-01 00:00:00 |                      0.288 |                    0.040 |     -0.248 |
| 2026-08-01 00:00:00 |                     -0.162 |                   -0.162 |      0.000 |
| 2026-09-01 00:00:00 |                      0.768 |                    0.768 |      0.000 |
| 2026-10-01 00:00:00 |                      0.577 |                    1.464 |      0.887 |
| 2026-11-01 00:00:00 |                      0.423 |                    0.423 |      0.000 |
| 2026-12-01 00:00:00 |                      0.257 |                    0.257 |      0.000 |

## Scenario Overrides

| scenario              | date                | external_var   | subcomponent_name                                       |   baseline_mom_pp |   scenario_mom_pp |   latest_region_weight | override_reason                  |
|:----------------------|:--------------------|:---------------|:--------------------------------------------------------|------------------:|------------------:|-----------------------:|:---------------------------------|
| tariff_july100_oct110 | 2026-07-01 00:00:00 | u_gkh          | Жилищные и коммунальные услуги (включая аренду квартир) |             2.793 |             0.000 |                  0.089 | u_gkh July set to index 100.0    |
| tariff_july100_oct110 | 2026-10-01 00:00:00 | u_gkh          | Жилищные и коммунальные услуги (включая аренду квартир) |             0.033 |            10.000 |                  0.089 | u_gkh October set to index 110.0 |

## Backtest Metrics

|   horizon |   observations |   mae_pp |   rmse_pp |   bias_pp |
|----------:|---------------:|---------:|----------:|----------:|
|     1.000 |        125.000 |    0.417 |     0.681 |    -0.103 |
|     2.000 |        124.000 |    0.424 |     0.699 |    -0.103 |
|    12.000 |        114.000 |    0.453 |     0.716 |    -0.108 |

## Interpretation

- This prototype is useful as a transparent scenario layer for fuel, ЖКУ,
  плодоовощи and other tagged components.
- The June baseline is intentionally mechanical and does not supersede the
  existing weekly-nowcast evidence on gasoline.
- It should not replace `SubcomponentMulti`, `Micro_SM`, Huber or Ensemble until
  its backtest is compared against those production candidates.
- July tariff behavior is handled explicitly: July ЖКУ can be set to 100.0 and
  October ЖКУ to 110.0 without changing the baseline component logic.
