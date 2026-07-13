# KBR 45-Component Mapping

Date: 2026-06-25

Purpose: create a stable bridge between the external Khabarovsk/Omsk
45-component model structure and SIRENA-KBR data, so that microcomponent
forecasting, fuel/tariff scenarios, and July/August control-point analysis use
one shared component map.

## What Was Built

- Builder:
  `experiments/kbr_45_component_mapping/build_kbr_45_component_mapping.py`
- Component map:
  `experiments/kbr_45_component_mapping/kbr_45_component_mapping.csv`
- Micro item map:
  `experiments/kbr_45_component_mapping/kbr_45_micro_item_mapping.csv`
- Parent summary:
  `experiments/kbr_45_component_mapping/kbr_45_component_mapping_summary.csv`
- Diagnostic report:
  `experiments/kbr_45_component_mapping/kbr_45_component_mapping_report.md`

## Data Sources

- Existing 45-component SIRENA structure:
  `data/raw/subcomp_sprav.csv`
- KBR item-level weights:
  `data/micro_sprav.csv`
- Current official long regional export through May 2026:
  `data/external/micro_cpi_region_export/region_cpi_long.csv`
- External code references:
  Khabarovsk `khab_mod.prg` and Omsk `arima_omsk.prg` under
  `experiments/code_repository_20260625/nested_extracted/`

## Main Result

- External 45 variables mapped: 45 / 45.
- Canonical 45-component weight sum: 1.00000.
- Latest May 2026 regional 45-component weight sum: 1.00000.
- `micro_sprav` item weight assigned to the 45 layer: 0.98565.
- Unmapped `micro_sprav` weight: 0.00000.

Parent-component reconciliation:

| Parent | Components | Canonical weight | Latest May 2026 weight | Micro-sprav assigned weight |
|---|---:|---:|---:|---:|
| Nonprod | 20 | 0.36534 | 0.36378 | 0.36119 |
| Prod | 15 | 0.39481 | 0.39859 | 0.39216 |
| Serv | 10 | 0.23985 | 0.23763 | 0.23230 |

## Scenario Handles

The map includes scenario tags for:

- fuel / gasoline: `n_topl`, code 42;
- ЖКУ / regulated tariff shift: `u_gkh`, code 14;
- плодоовощи: `p_ovosh`, code 33;
- eggs: `p_egg`, code 52;
- sugar: `p_sugar`, code 36;
- education: `u_obr`, code 44;
- transport: `u_transp`, code 47;
- telecom: `u_sv`, code 48;
- tourism: `u_z_tour`, code 67.

## Caveats

- Service item allocation is inferred by item-name rules because
  `data/micro_sprav.csv` does not contain service subcomponent labels.
- The component map is ready for scenario diagnostics and a KBR45 forecast
  prototype, but it is not itself a promoted forecasting model.
- Production forecast cache and dashboard artifacts were not changed.

## Verification

Executed:

```bash
python3 experiments/kbr_45_component_mapping/build_kbr_45_component_mapping.py
python3 -m py_compile experiments/kbr_45_component_mapping/build_kbr_45_component_mapping.py
```

The script produced the mapping artifacts and reconciled 45-component weights to
1.00000 under both canonical and May 2026 official regional weights.

## Next Step

Build `experiments/kbr_45_forecast_prototype/` on top of this mapping:

1. 45-component baseline forecasts;
2. scenario overrides for fuel, ЖКУ, плодоовощи and other tagged components;
3. headline aggregation under latest and canonical weights;
4. comparison against `SubcomponentMulti`, `Micro_SM`, Huber and Ensemble.
