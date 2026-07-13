# KBR 45-Component Mapping

Date: 2026-06-25

Purpose: canonical bridge between the external Khabarovsk/Omsk 45-component
model style and SIRENA-KBR data.

## Inputs

- SIRENA 45-component weights: `data/raw/subcomp_sprav.csv`
- KBR item-level weights: `data/micro_sprav.csv`
- Current official regional long export: `data/external/micro_cpi_region_export/region_cpi_long.csv`

## Outputs

- Component mapping: `experiments/kbr_45_component_mapping/kbr_45_component_mapping.csv`
- Micro item mapping: `experiments/kbr_45_component_mapping/kbr_45_micro_item_mapping.csv`
- Parent summary: `experiments/kbr_45_component_mapping/kbr_45_component_mapping_summary.csv`
- Unmapped item diagnostics: `experiments/kbr_45_component_mapping/kbr_45_unmapped_micro_items.csv`

## Coverage

- 45 components mapped to external variable names: `45` / `45`
- Canonical 45-component weight sum: `1.00000`
- Latest regional 45-component weight sum (2026-05-01): `1.00000`
- Micro-sprav item weight assigned to the 45 layer: `0.98565`
- Unmapped micro-sprav item weight: `0.00000`

## Parent Components

| parent_component   |   components |   canonical_weight |   latest_region_weight |   micro_weight_sum |   micro_item_count |
|:-------------------|-------------:|-------------------:|-----------------------:|-------------------:|-------------------:|
| Nonprod            |           20 |            0.36534 |                0.36378 |            0.36119 |                284 |
| Prod               |           15 |            0.39481 |                0.39859 |            0.39216 |                134 |
| Serv               |           10 |            0.23985 |                0.23763 |            0.23230 |                119 |

## Largest Components

|   subcomponent_code | external_var   | subcomponent_name                                       | parent_component   |   canonical_weight |   latest_region_weight |   micro_weight_sum | scenario_tags                                             |
|--------------------:|:---------------|:--------------------------------------------------------|:-------------------|-------------------:|-----------------------:|-------------------:|:----------------------------------------------------------|
|                  26 | p_meat         | Мясопродукты                                            | Prod               |            0.09904 |                0.09993 |            0.09768 | food;core_food                                            |
|                  14 | u_gkh          | Жилищные и коммунальные услуги (включая аренду квартир) | Serv               |            0.09094 |                0.08898 |            0.08911 | services;regulated;tariff;gkh;july_october_scenario       |
|                  29 | n_odegd        | Одежда и белье                                          | Nonprod            |            0.06473 |                0.06545 |            0.06545 | nonfood;import_sensitive                                  |
|                  33 | p_ovosh        | Плодоовощная продукция, включая картофель               | Prod               |            0.05888 |                0.05546 |            0.05546 | food;volatile;fruit_vegetables;seasonal;weather_sensitive |
|                  54 | n_proch        | Другие непродовольственные товары                       | Nonprod            |            0.04447 |                0.03996 |            0.03996 | nonfood;core_nonfood                                      |
|                  12 | u_bit          | Бытовые услуги                                          | Serv               |            0.04365 |                0.04221 |            0.04284 | services;household_services;labor_cost_sensitive          |
|                  53 | p_proch        | Другие продовольственные товары                         | Prod               |            0.03665 |                0.03664 |            0.03664 | food;core_food                                            |
|                  42 | n_topl         | Топливо моторное                                        | Nonprod            |            0.03302 |                0.03384 |            0.03195 | nonfood;fuel;volatile;gasoline;weekly_nowcast             |
|                  17 | n_avto         | Легковые автомобили                                     | Nonprod            |            0.03074 |                0.03135 |            0.03135 | nonfood;durable;credit_sensitive;import_sensitive         |
|                  24 | p_milk         | Молоко и молочная продукция                             | Prod               |            0.03051 |                0.03299 |            0.02881 | food;core_food                                            |
|                  48 | u_sv           | Услуги телекоммуникационные                             | Serv               |            0.03021 |                0.02966 |            0.02998 | services;telecom;regulated_partial                        |
|                  16 | p_sweet        | Кондитерские изделия                                    | Prod               |            0.02899 |                0.02849 |            0.02849 | food;core_food                                            |

## Diagnostics

- Components without micro items: `0`
- Service micro allocation is inferred by item-name rules because
  `data/micro_sprav.csv` does not contain service subcomponent labels.
- The mapping is suitable for scenario diagnostics and a KBR45 prototype, but
  it is not yet a promoted forecasting model.

## Immediate Use

1. Use `scenario_tags` to apply transparent expert overrides to fuel, ЖКУ,
   плодоовощи, eggs, sugar, tourism, education and transport.
2. Use `latest_region_weight` for current headline contribution calculations
   when the official long export is available.
3. Use `canonical_weight` to stay compatible with existing
   `SubcomponentForecaster` and historical `data/raw/subcomp.csv` structure.
4. Build the first `KBR45_ARIMA` or scenario-aggregation prototype on top of
   this mapping, not on a new private spreadsheet.
