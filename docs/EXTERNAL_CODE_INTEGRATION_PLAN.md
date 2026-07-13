# External Code Integration Plan For SIRENA-KBR

Date: 2026-06-25

Purpose: turn the unpacked external code repository into a structured SIRENA-KBR
improvement program. This document is the working plan for what to study,
prototype, backtest, document, and eventually promote.

Source intake:

- Archive and extraction map: `experiments/code_repository_20260625/`
- First-study list: `experiments/code_repository_20260625/STUDY_FILES.md`
- Direct code review note:
  `archive/results/analysis_notes/2026-06-25_external_code_deep_dive.md`

## Core Principle

External code is not production evidence.

Every useful external idea must pass through the SIRENA-KBR route:

1. code review;
2. Python-native prototype using SIRENA data loaders;
3. cutoff-safe backtest;
4. diagnostics and leakage review;
5. documented decision: reject, keep as diagnostic, or promote.

Do not directly copy EViews/R/Excel workflows into production.

## Current State

Already done:

- external archive moved and unpacked;
- nested archive inventory created;
- first code review completed;
- `weekCpiR` weighted weekly mechanics identified as the most immediate nowcast
  improvement;
- first `Weekly Laspeyres Nowcast` prototype built and run for June 2026.

Important current result:

- June 2026 weighted weekly pressure from matched items is about `+0.58` to
  `+0.61` p.p.
- This supports a June nowcast materially above the old calm `100.3` area.
- Prototype remains diagnostic; it is not in the production ensemble.

## Workstreams

### WS1. Weighted Weekly Nowcast

Source inspiration:

- `weekCpiR` under
  `experiments/code_repository_20260625/nested_extracted/a24cd898db7c9/5 Недельная инфляция/weekCpiR/`

Existing SIRENA locations:

- `sirena/data/weekly_bridge.py`
- `docs/NOWCASTING.md`
- `scripts/weekly_bridge_nowcast.py`
- `scripts/precompute_forecasts.py`

Current prototype:

- `experiments/weekly_laspeyres_nowcast/run_weekly_laspeyres_nowcast.py`
- outputs:
  `archive/results/weekly_laspeyres_nowcast_20260625/`

What to do next:

1. Build a cutoff-safe backtest for weighted weekly signals.
2. Compare current `weekly_bridge_v1` versus weighted partial headline signal.
3. Add stable diagnostics to precompute output only after backtest sanity.
4. Keep production `Ensemble` unchanged until evidence is available.

Acceptance gates:

- no use of monthly facts before publication;
- item matching table versioned and reviewable;
- coverage reported every run;
- top drivers reported every run;
- backtest beats or explains current weekly bridge on h=0/h=1 nowcast.

Decision status:

- **Prototype active**
- **Not production**

### WS2. KBR 45-Component Bottom-Up Layer

Source inspiration:

- Khabarovsk:
  `experiments/code_repository_20260625/nested_extracted/a2363877985ef/Прогноз_ИПЦ_по компонентам_Хабаровск/khab_mod.prg`
- Omsk:
  `experiments/code_repository_20260625/nested_extracted/afa0cc7c1c88b/ARIMA-45 (Омск, СГУ)/Скрипт/arima_omsk.prg`
- Mordovia Python baseline:
  `experiments/code_repository_20260625/nested_extracted/a0268154c4591/ВВГУ_Мордовия_КСП инфляции Python/infl.py`

Existing SIRENA locations:

- `sirena/models/subcomponent_multi.py`
- `sirena/models/micro_statsmodels_external.py`
- `data/external/micro_cpi_region_export/region_cpi_long.csv`
- `data/micro_sprav.csv`
- `data/access_weights.csv`

Why this matters:

- current full micro is useful but noisy and hard to interpret;
- current `SubcomponentMulti` is strong but internally SIRENA-specific;
- a 45-component middle layer can become the stable policy/explanation layer:
  fuel, ЖКУ, плодоовощи, services, food, non-food.

What to do next:

1. Build a baseline `KBR45_ARIMA` or `KBR45_ScenarioLayer` prototype with
   conservative fallbacks.
2. Compare against `SubcomponentMulti`, `Micro_SM`, Huber and Ensemble.
3. Use the 45-layer mapping for transparent July/August and tariff/fuel
   scenario overrides before promoting any model.

Suggested experiment folder:

```text
experiments/kbr_45_component_mapping/
```

Current mapping artifacts:

- builder:
  `experiments/kbr_45_component_mapping/build_kbr_45_component_mapping.py`
- component map:
  `experiments/kbr_45_component_mapping/kbr_45_component_mapping.csv`
- item map:
  `experiments/kbr_45_component_mapping/kbr_45_micro_item_mapping.csv`
- summary:
  `experiments/kbr_45_component_mapping/kbr_45_component_mapping_summary.csv`
- report:
  `experiments/kbr_45_component_mapping/kbr_45_component_mapping_report.md`

Current mapping result:

- external 45 variables mapped: 45 / 45;
- canonical 45-component weights sum to 1.00000;
- latest May 2026 regional 45-component weights sum to 1.00000;
- item-level `micro_sprav` weight assigned to the 45 layer: 0.98565;
- service item allocation is inferred by name rules because `micro_sprav.csv`
  does not carry service subcomponent labels.

Current forecast prototype:

- runner:
  `experiments/kbr_45_forecast_prototype/run_kbr45_forecast_prototype.py`
- outputs:
  `archive/results/kbr45_forecast_prototype_20260625/`
- report:
  `archive/results/kbr45_forecast_prototype_20260625/kbr45_forecast_report.md`

Current forecast prototype result:

- last official fact used: May 2026;
- latest-weight baseline: June 100.136, July 100.288, August 99.838;
- tariff scenario with July ЖКУ=100 and October ЖКУ=110:
  July baseline 100.288 -> 100.040, October baseline 100.577 -> 101.464;
- cutoff-safe diagnostic backtest, canonical weights:
  h=1 MAE 0.417 p.p., h=2 MAE 0.424 p.p., h=12 MAE 0.453 p.p.

Important caveat:

- June baseline does not include the separate weekly gasoline nowcast and must
  not replace the weekly Laspeyres signal for June 2026.

Current comparison package:

- builder:
  `experiments/kbr45_forecast_comparison/build_kbr45_comparison.py`
- outputs:
  `archive/results/kbr45_forecast_comparison_20260625/`
- report:
  `archive/results/kbr45_forecast_comparison_20260625/kbr45_forecast_comparison_report.md`

Current comparison result:

- production cache `Ensemble`: June 100.484, July 100.649, August 100.340;
- production cache `Nowcast`: June 100.810;
- KBR45 baseline with June weekly fuel overlay: June 100.682, July 100.288,
  August 99.838;
- KBR45 tariff scenario with June weekly fuel overlay: June 100.682,
  July 100.040, August 99.838;
- June weekly fuel overlay uses 2026-06-22 weekly observation and adds about
  0.546 p.p. versus the mechanical KBR45 June fuel contribution.

Acceptance gates:

- no double-counting of aggregate and leaf items;
- component weights reconcile;
- h=1/h=2/h=12 backtests saved;
- scenario overrides can be applied to fuel/ЖКУ without breaking aggregation;
- trajectory realism checked.

Decision status:

- **Mapping prototype complete**
- **Forecast prototype active**
- **Comparison package active**
- **Not production**

### WS3. Variable-Selection Gate

Source inspiration:

- `experiments/code_repository_20260625/nested_extracted/a0ac088b0b1be/Методика отбора переменных, обеспечивающих надежные прогнозы/variable_models_code.prg`

Existing SIRENA artifacts:

- `archive/results/variable_selection_pilot_20260625/`
- `sirena/macro_features.py`
- `sirena/models/ridge_macro.py`
- `sirena/models/var_policy.py`

Why this matters:

- we need a disciplined way to decide which macro or policy variables enter
  models;
- simple correlation is not enough;
- direct future exogenous values are not allowed.

What to do next:

1. Promote the pilot into a reusable experiment runner.
2. Evaluate variable blocks:
   - monetary: Ki, Ruonia, spread;
   - demand: retail, paid services, deposits;
   - external: USD, Brent;
   - weekly/fuel;
   - policy/tariff dummies.
3. Produce `selected_features_by_horizon.csv`.
4. Use selected blocks only as challengers first.

Suggested experiment folder:

```text
experiments/variable_selection/
```

Acceptance gates:

- h=1, h=2 and h=12 reported separately;
- AR benchmark included;
- RRMSE and outperform ratio reported;
- features are lagged or deterministic at cutoff;
- selected variables improve an existing model in rolling backtest.

Decision status:

- **Required before new macro features are promoted**

### WS4. ARIMAX / ARDL Challengers

Source inspiration:

- ARIMAX:
  `experiments/code_repository_20260625/nested_extracted/ae00677697d94/ARIMAX.R`
- ARDL:
  `experiments/code_repository_20260625/nested_extracted/ac2294d8a53ba/ARDL_Челябинск/`

Existing SIRENA locations:

- `sirena/models/ridge.py`
- `sirena/models/huber.py`
- `sirena/models/exog_forecaster.py`
- `sirena/macro_features.py`

Use only after WS3.

Rules:

- do not port `Прогноз регрессоров.xlsx` as fact;
- exogenous paths must be last-observed, AR forecast, or documented scenario;
- use as transparent challenger, not as default production replacement.

Acceptance gates:

- residual diagnostics;
- leakage-safe exogenous paths;
- comparison against Huber/Ridge/SubcomponentMulti;
- trajectory realism for h=12.

Decision status:

- **Later challenger**

### WS5. ML Challengers

Source inspiration:

- Volgograd ML Rmd files under
  `experiments/code_repository_20260625/nested_extracted/a9c66f72e388c/`

Existing SIRENA locations:

- `sirena/models/xgboost_model.py`
- `sirena/models/ngboost_model.py`
- `sirena/models/ebm.py`
- `scripts/backtest_framework.py`

Finding:

- unpacked `main1.xlsx` has only `date` and `ipc`;
- R `tidymodels` code is not worth porting directly;
- ensemble weighting in `Ensembles.Rmd` uses realized holdout errors and should
  not be copied.

What to do:

- use only as a feature-family reminder;
- if new ML work is done, do it through existing SIRENA model classes and
  rolling backtests.

Decision status:

- **Not immediate**

## Documentation Updates Required As Work Progresses

When a workstream advances, update:

- `docs/EXTERNAL_CODE_INTEGRATION_PLAN.md` - status and decisions;
- `docs/NOWCASTING.md` - if weekly weighted diagnostics become stable;
- `docs/MODEL_CATALOG.md` - only if a model is integrated;
- `docs/ADDING_MODEL_GUIDE.md` - only if new recurring model-integration rules
  appear;
- `archive/results/analysis_notes/analysis_index.csv` - every saved analysis or
  prototype result.

If a prototype becomes production-visible, also update:

- `scripts/precompute_forecasts.py` documentation/comments;
- relevant verification notes under `archive/results/`;
- dashboard docs if charts/tabs change.

## Promotion Rules

Use these statuses:

- `reviewed` - code inspected, no SIRENA artifact yet;
- `prototype` - Python experiment exists and produces outputs;
- `diagnostic` - regularly generated but not in production ensemble;
- `challenger` - included in backtests against production models;
- `production_candidate` - passed backtest and diagnostics, ready for review;
- `production` - registered and visible in model catalog/dashboard path;
- `rejected` - documented reason not to continue.

Do not call a workstream "production" unless model registration, backtests and
forecast/chart verification are complete.

## Recommended Next Move

Move next to **WS2 validation: compare KBR45 scenario layer against production
models and add a weekly overlay**.

Reason:

- WS1 already has a prototype and a clear backtest task;
- WS2 mapping and first forecast prototype are now available;
- June/July/August control-point discussion needs an explainable component
  layer;
- tariff and gasoline scenarios are better handled at this stable middle layer
  than by full micro or headline-only models.

Concrete next artifact:

```text
archive/results/kbr45_forecast_comparison_20260625/
```

It should produce:

- side-by-side h=1/h=2/h=12 errors for KBR45, `SubcomponentMulti`, `Micro_SM`,
  Huber and Ensemble;
- a June 2026 weekly overlay for `n_topl` / fuel;
- explicit July/August control-point table with baseline, tariff-shift and
  gasoline downside/upside scenarios.

Only after that should the layer be considered for model catalog or dashboard
visibility.
