# Core Inflation Tool Implementation Notes

Date: 2026-05-27

## Scope

Implementation is isolated under `experiments/core_inflation_tool/`. Planning and status notes are isolated under `experiments/core_inflation_tool_plan/`.

No production data under `data/`, existing model code under `sirena/`, operational scripts under `scripts/`, or chart assets under `assets/charts/` are edited by this MVP.

## Method

- Load component MoM/YoY indices from `data/kbr_indices.csv`.
- Load annual KBR weights from `data/access_weights.csv`.
- Map weights to monthly rows by `date.year`.
- Convert index values to growth as `index - 100`.
- Exclude aggregate item codes before component aggregation.
- Calculate exclusion core, weighted trimmed mean with partial boundary trimming, and weighted median.
- Emit diagnostics for MoM/YoY contamination, MoM ranges, weight-sum stability, final numeric values, and jumps.
- Write all generated outputs under `experiments/core_inflation_tool/outputs/`.

## Verification Plan

The final acceptance gate is:

```bash
git status --short --branch
python3 -m pytest experiments/core_inflation_tool/tests
python3 -m py_compile $(find experiments/core_inflation_tool -name '*.py' -print)
python3 -m experiments.core_inflation_tool.core_inflation.cli \
  --config experiments/core_inflation_tool/config/core_inflation_config.yaml \
  --output experiments/core_inflation_tool/outputs/latest
test -f experiments/core_inflation_tool/outputs/latest/core_inflation_series.csv
test -f experiments/core_inflation_tool/outputs/latest/core_inflation_diagnostics.csv
test -f experiments/core_inflation_tool/outputs/latest/core_inflation_contributions.csv
test -f experiments/core_inflation_tool/outputs/latest/core_inflation_jump_report.md
git diff --check
```

## Current Gate Evidence

Controller checks run on 2026-06-05 for the extended ordinary+SA and long-run version:

- `python3 -m pytest experiments/core_inflation_tool/tests -q` -> 31 passed.
- `python3 -m experiments.core_inflation_tool.core_inflation.cli --config experiments/core_inflation_tool/config/core_inflation_config.yaml --output experiments/core_inflation_tool/outputs/latest` -> passed and wrote eight output artifacts.
- New output artifacts:
  - `core_inflation_series.csv` with ordinary and SA columns, including `stable_core_mom` and `stable_core_sa_mom`.
  - `core_inflation_sa_contributions.csv`.
  - `core_inflation_longrun_metrics.csv`.
  - `core_inflation_dynamics_report.md`.
- Real-data diagnostics produced no `fail` statuses. Warnings remain for extreme component MoM values, weight-sum variation, SA rows with missing component values, and unequal ordinary/SA edge coverage.
- Long-run metrics cover ordinary rows from 2016-01-01 to 2026-01-01 and SA rows from 2016-01-01 to 2026-04-01.
- `stable_core_sa_mom` long-run metrics: 124 months, mean 0.484% MoM, annualized mean 5.968%, std 0.550 p.p., MAE to next-12-month average headline 0.338 p.p., gain vs current headline as naive estimate 0.118 p.p.
- `stable_core_mom` long-run metrics: 121 months, mean 0.471% MoM, annualized mean 5.796%, std 0.507 p.p., MAE to next-12-month average headline 0.334 p.p., gain vs current headline as naive estimate 0.122 p.p.

Notes on interpretation:

- The working indicator is `stable_core = mean(exclusion_core, trimmed_mean)`.
- `weighted_median` is retained as a robust diagnostic/lower central estimate, not the default management indicator, because detailed item-level data can over-represent zero/small movements.
- The SA branch cannot run MoM/YoY contamination checks because `data/mom_sa_kbr.csv` does not include YoY columns; this is recorded as `expected_skip`.

Earlier MVP evidence:

Controller checks run on 2026-05-27:

- `python3 -m pytest experiments/core_inflation_tool/tests -q` -> 26 passed.
- `python3 -m py_compile $(find experiments/core_inflation_tool -name '*.py' -print)` -> passed.
- Real-data CLI with `experiments/core_inflation_tool/config/core_inflation_config.yaml` wrote outputs under `experiments/core_inflation_tool/outputs/latest`.
- Required output files exist: series, diagnostics, contributions, jump report.
- Real-data diagnostics produced pass/warning statuses, no failed diagnostics: `mom_yoy_distinct` passed; range and weight-sum stability warned.

## Acceptance Gate Blocker

The exact final command `git diff --check` currently fails on an unrelated pre-existing change outside the allowed write scope:

```text
GEMINI.md:385: trailing whitespace.
```

This goal's hard rules only allow implementation under `experiments/core_inflation_tool/` and planning/status docs under `experiments/core_inflation_tool_plan/`. The controller therefore did not edit `GEMINI.md`. The isolated experiment diff should be checked separately until the unrelated whitespace is fixed by its owner.

Second confirmation on 2026-05-27:

- `python3 -m pytest experiments/core_inflation_tool/tests -q` -> 26 passed.
- `python3 -m py_compile $(find experiments/core_inflation_tool -name '*.py' -print)` -> passed.
- Real-data CLI wrote the same five output artifacts under `experiments/core_inflation_tool/outputs/latest`.
- Required output file checks passed.
- `git diff --check -- experiments/core_inflation_tool experiments/core_inflation_tool_plan experiments/__init__.py` -> passed.
- Repository-wide `git diff --check` still fails only on `GEMINI.md:385: trailing whitespace`.

Third confirmation on 2026-05-27:

- `python3 -m pytest experiments/core_inflation_tool/tests -q` -> 26 passed.
- `python3 -m py_compile $(find experiments/core_inflation_tool -name '*.py' -print)` -> passed.
- Real-data CLI again wrote the five output artifacts under `experiments/core_inflation_tool/outputs/latest`.
- Required output file checks passed.
- `git diff --check -- experiments/core_inflation_tool experiments/core_inflation_tool_plan experiments/__init__.py` -> passed.
- Repository-wide `git diff --check` still fails only on `GEMINI.md:385: trailing whitespace`, outside this goal's allowed write scope.
