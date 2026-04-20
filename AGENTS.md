# AGENTS.md

This file provides compact operational guidance for AI agents working with the СИРЕНА-КБР repository.

## Core Rule: do not pretend work is verified

- Never say "done", "works", or "verified" without checking the exact thing the user asked for.
- "Script ran" is not the same as "feature works".
- If visual verification is impossible, say so plainly and name the exact file, tab, chart, or artifact the user should check.
- If the user says something is broken, read the existing code first. Do not invent replacement workflows before understanding the current one.

## Start Here

When entering a new task:

1. Check `docs/index.md` first.
2. Open only the docs needed for the current task.
3. Reuse existing models, loaders, scripts, and verification paths before creating anything new.

### Lazy-loading documentation

- Adding or updating a model → `docs/ADDING_MODEL_GUIDE.md`, `docs/MODEL_CATALOG.md`
- Verification or "is this really working?" → `docs/VERIFICATION_GUIDE.md`
- Nowcasting / weekly data → `docs/NOWCASTING.md`
- Full project context → `README.md`

Do not read the entire docs tree by default. Use `docs/index.md` as the navigation hub.

## Prefer Skills / Existing Workflows

If the environment exposes skills or commands, prefer them over ad-hoc flows.

### Project skills / commands to prefer

- `/models-status` — inspect current model landscape/status before proposing a new model path
- `/charts` — regenerate or inspect chart-related outputs instead of creating custom chart scripts
- `/add-model` — use when adding a new model or extending model registration workflow

### Repository workflows to prefer

- `/update-nowcast` — nowcast refresh workflow
- `/run-backtest` — backtest workflow

### Reuse before reinventing

Before creating a new model, feature engineering path, or data loader, check for existing patterns in:

- `sirena/models/`
- `scripts/backtest_framework.py`
- `sirena/data/`
- `sirena/macro_features.py`
- `docs/MODEL_CATALOG.md`

If a similar implementation already exists, extend it or mirror its pattern instead of building a parallel one.

## Project Context

**СИРЕНА-КБР v5.4** — inflation forecasting system for Kabardino-Balkarian Republic (КБР), Russia.

- Domain: inflation forecasting, time series, macroeconomics
- Main KPI: h=1 forecast quality, especially keeping error within ±0.5 p.p.
- Dashboard: `http://localhost:8503`

## Canonical Data and Artifact Paths

### Source-of-truth data

- `data/inflation_data.csv` — main monthly inflation data, components, macro indicators
- `data/infl_kbr.csv` — alternative KBR inflation format
- `data/kbr_weekly_prices_2008_2026.csv` — weekly prices for nowcasting
- `data/precomputed_forecasts.json` — cached forecast outputs
- `data/micro_sprav.csv` — primary item weights reference
- `data/access_weights.csv` — extended ACCDB-derived weights reference

### Results and charts

- `archive/results/` — production backtest outputs and comparison tables
- `assets/charts/` — HTML and chart artifacts users actually inspect
- `sync/` — Syncthing export folder

### Important model / method locations

- `sirena/models/` — forecasting models
- `sirena/models/registry.py` — model registration pattern
- `sirena/models/base.py` — common forecaster contract
- `sirena/macro_features.py` — reusable macro and related feature engineering
- `sirena/data/weekly_loader.py` — weekly data loading for nowcasting

## Model Work Rules

### When modifying or adding models

Always look for an existing pattern first:

- linear / robust production-style models → `ridge.py`, `ridge_extended.py`, `huber.py`, `ridge_shock_dummies.py`
- subcomponent / bottom-up logic → `subcomponent_multi.py`
- exogenous trajectories → `exog_forecaster.py`
- nowcasting / weekly signal models → weekly and nowcast modules under `sirena/models/` and `sirena/data/weekly_loader.py`

### Minimum reuse checklist

Before writing new model code, check whether the task should instead:

1. extend an existing model file,
2. reuse `ModelRegistry`,
3. reuse the existing backtest framework,
4. reuse current feature engineering helpers,
5. reuse current chart/report generation scripts.

### Adding a new model

At minimum:

1. Create `sirena/models/{name}.py`
2. Register it with `@ModelRegistry.register("name")`
3. Export it in `sirena/models/__init__.py`
4. Wire it into the existing backtest/dashboard path if the task requires production visibility
5. Run the relevant backtests
6. Regenerate forecast and chart artifacts if outputs changed

Do not create a brand-new one-off evaluation path if the repository already has one.

## Data Source Rules

- Do not modify production data under `data/` without a clear reason and backup awareness.
- When discussing findings, name exact file paths.
- Treat weekly, macro, budget, and weights data as distinct source classes; do not conflate them.
- If a model uses external or auxiliary data, identify the exact loader/file and whether the use is production, experimental, or nowcasting-only.

## Verification Rules

### Before claiming completion on model / forecast / dashboard work

Run the closest relevant checks, typically:

```bash
python3 scripts/verify_all_tabs.py
python3 scripts/precompute_forecasts.py
python3 scripts/generate_charts.py
```

Use the smallest valid verification set for the task, but do not skip chart regeneration when model outputs or forecasts changed and users rely on `assets/charts/`.

### For new model integration

Also use:

```bash
python3 scripts/add_model_checklist.py ModelName
```

If verification is partial, say exactly what was checked and what remains unchecked.

## Backtests, charts, and sync

### Main backtests

```bash
python3 scripts/run_backtest_h1.py
python3 scripts/run_backtest_h2.py
python3 scripts/run_backtest_h12.py
```

### Forecast/charts refresh

```bash
python3 scripts/precompute_forecasts.py
python3 scripts/generate_charts.py
```

### Sync trigger

When the user says **"синхронизируй"** or **"обнови sync"**:

```bash
python3 scripts/sync_to_share.py
```

## File Organization

- Root: keep only essential top-level project files
- `docs/` for documentation
- `data/` for datasets and exported data files
- `scripts/` for operational scripts
- `assets/charts/` for generated chart artifacts
- `experiments/` for isolated experiment work
- `archive/` for backtests, older artifacts, and historical outputs

Avoid cluttering the project root with new scratch files.

## Related references

- `GEMINI.md`
- `CLAUDE.md`
- `README.md`
- `docs/index.md`

---

Updated to align agent guidance with `GEMINI.md` and `CLAUDE.md`, with stronger emphasis on skill usage, model/data-source reuse, and verification discipline.
