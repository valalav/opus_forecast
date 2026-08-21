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
- OPR send-ready forecast form / policy trajectory → `docs/OPR_FORECAST_LINKAGE.md`
- Forecast-vs-fact deviation reports → `docs/FORECAST_FACT_ANALYSIS.md`
- Food/tariff expert forecasts → `docs/FOOD_TARIFF_FORECAST_2026_2027.md`
- Saved short scenario notes / forecast memory → `docs/ANALYSIS_NOTES.md`, then `archive/results/analysis_notes/analysis_index.csv`
- Send-ready forecast package / verification report → `archive/results/full_forecast_package_2026_2027/verification_report.md`
- Full project context → `README.md`

Do not read the entire docs tree by default. Use `docs/index.md` as the navigation hub.

## Completed Task Log

- After every completed task, add one newest-first row to `docs/TASK_LOG.md` before committing.
- Required fields: ISO date, concise task, one-sentence brief description, and repo-relative link to the primary report or artifact.
- Keep specialized registries such as `archive/results/analysis_notes/analysis_index.csv` updated as well; the task log does not replace them.
- Do not mark a task complete until its log row has been added.

## Prefer Skills / Existing Workflows

If the environment exposes skills or commands, prefer them over ad-hoc flows.

### Project skills / commands to prefer

- `/models-status` — inspect current model landscape/status before proposing a new model path
- `/charts` — regenerate or inspect chart-related outputs instead of creating custom chart scripts
- `/add-model` — use when adding a new model or extending model registration workflow

### Repository workflows to prefer

- `/update-nowcast` — nowcast refresh workflow
- `/run-backtest` — backtest workflow
- Saved short scenario analysis — save the Markdown note under `archive/results/analysis_notes/` and add a row to `archive/results/analysis_notes/analysis_index.csv`
- Forecast-vs-fact analysis — follow `docs/FORECAST_FACT_ANALYSIS.md`; use `archive/results/april_2026_deviation_analysis/` as the worked example
- OPR forecast form updates — follow `docs/OPR_FORECAST_LINKAGE.md`; treat `assets/06_2026_02_Прогноз.xlsx` as the send-ready policy form, not a mechanical ensemble dump
- Food/tariff expert forecast — follow `docs/FOOD_TARIFF_FORECAST_2026_2027.md` before overriding ensemble forecasts for 2026–2027
- Mandatory VAR-family work — read `docs/VAR_MODEL_RESEARCH.md` before proposing or changing any VAR/BVAR/VARX/FAVAR model
- Full 2026–2027 forecast package — rebuild with `archive/results/full_forecast_package_2026_2027/build_package.py`; keep `verification_report.md` in the ZIP

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
- `archive/results/analysis_notes/analysis_index.csv` — stable registry of saved short scenario analyses; check it before continuing prior forecast discussions
- `assets/06_2026_02_Прогноз.xlsx` — current OPR send-ready forecast form; see `docs/OPR_FORECAST_LINKAGE.md` before editing
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

### VAR-family model rules

Before adding or changing VAR/BVAR/VARX/FAVAR models, read `docs/VAR_MODEL_RESEARCH.md`.

Do not start with naive grid-search alone. The minimum professional VAR workflow is:

1. diagnose outliers, shock periods, structural breaks, stationarity/scale, seasonality, and residual behavior;
2. test parsimonious variable subsets before all-macro specifications;
3. handle shocks robustly with cutoff-safe rules, robust estimation, or declared intervention logic;
4. keep exogenous paths deterministic and cutoff-safe; never use future actual USD/RUONIA/Ki values;
5. evaluate h=1, h=2, and h=12 separately when the model produces trajectories;
6. evaluate h=12 trajectory realism: seasonality, flatness, volatility, jumps, and explosive paths;
7. never add random noise to point forecasts to make a trajectory look realistic.

Current VAR-family recommendation:

- h=1 mandatory VAR component: `RegimeMacroVARX_l1`
- h=12 trajectory VAR component: `SeasonalVAR_CPI_F_NF_S`
- horizon-specific policy artifact: `experiments/var_sa_research/final_var_policy_report.md`

Rejected context to preserve:

- `fine_seasonal_resid_var_tc_roll42_l5` was a single-window overfit and must not be promoted.
- direct revised SA VAR backtests are not production evidence unless real-time vintages exist.
- all-macro VARX can help h=1 but often damages h=12 trajectory realism.

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
- Official monthly facts come from `data/inflation_data.csv`; weekly files are operational/nowcast signals, not monthly facts.
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

## Git workflow

- After completing and verifying a task that changes tracked project files, create a focused commit before yielding.
- Push every new commit immediately; never leave completed commits accumulating only in the local repository.
- Push the current branch to its configured upstream. If it has no upstream, use `git push --set-upstream origin HEAD`.
- If a push fails, keep the local commit intact, report the exact error, and do not claim the task is fully complete.
- Never include unrelated user changes, secrets, generated outputs, local workspaces, or oversized binaries merely to make the tree clean.

## Forecast-vs-fact reports

When the user asks to explain forecast deviation from fact:

1. Open `docs/FORECAST_FACT_ANALYSIS.md` first.
2. Use the sent forecast value from the user and the official fact from `data/inflation_data.csv`.
3. Decompose MoM with weights: `Prod=0.3986`, `Nonprod=0.3638`, `Serv=0.2376`.
4. Compare with the same calendar month historically using mean, median, trimmed mean, IQR fences, outliers, and robust z-score.
5. Identify product drivers from weekly data, but never call next-month weekly observations official facts.
6. Produce send-ready DOCX plus XLSX calculations under `archive/results/<month>_deviation_analysis/`.
7. Verify the report with independent scripts or a saved verification note before claiming it is send-ready.
8. Use `archive/results/april_2026_deviation_analysis/` as the example.

## Food/tariff expert forecast rules

When forecasting May 2026–April 2027 or similar food/tariff regimes:

1. Open `docs/FOOD_TARIFF_FORECAST_2026_2027.md` first.
2. Treat `data/mom_sa_kbr.csv` as the current KBR SA subcomponent source.
3. Use weekly prices only as nowcast signals until monthly facts appear in `data/inflation_data.csv`.
4. Check ACCDB exports for item history/weights: `data/kbr_indices.csv`, `data/access_weights.csv`, `data/items_names.csv`.
5. Account for 2026 tariff timing: no July services indexation; main services indexation shifts to October at 10%+.
6. Do not use pure ensemble forecasts when documented food shock or tariff-shift assumptions dominate; report expert adjustments explicitly.
7. For the current send-ready package, use `archive/results/full_forecast_package_2026_2027/`: CSV forecast, DOCX explanation with glossary for management, April forecast-vs-fact DOCX/XLSX, and `verification_report.md`.

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
