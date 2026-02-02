# AGENTS.md

This file provides guidance to AI agents (Claude, Gemini, etc.) when working with the СИРЕНА-КБР project.

## 🎯 Project Context

**СИРЕНА-КБР v5.3** — Inflation forecasting system for Kabardino-Balkarian Republic (КБР), Russia.

**Domain:** Economic forecasting, time series analysis, macroeconomics.

**Language:** Russian (data, docs), English (code comments, some docs).

---

## 📁 Critical File Locations

### Data Files (Source of Truth)

| File | Path | Description |
|------|------|-------------|
| **Main inflation data** | `data/inflation_data.csv` | Monthly inflation (mom), components, macro indicators |
| **KBR detailed data** | `data/infl_kbr.csv` | Alternative format with components |
| **Weekly prices** | `data/kbr_weekly_prices_2008_2026.csv` | Weekly price data for nowcasting |
| **Precomputed forecasts** | `data/precomputed_forecasts.json` | Cached forecasts |

### Results & Backtests

**Production results:**
```
archive/results/
├── backtest_h1_predictions.csv       # h=1 forecasts (most important KPI)
├── backtest_h1_metrics.csv           # h=1 metrics
├── backtest_h2_predictions.csv       # h=2 forecasts
├── backtest_h12_predictions.csv      # h=12 forecasts (annual trajectory)
└── model_comparison.csv              # Model comparison table
```

**Charts & Visualizations:**
```
assets/charts/
├── backtest_h1_predictions.html      # Interactive h=1 charts
├── forecasts.html                    # Forecast visualizations
├── nowcast.html                      # Nowcast display
└── sirena_score_dynamics.html        # SIRENA Score tracking
```

### Model Files

**Production models (40+):**
```
sirena/models/
├── ridge.py                          # Baseline Ridge model
├── huber.py                          # Best h=1 model (MAE 0.289)
├── subcomponent_multi.py             # Best SIRENA Score (0.515)
├── prophet.py                        # Best h=12 model
└── ...
```

### Experiments

**Isolated experiments:**
```
experiments/
└── {experiment_name}/
    ├── models/                       # Experimental models
    ├── scripts/                      # Experiment scripts
    ├── results/                      # Experiment results & charts
    └── docs/                         # Documentation
```

---

## 🧠 Key Concepts

### Forecasting Horizons
- **h=1**: 1 month ahead — **PRIMARY KPI** (most important)
- **h=2**: 2 months ahead
- **h=3, h=6**: Medium-term
- **h=12**: 12 months ahead — Annual trajectory

### Key Metrics
- **MAE**: Mean Absolute Error (main metric)
- **KPI Rate**: % of forecasts with |error| ≤ 0.5
- **SIRENA Score**: Weighted composite score `(0.5×MAE_h1 + 0.3×MAE_h2 + 0.2×MAE_h12)`

### Seasonality Handling
- **Global seasonality**: Historical average (all years, excluding outliers)
- **Rolling seasonality**: Last N months (adaptive)
- **ETS weights**: Exponential smoothing seasonal weights

### Data Components
- **Total**: "Все товары и услуги" (All goods and services)
- **Food**: "Продовольственные товары" (39.48% weight)
- **NonFood**: "Непродовольственные товары" (36.53% weight)
- **Services**: "Услуги" (23.42% weight)

---

## 🛠️ Development Workflow

### Adding a New Model
1. Create model in `sirena/models/{name}.py`
2. Register in `ModelRegistry` (use decorator `@ModelRegistry.register("name")`)
3. Export in `sirena/models/__init__.py`
4. Add to `pages/constants.py` if needed for dashboard
5. Run backtests: `python3 scripts/run_backtest_h1.py`
6. Document results in `archive/results/`

### Running Backtests
```bash
# Production backtests
python3 scripts/run_backtest_h1.py    # Results → archive/results/backtest_h1_*
python3 scripts/run_backtest_h2.py    # Results → archive/results/backtest_h2_*
python3 scripts/run_backtest_h12.py   # Results → archive/results/backtest_h12_*
```

### Experiment Workflow
```bash
# Create experiment
cd experiments
mkdir -p new_experiment/{models,scripts,results,docs}

# Run experiment
cd new_experiment
python3 scripts/run_experiment.py     # Results → experiments/new_experiment/results/

# Visualize
python3 scripts/plot_results.py       # Charts → experiments/new_experiment/results/*.png
```

---

## 📊 Output Artifacts

### Must be saved explicitly:

1. **Backtest results**: `archive/results/backtest_{h}_{timestamp}.csv`
2. **Metrics**: `archive/results/backtest_{h}_metrics.csv`
3. **Charts**: `assets/charts/` (HTML/PNG)
4. **Experiment results**: `experiments/{name}/results/`

### Naming conventions:
- Results: `{type}_{timestamp}.csv` (e.g., `backtest_h1_20260202_155929.csv`)
- Predictions: `predictions_{model}_{timestamp}.csv`
- Charts: `{description}.png` or `{description}.html`

---

## ⚠️ Critical Rules

1. **Never modify production data** in `data/` without backup
2. **Always commit results** from experiments to git
3. **Document paths explicitly** when mentioning files
4. **Use full paths** when referencing results: `archive/results/backtest_h1_metrics.csv`
5. **Preserve experiment isolation**: experiments should not affect production code

---

## 🔗 Related Documentation

- [GEMINI.md](GEMINI.md) — Main project guidance
- [README.md](README.md) — Full project documentation
- [docs/index.md](docs/index.md) — Documentation navigation
- [experiments/rolling_seasonality/README.md](experiments/rolling_seasonality/README.md) — Example experiment

---

*Created: 2026-02-02*  
*Version: 1.0*
