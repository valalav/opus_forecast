
import json

prd_path = 'edge_lab/tasks/prd.json'

with open(prd_path, 'r') as f:
    prd = json.load(f)

# 1. Удаляем старые, слишком общие задачи
ids_to_remove = [24, 25, 29, 30]
prd['user_stories'] = [t for t in prd['user_stories'] if t['id'] not in ids_to_remove]

# 2. Создаем новые атомарные задачи

# --- ExogProphet (Brent) ---
tasks_exog = [
    {
        "id": 241,
        "title": "ExogProphet: Lag Optimization",
        "description": "Before adding Brent to Prophet, determine the optimal lag. \n1. Load CPI and Brent data. \n2. Calculate Cross-Correlation Function (CCF) for lags 0-12 months. \n3. Identify the lag with highest correlation. \n4. OUTPUT: 'data/brent_lag_analysis.json' (optimal_lag: int, correlation: float).",
        "priority": "medium",
        "status": "TODO",
        "acceptance_criteria": [
            "Output JSON exists",
            "Optimal lag is mathematically calculated (not hardcoded)"
        ]
    },
    {
        "id": 242,
        "title": "ExogProphet: Controlled Experiment",
        "description": "Run a comparative backtest. \n1. Model A: Prophet (Univariate). \n2. Model B: Prophet + Brent (shifted by optimal lag from Task 241). \n3. Horizon: h=1, 12 months backtest. \n4. OUTPUT: 'data/exog_prophet_experiment.csv' (Model, MAE).",
        "priority": "medium",
        "status": "TODO",
        "acceptance_criteria": [
            "Experiment results saved",
            "Code proves Model B actually uses the regressor"
        ]
    }
]

# --- Reporting ---
tasks_report = [
    {
        "id": 251,
        "title": "Reporting: Metrics Aggregation",
        "description": "Collect backtest results from all horizons (h=1, h=2, h=12). \n1. Read 'archive/results/backtest_h*_metrics.csv'. \n2. Standardize column names. \n3. Calculate 'Weighted Score' (50% h1 + 30% h2 + 20% h12). \n4. OUTPUT: 'data/consolidated_metrics.csv'.",
        "priority": "low",
        "status": "TODO",
        "acceptance_criteria": [
            "Consolidated CSV contains all models",
            "Weighted Score calculated correctly"
        ]
    },
    {
        "id": 252,
        "title": "Reporting: Visualization Generator",
        "description": "Create a script 'scripts/generate_report_charts.py'. \n1. Generate 'assets/charts/mae_comparison.png' (Bar Chart). \n2. Generate 'assets/charts/forecast_trajectories.png' (Line Chart). \n3. Use 'consolidated_metrics.csv' as source.",
        "priority": "low",
        "status": "TODO",
        "acceptance_criteria": [
            "PNG files generated successfully",
            "Charts include title and legends"
        ]
    },
    {
        "id": 253,
        "title": "Reporting: HTML Assembly",
        "description": "Generate 'assets/reports/model_performance.html'. \n1. Embed the PNGs from Task 252. \n2. Convert 'consolidated_metrics.csv' to an HTML table. \n3. Add timestamp and interpretation text.",
        "priority": "low",
        "status": "TODO",
        "acceptance_criteria": [
            "HTML file exists and opens in browser",
            "Contains both images and data table"
        ]
    }
]

# --- Caching (Persistence) ---
tasks_cache = [
    {
        "id": 291,
        "title": "Infrastructure: Persistent Disk Cache",
        "description": "Implement 'joblib.Memory' or 'diskcache' for forecasts. \n1. Create 'sirena/cache_manager.py'. \n2. Configure cache dir: '.cache/forecasts'. \n3. Decorate 'fit()' and 'predict()' methods of BaseForecaster. \n4. Ensure cache invalidates if data changes (check data hash).",
        "priority": "medium",
        "status": "TODO",
        "acceptance_criteria": [
            "Cache files appear in .cache/ directory",
            "Second run of a model takes < 1 second"
        ]
    }
]

# --- Documentation ---
tasks_docs = [
    {
        "id": 301,
        "title": "Docs: Auto-Inspector",
        "description": "Create 'scripts/doc_gen.py'. \n1. Inspect 'sirena.models' package. \n2. Extract Docstrings and __init__ parameters for every model class. \n3. OUTPUT: 'docs/MODELS_AUTO.md'.",
        "priority": "medium",
        "status": "TODO",
        "acceptance_criteria": [
            "Markdown file contains all 37+ models",
            "Parameters list is accurate"
        ]
    },
    {
        "id": 302,
        "title": "Docs: Performance Leaderboard",
        "description": "Update 'CLAUDE.md' and 'README.md' with current metrics. \n1. Read 'data/consolidated_metrics.csv'. \n2. Format Top-5 models as a Markdown table. \n3. Inject into documentation files using placeholders.",
        "priority": "medium",
        "status": "TODO",
        "acceptance_criteria": [
            "CLAUDE.md contains up-to-date MAE values",
            "No manual editing required"
        ]
    }
]

# Добавляем новые задачи в конец списка
all_new = tasks_exog + tasks_report + tasks_cache + tasks_docs
prd['user_stories'].extend(all_new)

# Сортируем по ID для порядка
prd['user_stories'].sort(key=lambda x: x['id'])

with open(prd_path, 'w') as f:
    json.dump(prd, f, indent=2, ensure_ascii=False)

print(f"PRD Refactored. Added {len(all_new)} atomic tasks.")
