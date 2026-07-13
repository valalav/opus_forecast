import json

prd_path = 'edge_lab/tasks/prd.json'

with open(prd_path, 'r') as f:
    prd = json.load(f)

# Новые задачи для остальных больших файлов
new_tasks = [
    {
        "id": 118,
        "title": "Deep Dive: Labor Market (111MB Monster)",
        "description": "Process 'Зарплаты и СЧР (полная база).xlsx'. This file is critical for Phillips Curve modeling. 1. Use Crawler to map sheets. 2. Locate KBR specific rows. 3. Extract Nominal Wage, Real Wage, and Employment data. 4. OUTPUT: 'data/kbr_labor_market.csv'.",
        "priority": "high",
        "status": "TODO",
        "acceptance_criteria": [
            "Extracted > 50 labor market series for KBR",
            "Validated against official Rosstat summaries",
            "Includes sectoral breakdown (Wages by industry)"
        ]
    },
    {
        "id": 119,
        "title": "Deep Dive: Producer Prices (PPI 73MB)",
        "description": "Process 'Цены производителей/ОКВЭДОКПД2_цены_производителей_полный.xlsx'. PPI is a leading indicator for CPI (cost-push). 1. Extract manufacturing and agricultural PPI for KBR. 2. Check lag correlation with CPI. 3. OUTPUT: 'data/kbr_ppi_detailed.csv'.",
        "priority": "high",
        "status": "TODO",
        "acceptance_criteria": [
            "Extracted PPI series for KBR",
            "Correlation analysis confirms PPI leads CPI by 1-3 months"
        ]
    },
    {
        "id": 120,
        "title": "Deep Dive: GRP Forecasts (15MB)",
        "description": "Process 'ВРП/Опережающий индикатор ВРП (с прогнозом).xlsm'. This file contains pre-calculated forecasts. We must extract them to use as Exogenous variables. 1. Extract historical GRP and the forecast columns. 2. OUTPUT: 'data/kbr_grp_forecast.csv'.",
        "priority": "high",
        "status": "TODO",
        "acceptance_criteria": [
            "Forecast horizon extracted (e.g. 2026-2027)",
            "Data aligned with standard monthly grid"
        ]
    },
    {
        "id": 121,
        "title": "Synthesis: The Ultimate Macro Dataset",
        "description": "Combine ALL extracted data (Tasks 114, 115, 118, 119, 120) into a single master dataset. 1. Merge on Date. 2. Handle missing values (interpolate or cut). 3. Select Top-10 features across ALL domains using correlation and Lasso. 4. OUTPUT: 'data/master_macro_dataset.csv'.",
        "priority": "high",
        "status": "TODO",
        "acceptance_criteria": [
            "Master dataset contains > 100 features from all source files",
            "Top-10 selection rationale documented in markdown",
            "Ready for model training"
        ]
    }
]

# Вставляем задачи после 117
idx = next((i for i, t in enumerate(prd['user_stories']) if t['id'] == 117), -1)
if idx != -1:
    prd['user_stories'][idx+1:idx+1] = new_tasks
else:
    prd['user_stories'].extend(new_tasks)

with open(prd_path, 'w') as f:
    json.dump(prd, f, indent=2, ensure_ascii=False)

print("PRD updated with Tasks 118-121 (Labor, PPI, GRP).")