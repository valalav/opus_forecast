import json

prd_path = 'edge_lab/tasks/prd.json'

with open(prd_path, 'r') as f:
    prd = json.load(f)

# Удаляем старые версии 114-118, если они были
prd['user_stories'] = [t for t in prd['user_stories'] if t['id'] < 114 or t['id'] > 118]

# Новые высокоточные задачи
new_tasks = [
    {
        "id": 114,
        "title": "Mining: KBR Macro Monolith (Sheet 010)",
        "description": "Extract all macro-economic series from Sheet '010' of 'Основная статистика ЮГУ.xlsx'. REQUIREMENTS: 1. Verify region 'Кбр' at cell (6, 1). 2. Iterate rows from 11 to EOF. 3. Use Column 3 as 'Metric Type' (м/м, г/г, БИ, 3mma). 4. Map data from row 10 (Dates) to values in each row. OUTPUT: 'data/kbr_macro_monolith.csv' with columns: Date, Indicator, Category, Metric_Type, Value. GOAL: Capture GRP, Industry, Retail, and Construction aggregates.",
        "priority": "high",
        "status": "TODO",
        "acceptance_criteria": [
            "Output 'data/kbr_macro_monolith.csv' contains > 50 unique series",
            "Values matched correctly with dates from row 10",
            "Metric types (м/м, г/г) are preserved as separate columns or labels"
        ]
    },
    {
        "id": 115,
        "title": "Mining: Deep Sectoral Blocks (Sheets 100-143)",
        "description": "Parse thematic sheets (100+) using 'Block-State' logic. LOGIC: 1. Maintain a variable 'current_indicator_name'. 2. Scan Column 0. If a cell contains a name (e.g., '1; Все товары', 'Мясо'), update 'current_indicator_name'. 3. If Column 0 matches exactly 'Кбр', extract the row data for the CURRENT indicator. 4. Dates are in Row 10, Columns 1+. OUTPUT: 'data/kbr_sectoral_details.csv'. CRITICAL: Must process ALL 43+ thematic sheets.",
        "priority": "high",
        "status": "TODO",
        "acceptance_criteria": [
            "Parsed > 40 sheets successfully",
            "Output contains detailed CPI components (Meat, Milk, etc.) specifically for KBR",
            "No data misalignment (Indicator name matches the region row)"
        ]
    },
    {
        "id": 116,
        "title": "Intelligence: Correlation & Regressor Ranking",
        "description": "Analyze the new 500+ series to find 'Gold Regressors'. 1. Merge 'kbr_macro_monolith.csv' and 'kbr_sectoral_details.csv' with Target Inflation (inflation_data.csv). 2. Calculate Pearson correlation for all lags (0-6 months). 3. Filter out series with >20% missing data. 4. OUTPUT: 'data/regressor_priority_list.csv' ranked by Correlation Score.",
        "priority": "high",
        "status": "TODO",
        "acceptance_criteria": [
            "Priority list contains Top-20 series with Correlation > 0.3",
            "Missing data report generated for all rejected series",
            "Ranked list includes optimal lag for each regressor"
        ]
    },
    {
        "id": 117,
        "title": "Production: Integrate New Regressors into Sirena",
        "description": "Upgrade existing models (RidgeMacro, Huber) using the Top-5 regressors found in Task 116. 1. Update data loaders to include new OPR features. 2. Retrain models and compare MAE vs Baseline (0.236). 3. Goal: Achieve MAE < 0.22 through 'Gold Regressors'.",
        "priority": "high",
        "status": "TODO",
        "acceptance_criteria": [
            "Sirena ensemble uses at least 3 new OPR-based features",
            "Backtest shows MAE improvement or documented justification why features didn't help"
        ]
    }
]

# Вставляем задачи после 113
idx = next((i for i, t in enumerate(prd['user_stories']) if t['id'] == 113), -1)
if idx != -1:
    prd['user_stories'][idx+1:idx+1] = new_tasks
else:
    prd['user_stories'].extend(new_tasks)

with open(prd_path, 'w') as f:
    json.dump(prd, f, indent=2, ensure_ascii=False)

print("PRD updated successfully.")