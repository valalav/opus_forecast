import json
import os

prd_path = 'edge_lab/tasks/prd.json'

with open(prd_path, 'r') as f:
    prd = json.load(f)

# 1. Remove the old fake Task 114
prd['user_stories'] = [t for t in prd['user_stories'] if t['id'] != 114]

# 2. Add the new rigorous tasks
new_tasks = [
    {
        "id": 114,
        "title": "Deep Dive Phase 1: The Crawler Engine",
        "description": "Develop a robust 'UniversalExcelCrawler' (agents/excel_crawler.py). OBJECTIVE: Do NOT parse data yet. Just map the territory. REQUIREMENTS: 1. Support 'openpyxl' read_only mode for huge files. 2. Recursive directory walking. 3. Open EVERY .xlsx/.xls file in 'assets/charts/ОПР_статистика'. 4. Extract Metadata: Filename, File Size, Sheet Name, Sheet Index. 5. OUTPUT: 'data/audit_file_index.csv'. CRITICAL: The output must contain EVERY sheet from 'Основная статистика ЮГУ.xlsx' (likely >100 sheets).",
        "priority": "high",
        "status": "TODO",
        "acceptance_criteria": [
            "Script agents/excel_crawler.py exists",
            "Output 'data/audit_file_index.csv' contains > 200 rows (sheets)",
            "Successfully lists all sheets in the 150MB file"
        ]
    },
    {
        "id": 115,
        "title": "Deep Dive Phase 2: The Monolith (Main Statistics)",
        "description": "Actually extract data from 'Основная статистика ЮГУ.xlsx' using the index from Task 114. LOGIC: Iterate through the sheet list. For EACH sheet: 1. Detect Date Row (regex for years/months). 2. Find 'Кабардино-Балкарская' row. 3. Extract the time series. 4. Identify the Indicator Name (usually cell A1 or text above the table). OUTPUT: 'data/raw_yugu_dump.csv' (Date, Value, Indicator, Sheet). CONSTRAINT: Use chunking/delays if needed to avoid memory crashes.",
        "priority": "high",
        "status": "TODO",
        "acceptance_criteria": [
            "Output 'data/raw_yugu_dump.csv' exists",
            "File size > 1MB (contains real data)",
            "Coverage: Data extracted from >90% of identified KBR sheets"
        ]
    },
    {
        "id": 116,
        "title": "Deep Dive Phase 3: Sectoral Mining (Economy)",
        "description": "Process subdirectories: 'Бюджеты', 'ВРП', 'жилье', 'Цены производителей'. METHOD: Use the Crawler to iterate files. Extract KBR-specific rows. CHALLENGE: Formats vary (some are cumulative, some monthly). Add logic to detect frequency. OUTPUT: 'data/raw_sectoral_dump.csv'.",
        "priority": "high",
        "status": "TODO",
        "acceptance_criteria": [
            "Extracts data from >80% of files in target folders",
            "Handles different date formats correctly",
            "Output contains Budget and GRP series"
        ]
    },
    {
        "id": 117,
        "title": "Deep Dive Phase 4: Social Mining (Labor & Expectations)",
        "description": "Process subdirectories: 'ЗП_безработица', 'инфляционные ожидания'. FOCUS: These are critical for Phillips Curve modeling. OUTPUT: 'data/raw_social_dump.csv'.",
        "priority": "high",
        "status": "TODO",
        "acceptance_criteria": [
            "Unemployment and Wage data extracted extracted",
            "Inflation expectations extracted (if available for region)"
        ]
    },
    {
        "id": 118,
        "title": "Deep Dive Phase 5: Synthesis & Validation",
        "description": "NOW we build the Data Map. 1. Merge all dumps (115-117) into `data/deep_macro_history.csv`. 2. Calculate completeness (how many non-null months since 2016). 3. Calculate correlation with KBR Inflation (target). 4. GENERATE: `data/opr_data_map_verified.md` listing ONLY the series that actually have data and correlation > 0.1. 5. NO HARDCODING allowed.",
        "priority": "high",
        "status": "TODO",
        "acceptance_criteria": [
            "Final dataset 'data/deep_macro_history.csv' ready for modeling",
            "Report lists Top-20 based on CALCULATED correlation",
            "No fake entries"
        ]
    }
]

# Insert new tasks after Task 113
# Find index of 113
idx = next((i for i, t in enumerate(prd['user_stories']) if t['id'] == 113), -1)
if idx != -1:
    prd['user_stories'][idx+1:idx+1] = new_tasks
else:
    prd['user_stories'].extend(new_tasks)

with open(prd_path, 'w') as f:
    json.dump(prd, f, indent=2, ensure_ascii=False)

print("PRD updated successfully: Task 114 replaced with Deep Dive pipeline (114-118).")