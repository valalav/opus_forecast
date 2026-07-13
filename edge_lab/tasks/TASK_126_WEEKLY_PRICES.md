# Task 126: Mining Weekly Prices (2008-2026)

## Context
We have discovered a high-value data asset: `data/raw/11511100200030200004.xlsx`.
This file contains **weekly absolute prices** (in RUB) for consumer goods in KBR, spanning from **January 2008 to January 2026**.

## File Structure Analysis
- **Sheet:** `Отчет`
- **Header:** Row 1 contains date ranges (e.g., `с 07.01.2008 по 14.01.2008`).
- **Column 0:** Product Name (e.g., `Маргарин, кг`).
- **Column 1:** **Product Code** (e.g., `1001.0`, `111.0`). This is critical for reliable mapping.
- **Columns 2+:** Price values (float).

## Objective
Create an ingestion agent `agents/weekly_prices_ingester.py` to parse this file and transform it into a machine-learning friendly format.

## Requirements

### 1. Parsing Logic
- Load the file using `pandas`.
- Extract the **Product Code** from Column 1 as the primary identifier (cast to int/string to remove `.0`).
- Parse the Date columns (Row 1). Convert text ranges (e.g., `с 07.01...`) to a single `datetime` object (use the **end date** of the week).
- Handle `NaN` values (missing prices). Interpolate linearly for gaps < 4 weeks; flag larger gaps.

### 2. Output Format
Generate a "Long Format" CSV: `data/kbr_weekly_prices_2008_2026.csv`.
Columns:
- `date`: YYYY-MM-DD
- `product_code`: Int (e.g., 1001)
- `product_name`: Str
- `price`: Float
- `price_prev_week`: Float (lag 1)
- `wow_growth`: Float % ((price / price_prev - 1) * 100)

### 3. Validation
- **Codes:** Ensure codes match the "Rosstat Standard" (cross-check with `indicator_mapping_registry.csv` if possible, but prioritize raw extraction first).
- **Dates:** Verify strictly weekly frequency (7-day delta).
- **Outliers:** Detect price jumps > 20% in a single week (potential data error or extreme shock).

## Acceptance Criteria
1. Output CSV exists and contains data from 2008 to 2026.
2. Product Codes are preserved and used as keys.
3. Calculated `wow_growth` column exists.
4. Report `weekly_data_quality.md` lists any products with >10% missing data.
