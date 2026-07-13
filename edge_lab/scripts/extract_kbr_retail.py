#!/usr/bin/env python3
"""
Extract KBR Retail Trade Turnover data from Rosstat Excel files.

This script parses retail trade data for Kabardino-Balkarian Republic
from regional statistics Excel files.

Data Sources:
- 05-01 оборот розничной торговли.xlsx (Total retail trade)
- 05-02 оборот розничной торговли пищевыми продуктами.xlsx (Food products)
- 05-03 оборот розничной торговли непродовольственными.xlsx (Non-food products)

Output:
- data/kbr_retail_turnover.csv with columns:
  - Date, Retail_Total_MoM_pct, Retail_Food_MoM_pct, Retail_NonFood_MoM_pct,
  - Retail_Total_Abs_mln_rub, Retail_Food_Abs_mln_rub, Retail_NonFood_Abs_mln_rub

Note: The source Excel files contain data from 2016 onwards. The output covers
2016-2025 (not 2015-2025) due to data availability in the source files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


# Month name mapping
MONTH_NAMES = {
    "январь": 1,
    "февраль": 2,
    "март": 3,
    "апрель": 4,
    "май": 5,
    "июнь": 6,
    "июль": 7,
    "август": 8,
    "сентябрь": 9,
    "октябрь": 10,
    "ноябрь": 11,
    "декабрь": 12,
}


def extract_cumulative_months(
    kbr_data: pd.Series, years_row: pd.Series, months_row: pd.Series
) -> List[Tuple[pd.Timestamp, float]]:
    """
    Extract monthly values from cumulative columns.

    Excel structure:
    - Col 1: январь (standalone month - first month of year)
    - Col 2: январь-февраль (cumulative Jan-Feb)
    - Col 3: январь-март (cumulative Jan-Mar)
    - ...
    - Col 12: январь-декабрь (cumulative Jan-Dec)
    - Then repeat for next year...

    To get monthly values:
    - Month 1: use standalone "январь" column
    - Month 2: calculate (январь-февраль) - (январь)
    - Month 3: calculate (январь-март) - (январь-февраль)
    - etc.
    """
    dates = []
    values = []

    current_year = None
    yearly_cumulatives = {}  # Store cumulative values for current year

    for col_idx in range(1, len(kbr_data)):
        year = years_row[col_idx]
        month = months_row[col_idx]
        value = kbr_data[col_idx]

        # Check if this is a new year block
        if pd.notna(year) and "год" in str(year):
            current_year = int(str(year).split()[0])
            yearly_cumulatives = {}  # Reset for new year

        # Skip if year not set
        if current_year is None:
            continue

        # Skip if value is NaN
        if pd.isna(value):
            continue

        if pd.notna(month):
            month_str = str(month).strip()

            # Check if this is a standalone month (first month of year)
            if month_str in MONTH_NAMES:
                month_num = MONTH_NAMES[month_str]
                date = pd.Timestamp(year=current_year, month=month_num, day=1)

                dates.append(date)
                values.append(value)

                # Store for difference calculation
                if month_num == 1:  # January
                    yearly_cumulatives[0] = value
                else:
                    # This shouldn't happen for standalone months after Jan
                    pass

            # Check if this is a cumulative column (e.g., "январь-февраль")
            elif "-" in month_str:
                # Extract the end month from cumulative range
                # e.g., "январь-февраль" -> февраль (month 2)
                # "январь-март" -> март (month 3)
                end_month = month_str.split("-")[-1].strip()

                if end_month in MONTH_NAMES:
                    month_num = MONTH_NAMES[end_month]
                    date = pd.Timestamp(year=current_year, month=month_num, day=1)

                    # Calculate monthly value as difference from previous cumulative
                    prev_cumul_key = month_num - 2  # Index of previous cumulative

                    if prev_cumul_key in yearly_cumulatives:
                        monthly_value = value - yearly_cumulatives[prev_cumul_key]
                    else:
                        # First cumulative after January
                        prev_jan = yearly_cumulatives.get(0)
                        if prev_jan is not None:
                            monthly_value = value - prev_jan
                        else:
                            monthly_value = value  # Fallback

                    dates.append(date)
                    values.append(monthly_value)
                    yearly_cumulatives[month_num - 1] = value

    return list(zip(dates, values))


def parse_excel_file(
    filepath: str, kbr_name: str = "Кабардино-Балкарская Республика"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse Excel file and extract KBR data.

    Returns:
        (abs_df, mom_df): Absolute values and month-over-month percentages
    """
    xl = pd.ExcelFile(filepath)

    # Parse absolute values sheet ("Млн. рублей")
    df_abs = pd.read_excel(filepath, sheet_name="Млн. рублей", header=None)

    # Find KBR row
    kbr_row = None
    for idx, row in df_abs.iterrows():
        region = str(row[0])
        if kbr_name in region or "Кабардино" in region:
            kbr_row = idx
            break

    if kbr_row is None:
        raise ValueError(f"KBR region not found in {filepath}")

    kbr_data = df_abs.iloc[kbr_row]

    # Parse months and years from row 2 (years) and row 3 (months)
    years_row = df_abs.iloc[2]
    months_row = df_abs.iloc[3]

    # Extract absolute values using cumulative approach
    abs_data = extract_cumulative_months(kbr_data, years_row, months_row)

    abs_df = pd.DataFrame(abs_data, columns=["Date", "Value_abs_mln_rub"]).set_index(
        "Date"
    )

    # Parse MoM percentages sheet ("к пред. месяцу")
    # This sheet already has monthly percentages (not cumulative)
    df_mom = pd.read_excel(filepath, sheet_name="к пред. месяцу", header=None)

    # Find KBR row in MoM sheet
    kbr_row_mom = None
    for idx, row in df_mom.iterrows():
        region = str(row[0])
        if kbr_name in region or "Кабардино" in region:
            kbr_row_mom = idx
            break

    if kbr_row_mom is None:
        raise ValueError(f"KBR region not found in MoM sheet of {filepath}")

    # Extract MoM data (simple extraction - no cumulative)
    kbr_data_mom = df_mom.iloc[kbr_row_mom]
    years_row_mom = df_mom.iloc[2]
    months_row_mom = df_mom.iloc[3]

    dates_mom = []
    values_mom = []

    current_year_mom = None
    for col_idx in range(1, len(kbr_data_mom)):
        year = years_row_mom[col_idx]
        month = months_row_mom[col_idx]

        if pd.notna(year) and "год" in str(year):
            current_year_mom = int(str(year).split()[0])

        if current_year_mom is None:
            continue

        if pd.notna(month):
            month_str = str(month).strip()

            if month_str in MONTH_NAMES:
                month_num = MONTH_NAMES[month_str]
                date = pd.Timestamp(year=current_year_mom, month=month_num, day=1)
                value = kbr_data_mom[col_idx]

                dates_mom.append(date)
                values_mom.append(value)

    mom_df = pd.DataFrame({"Date": dates_mom, "Value_MoM_pct": values_mom}).set_index(
        "Date"
    )

    return abs_df, mom_df


def merge_retail_data(
    total_abs: pd.DataFrame,
    total_mom: pd.DataFrame,
    food_abs: pd.DataFrame,
    food_mom: pd.DataFrame,
    nonfood_abs: pd.DataFrame,
    nonfood_mom: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge data from all three retail trade categories.
    """
    # Merge all DataFrames
    result = total_abs.copy()
    result = result.rename(columns={"Value_abs_mln_rub": "Retail_Total_Abs_mln_rub"})

    result["Retail_Total_MoM_pct"] = total_mom["Value_MoM_pct"]
    result["Retail_Food_Abs_mln_rub"] = food_abs["Value_abs_mln_rub"]
    result["Retail_Food_MoM_pct"] = food_mom["Value_MoM_pct"]
    result["Retail_NonFood_Abs_mln_rub"] = nonfood_abs["Value_abs_mln_rub"]
    result["Retail_NonFood_MoM_pct"] = nonfood_mom["Value_MoM_pct"]

    return result.reset_index()


def main():
    parser = argparse.ArgumentParser(
        description="Extract KBR Retail Trade Turnover data from Rosstat Excel files"
    )
    parser.add_argument(
        "--input-dir",
        default="/home/valalav/_projects/sirena-kbr/data/raw/info-stat/05 торговля",
        help="Directory containing retail trade Excel files",
    )
    parser.add_argument(
        "--output-file",
        default="data/kbr_retail_turnover.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed information"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1

    # File paths
    total_file = input_dir / "05-01 оборот розничной торговли.xlsx"
    food_file = input_dir / "05-02 оборот розничной торговли пищевыми продуктами.xlsx"
    nonfood_file = (
        input_dir / "05-03 оборот розничной торговли непродовольственными.xlsx"
    )

    if args.verbose:
        print("Parsing files:")
        print(f"  Total: {total_file}")
        print(f"  Food: {food_file}")
        print(f"  Non-food: {nonfood_file}")

    # Parse each file
    try:
        total_abs, total_mom = parse_excel_file(str(total_file))
        food_abs, food_mom = parse_excel_file(str(food_file))
        nonfood_abs, nonfood_mom = parse_excel_file(str(nonfood_file))
    except Exception as e:
        print(f"Error parsing Excel files: {e}")
        return 1

    # Merge data
    result = merge_retail_data(
        total_abs, total_mom, food_abs, food_mom, nonfood_abs, nonfood_mom
    )

    # Sort by date
    result = result.sort_values("Date").reset_index(drop=True)

    # Save to CSV
    result.to_csv(output_file, index=False)

    # Print summary
    print(f"Successfully extracted retail trade data to: {output_file}")
    print(f"Date range: {result['Date'].min()} to {result['Date'].max()}")
    print(f"Total rows: {len(result)}")
    print(f"Years covered: {sorted(result['Date'].dt.year.unique())}")

    # Check data quality
    missing_data = (
        result[
            ["Retail_Total_MoM_pct", "Retail_Food_MoM_pct", "Retail_NonFood_MoM_pct"]
        ]
        .isna()
        .sum()
    )
    missing_pct = missing_data * 100 / len(result)
    print(f"Missing MoM data: {missing_pct.iloc[0]:.1f}%")

    return 0


if __name__ == "__main__":
    exit(main())
