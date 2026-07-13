#!/usr/bin/env python3
"""
Task 124: Mining High-Freq Indicators (HH.ru & DomClick)

Extracts HH Index and Housing Prices for KBR, merges into single CSV.
"""

import pandas as pd
import openpyxl
from pathlib import Path
from datetime import datetime


def extract_hh_index():
    """Extract HH Index for KBR from hh_индекс.xlsx."""
    hh_file = Path(
        "/home/valalav/_projects/sirena-kbr/edge_lab/data/raw/opr_stat/hh_индекс.xlsx"
    )

    wb = openpyxl.load_workbook(hh_file, read_only=True, data_only=True)
    ws = wb["Лист1"]

    data = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] and "Кабардин" in str(row[0]):
            data.append(
                {
                    "Date": row[2],
                    "Region": row[0],
                    "Profession": row[1],
                    "HH_Index": row[3],
                }
            )

    if not data:
        print("WARNING: No KBR HH Index data found!")
        return None

    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    df_agg = (
        df.groupby(df["Date"].dt.to_period("M"))
        .agg({"HH_Index": "mean", "Date": "first"})
        .reset_index(drop=True)
    )
    df_agg["Date"] = pd.to_datetime(df_agg["Date"])

    print(f"  Extracted {len(df)} raw HH Index rows for KBR")
    print(f"  Aggregated to {len(df_agg)} monthly observations")

    return df_agg[["Date", "HH_Index"]]


def extract_housing_from_opr():
    """Extract housing prices from OPR annual data."""
    data_dir = Path("/home/valalav/_projects/sirena-kbr/edge_lab/data")
    housing_file = "extracted_kbr/10 цены производителей_10-05 цены на первичном и вторичном рынках жилья.csv"
    file_path = data_dir / housing_file

    if file_path.exists():
        df = pd.read_csv(file_path)
        print(f"  Found housing file: {housing_file}")

        if "region" in df.columns:
            kbr_rows = df[df["region"] == "KBR"].copy()
            if len(kbr_rows) > 0:
                print(f"  Found {len(kbr_rows)} KBR annual housing price rows!")
                kbr_rows["Year"] = (
                    kbr_rows["column"].str.extract(r"(\d{4})").astype(int)
                )
                kbr_rows["Date"] = pd.to_datetime(
                    kbr_rows["Year"].astype(str) + "-12-31"
                )
                result = kbr_rows[["Date", "value"]].copy()
                result.columns = ["Date", "Housing_Price_Annual_RUB_m2"]
                return result

    return None


def interpolate_housing_monthly(df_annual):
    """Interpolate annual housing prices to monthly frequency."""
    df_annual = df_annual.copy()
    df_annual["Year"] = df_annual["Date"].dt.year

    year_to_value = dict(
        zip(df_annual["Year"], df_annual["Housing_Price_Annual_RUB_m2"])
    )

    start_date = "2019-01-01"
    end_date = "2025-12-01"
    monthly_dates = pd.date_range(start=start_date, end=end_date, freq="MS")

    monthly_values = []
    for date in monthly_dates:
        year = date.year
        if year in year_to_value and year + 1 in year_to_value:
            val_this = year_to_value[year]
            val_next = year_to_value[year + 1]
            month_ratio = date.month / 12.0
            interpolated = val_this + (val_next - val_this) * month_ratio
            monthly_values.append(interpolated)
        elif year in year_to_value:
            monthly_values.append(year_to_value[year])
        else:
            monthly_values.append(None)

    df_monthly = pd.DataFrame(
        {"Date": monthly_dates, "Housing_Price_Monthly_RUB_m2": monthly_values}
    )

    df_monthly["Housing_Price_MoM"] = (
        df_monthly["Housing_Price_Monthly_RUB_m2"].pct_change(fill_method=None) * 100
        + 100
    )

    return df_monthly


def load_cpi_data():
    """Load CPI data."""
    cpi_file = Path(
        "/home/valalav/_projects/sirena-kbr/edge_lab/data/enhanced_inflation_data.csv"
    )
    df = pd.read_csv(cpi_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df_cpi = df[["Date", "mom"]].copy()
    df_cpi.columns = ["Date", "CPI_MoM"]
    return df_cpi


def merge_data(df_hh, df_housing, df_cpi):
    """Merge all data sources."""
    result = df_hh.copy()

    result = result.merge(df_housing, on="Date", how="left")
    result = result.merge(df_cpi, on="Date", how="left")

    result = result.sort_values("Date").reset_index(drop=True)
    return result


def calculate_correlations(df):
    """Calculate correlations between indicators."""
    results = {}

    df_clean = df[["HH_Index", "CPI_MoM"]].dropna()
    if len(df_clean) > 2:
        results["HH_Index_vs_CPI"] = df_clean["HH_Index"].corr(df_clean["CPI_MoM"])

    if "Housing_Price_MoM" in df.columns:
        df_clean = df[["Housing_Price_MoM", "CPI_MoM"]].dropna()
        if len(df_clean) > 2:
            results["Housing_Price_MoM_vs_CPI"] = df_clean["Housing_Price_MoM"].corr(
                df_clean["CPI_MoM"]
            )

    if "Housing_Price_MoM" in df.columns:
        df_clean = df[["HH_Index", "Housing_Price_MoM"]].dropna()
        if len(df_clean) > 2:
            results["HH_Index_vs_Housing_Price_MoM"] = df_clean["HH_Index"].corr(
                df_clean["Housing_Price_MoM"]
            )

    return results


def generate_report(corr_results, notes):
    """Generate correlation report."""
    report = []
    report.append("# Task 124 Correlation Report: High-Freq Indicators")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("## Data Sources")
    report.append(
        "1. **HH Index (HH.ru)**: Labor market tension indicator from hh_индекс.xlsx"
    )
    report.append(
        "2. **Housing Prices**: From OPR stats 'цены на первичном и вторичном рынках жилья' (annual, interpolated to monthly)"
    )
    report.append("3. **CPI**: Monthly inflation from enhanced_inflation_data.csv")
    report.append("")
    report.append("## Technical Notes")
    report.append(notes)
    report.append("")
    report.append("## Correlation Results")

    for key, value in corr_results.items():
        report.append(f"- **{key}**: {value:.4f}")

    return "\n".join(report)


def main():
    print("=== Task 124: Mining High-Freq Indicators ===")
    print()

    notes = [
        "**DomClick Data Limitation:**",
        "- The DomClick Excel files (Цены на жилье) use dashboard slicers/filters",
        "- KBR data is present in the file but hidden from openpyxl and pandas",
        "- Attempted solutions:",
        "  * XML structure analysis (found KBR references but no price values)",
        "  * Multiple Excel engine attempts (openpyxl, pandas)",
        "  * Slicer cache inspection (references KBR but no data)",
        "",
        "**Alternative Solution Used:**",
        "- Used OPR annual housing price data file instead",
        "- This file contains KBR housing prices (2016-2025)",
        "- Annual data interpolated to monthly frequency for analysis",
        "- For production use, consider using xlwings (Excel COM) or SberIndex API",
    ]
    notes_text = "\n".join(notes)

    print("Step 1: Extracting HH Index for KBR...")
    df_hh = extract_hh_index()

    if df_hh is None:
        print("ERROR: Could not extract HH Index data!")
        return

    print("Step 2: Extracting annual housing prices from OPR...")
    df_housing_annual = extract_housing_from_opr()

    if df_housing_annual is None:
        print("ERROR: Could not extract housing price data!")
        return

    print("Step 3: Interpolating housing prices to monthly...")
    df_housing_monthly = interpolate_housing_monthly(df_housing_annual)
    print(f"  Generated {len(df_housing_monthly)} monthly data points")

    print("Step 4: Loading CPI data...")
    df_cpi = load_cpi_data()
    print(f"  Loaded {len(df_cpi)} CPI data points")

    print("Step 5: Merging data sources...")
    df_final = merge_data(df_hh, df_housing_monthly, df_cpi)
    print(f"  Final dataset: {len(df_final)} rows")
    print(f"  Columns: {df_final.columns.tolist()}")

    print("Step 6: Calculating correlations...")
    corr_results = calculate_correlations(df_final)
    for key, value in corr_results.items():
        print(f"  {key}: {value:.4f}")

    print("Step 7: Saving output CSV...")
    output_file = Path(
        "/home/valalav/_projects/sirena-kbr/edge_lab/data/kbr_high_freq_indicators.csv"
    )
    df_final.to_csv(output_file, index=False)
    print(f"  Saved to {output_file}")

    print("Step 8: Generating correlation report...")
    report = generate_report(corr_results, notes_text)
    report_file = Path(
        "/home/valalav/_projects/sirena-kbr/edge_lab/data/task124_correlation_report.md"
    )
    with open(report_file, "w") as f:
        f.write(report)
    print(f"  Report saved to {report_file}")

    print("\n=== Task 124 Complete ===")
    print(f"  Output: {output_file}")
    print(f"  Rows: {len(df_final)}")
    print(f"  HH Index extracted: YES")
    print(
        f"  Housing price data included: YES (annual OPR data interpolated to monthly)"
    )


if __name__ == "__main__":
    main()
