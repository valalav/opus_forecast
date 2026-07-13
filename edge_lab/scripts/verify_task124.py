#!/usr/bin/env python3

import pandas as pd
from pathlib import Path


def verify_task124():
    """Verify Task 124 completion against acceptance criteria"""

    print("=" * 60)
    print("Task 124 Verification")
    print("=" * 60)

    # Read output CSV
    output_file = Path("data/kbr_high_freq_indicators.csv")
    df = pd.read_csv(output_file)

    all_passed = True

    # Criterion 1: Extracted HH Index and Housing Prices for KBR
    print("\n[1] Extracted HH Index and Housing Prices for KBR")

    # Check HH Index
    has_hhi = "HH_Index" in df.columns
    print(f"  HH Index extracted: {has_hhi}")
    if has_hhi:
        print(f"    - Rows: {len(df)}")
        print(f"    - Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"    - HH Index Average: {df['HH_Index'].mean():.2f}")
    else:
        print("    ✗ FAIL: HH_Index column not found")
        all_passed = False

    # Check Housing Prices
    has_housing_price = "Housing_Price_Monthly_RUB_m2" in df.columns
    has_housing_mom = "Housing_Price_MoM" in df.columns
    print(f"  Housing Prices extracted: {has_housing_price or has_housing_mom}")
    if has_housing_price or has_housing_mom:
        print(f"    - Housing_Price_Monthly_RUB_m2 column: {has_housing_price}")
        print(f"    - Housing_Price_MoM column: {has_housing_mom}")
        if has_housing_price:
            print(
                f"    - Average housing price: {df['Housing_Price_Monthly_RUB_m2'].mean():.0f} RUB/m2"
            )
    else:
        print("    ✗ FAIL: Housing price columns not found")
        all_passed = False

    # Criterion 2: Merged into a single CSV with Date index
    print("\n[2] Merged into a single CSV with Date index")
    has_date = "Date" in df.columns
    file_exists = output_file.exists()
    has_cpi = "CPI_MoM" in df.columns

    print(f"  Output file exists: {file_exists}")
    print(f"  Date column exists: {has_date}")
    print(f"  CPI_MoM column exists: {has_cpi}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    if not (file_exists and has_date):
        all_passed = False

    # Criterion 3: Correlation check against CPI included in report
    print("\n[3] Correlation check against CPI included in report")
    report_file = Path("data/task124_correlation_report.md")
    report_exists = report_file.exists()

    print(f"  Report exists: {report_exists}")

    if report_exists:
        with open(report_file, "r", encoding="utf-8") as f:
            report = f.read()

        has_corr_hh_cpi = "HH_Index_vs_CPI" in report
        has_corr_housing_cpi = "Housing_Price_MoM_vs_CPI" in report
        has_corr_hh_housing = "HH_Index_vs_Housing_Price_MoM" in report

        print(f"  HH Index vs CPI correlation in report: {has_corr_hh_cpi}")
        print(f"  Housing Price vs CPI correlation in report: {has_corr_housing_cpi}")
        print(
            f"  HH Index vs Housing Price correlation in report: {has_corr_hh_housing}"
        )

        if has_cpi and (has_corr_hh_cpi or has_corr_housing_cpi):
            print("\n  Correlation values found:")
            lines = report.split("\n")
            for line in lines:
                if "_vs_" in line and ":" in line:
                    print(f"    {line.strip()}")
        else:
            all_passed = False
    else:
        print("  ✗ FAIL: Report file not found")
        all_passed = False

    # Final summary
    print("\n" + "=" * 60)
    print("Verification Summary:")
    print("=" * 60)
    if all_passed:
        print("  Result: ✓ ALL ACCEPTANCE CRITERIA PASSED")
        print("  HH Index: PASS")
        print("  Housing Prices: PASS (from OPR annual data, interpolated)")
        print("  Merged CSV: PASS")
        print("  Correlation Report: PASS")
    else:
        print("  Result: ✗ SOME ACCEPTANCE CRITERIA FAILED")
        print(f"  HH Index: {'PASS' if has_hhi else 'FAIL'}")
        print(
            f"  Housing Prices: {'PASS' if (has_housing_price or has_housing_mom) else 'FAIL'}"
        )
        print(f"  Merged CSV: {'PASS' if file_exists and has_date else 'FAIL'}")
        print(f"  Correlation Report: {'PASS' if report_exists else 'FAIL'}")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys

    sys.exit(verify_task124())
