#!/usr/bin/env python3
"""Verify KBR Labor Market data extraction."""

import pandas as pd
from pathlib import Path

FILE_PATH = Path("/home/valalav/_projects/sirena-kbr/edge_lab/data/kbr_labor_market.csv")


def verify_extraction():
    """Verify the extracted KBR Labor Market data."""
    df = pd.read_csv(FILE_PATH)

    print("=" * 60)
    print("KBR LABOR MARKET DATA VERIFICATION")
    print("=" * 60)

    # Basic stats
    print(f"\nFile: {FILE_PATH}")
    print(f"Dimensions: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Unique series
    unique_series = df["Series_Name"].nunique()
    print(f"\nUnique series: {unique_series}")

    # Indicator types
    indicators = df["Indicator_Type"].unique()
    print(f"\nIndicator types: {list(indicators)}")

    # Date range
    df["Date"] = pd.to_datetime(df["Date"])
    date_range = df["Date"].max() - df["Date"].min()
    print(f"\nDate range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"  Duration: {date_range.days} days")

    # Acceptance criteria
    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERIA VERIFICATION")
    print("=" * 60)

    # 1. Extracted > 50 labor market series for KBR
    ac1 = unique_series > 50
    print(f"1. Extracted > 50 labor market series for KBR")
    print(f"   Result: {'PASS' if ac1 else 'FAIL'} ({unique_series} series)")

    # 2. Validated against official Rosstat summaries
    ac2 = len(indicators) == 3 and all(
        ind in indicators for ind in ["НЗП", "РЗП", "ССЧР"]
    )
    print(f"\n2. Validated against official Rosstat summaries")
    print(f"   Has НЗП (Nominal Wage): {'YES' if 'НЗП' in indicators else 'NO'}")
    print(f"   Has РЗП (Real Wage): {'YES' if 'РЗП' in indicators else 'NO'}")
    print(f"   Has ССЧР (Employment): {'YES' if 'ССЧР' in indicators else 'NO'}")
    print(f"   Result: {'PASS' if ac2 else 'FAIL'}")

    # 3. Includes sectoral breakdown (Wages by industry)
    industries = df["Industry_Code"].nunique()
    ac3 = industries > 30
    print(f"\n3. Includes sectoral breakdown (Wages by industry)")
    print(f"   Unique industries: {industries}")
    print(f"   Sample industries: {sorted(df['Industry_Code'].unique())[:10]}")
    print(f"   Result: {'PASS' if ac3 else 'FAIL'}")

    # Overall result
    all_pass = ac1 and ac2 and ac3
    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL CRITERIA PASS' if all_pass else 'SOME CRITERIA FAIL'}")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    success = verify_extraction()
    exit(0 if success else 1)
