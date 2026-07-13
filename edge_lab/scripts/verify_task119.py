#!/usr/bin/env python3
"""
Verification script for Task 119: Deep Dive - Producer Prices (PPI)
"""

import pandas as pd
from pathlib import Path


def main():
    print("=" * 60)
    print("Task 119 Verification")
    print("=" * 60)

    # Criterion 1: Extracted PPI series for KBR
    print("\nCriterion 1: Extracted PPI series for KBR")
    ppi_file = Path("/home/valalav/_projects/sirena-kbr/edge_lab/data/kbr_ppi_detailed.csv")

    if not ppi_file.exists():
        print(f"  ❌ FAIL: Output file not found: {ppi_file}")
        return False

    df = pd.read_csv(ppi_file)
    print(f"  ✅ File exists: {ppi_file}")
    print(f"  ✅ Total records: {len(df)}")
    print(f"  ✅ Unique indicators: {df['indicator'].nunique()}")

    # Check for manufacturing PPI
    manuf_count = len(df[df["indicator"].str.contains("Manufacturing", na=False)])
    print(f"  ✅ Manufacturing PPI records: {manuf_count}")

    # Check for agricultural PPI (Food production)
    agric_count = len(df[df["indicator"].str.contains("Food production", na=False)])
    print(f"  ✅ Agricultural PPI (Food production) records: {agric_count}")

    # Check for SKFO (KBR proxy)
    skfo_count = len(df[df["code"] == "SKFO"])
    print(f"  ✅ SKFO (KBR proxy) records: {skfo_count}")

    if manuf_count == 0 or agric_count == 0 or skfo_count == 0:
        print("  ❌ FAIL: Missing required PPI series")
        return False

    # Criterion 2: Correlation analysis confirms PPI leads CPI by 1-3 months
    print("\nCriterion 2: Correlation analysis confirms PPI leads CPI by 1-3 months")
    corr_file = Path(
        "/home/valalav/_projects/sirena-kbr/edge_lab/data/ppi_cpi_correlations.csv"
    )

    if not corr_file.exists():
        print(f"  ❌ FAIL: Correlation file not found: {corr_file}")
        return False

    corr_df = pd.read_csv(corr_file)
    print(f"  ✅ Correlation file exists: {corr_file}")
    print(f"  ✅ Total correlation records: {len(corr_df)}")

    # Find best lag for each indicator
    best_idx = corr_df.groupby("indicator")["correlation"].apply(
        lambda x: x.abs().idxmax()
    )
    best_lags = corr_df.loc[best_idx]

    # Check if any indicators show PPI leading CPI by 1-3 months
    leading_indicators = best_lags[
        (best_lags["lag_months"] >= 1) & (best_lags["lag_months"] <= 3)
    ]

    print(f"\n  Indicators with PPI leading CPI by 1-3 months:")
    for _, row in leading_indicators.iterrows():
        indicator = row["indicator"][:40]
        lag = row["lag_months"]
        corr = row["correlation"]
        print(f"    - {indicator}: lag={lag}mo, corr={corr:.4f}")

    if len(leading_indicators) == 0:
        print("  ❌ FAIL: No indicators show PPI leading CPI by 1-3 months")
        return False

    print(
        f"  ✅ PASS: {len(leading_indicators)} indicators confirm PPI leads CPI by 1-3 months"
    )

    print("\n" + "=" * 60)
    print("All acceptance criteria met!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys

    sys.exit(0 if main() else 1)
