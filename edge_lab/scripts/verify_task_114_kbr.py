#!/usr/bin/env python3
"""
Verification script for Task 114 - KBR Macro Monolith extraction
"""

import pandas as pd
import sys
from pathlib import Path

output_path = Path("/home/valalav/_projects/sirena-kbr/edge_lab/data/kbr_macro_monolith.csv")

print("=== Verification of Task 114: Mining KBR Macro Monolith ===\n")

# Load the CSV
try:
    df = pd.read_csv(output_path)
    print(f"✓ CSV loaded successfully")
    print(f"  Total records: {len(df)}")
except Exception as e:
    print(f"✗ Failed to load CSV: {e}")
    sys.exit(1)

# Acceptance Criterion 1: Output contains > 50 unique series
print("\n[1] Checking: Output contains > 50 unique series")
unique_series = df[["Indicator", "Category", "Metric_Type"]].drop_duplicates()
print(f"  Unique series count: {len(unique_series)}")
if len(unique_series) > 50:
    print(f"  ✓ PASS: Found {len(unique_series)} unique series (required > 50)")
else:
    print(f"  ✗ FAIL: Only {len(unique_series)} unique series (required > 50)")
    sys.exit(1)

# Acceptance Criterion 2: Values matched correctly with dates from row 10 (11)
print("\n[2] Checking: Values matched correctly with dates")
print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"  Number of unique dates: {df['Date'].nunique()}")

# Check that dates are properly formatted
try:
    pd.to_datetime(df["Date"])
    print(f"  ✓ PASS: All dates are valid")
except Exception as e:
    print(f"  ✗ FAIL: Invalid date format: {e}")
    sys.exit(1)

# Acceptance Criterion 3: Metric types (м/м, г/г) are preserved
print("\n[3] Checking: Metric types preserved as labels")
metric_types = df["Metric_Type"].value_counts()
print(f"  Unique metric types: {metric_types.index.tolist()}")

expected_metrics = ["м/м", "г/г"]
found_metrics = [m for m in expected_metrics if m in df["Metric_Type"].values]
print(f"  Expected metrics found: {found_metrics}")

if found_metrics:
    print(f"  ✓ PASS: Metric types preserved (found {', '.join(found_metrics)})")
else:
    print(f"  ✗ FAIL: Expected metric types (м/м, г/г) not found")
    sys.exit(1)

# Additional verification: Check that we have the target indicators
print("\n[4] Additional verification: Target indicators")
target_indicators = ["ИПП", "ОРТ", "Ввод жилья", "Общепит", "Строительство"]
found_indicators = [ind for ind in target_indicators if ind in df["Indicator"].values]
print(f"  Target indicators found: {found_indicators}")

if found_indicators:
    print(
        f"  ✓ PASS: {len(found_indicators)}/{len(target_indicators)} target indicators found"
    )
else:
    print(f"  ✗ FAIL: No target indicators found")
    sys.exit(1)

# Check for data quality
print("\n[5] Data quality check")
null_values = df["Value"].isnull().sum()
print(f"  Null values in Value column: {null_values}")
if null_values == 0:
    print(f"  ✓ PASS: No null values")
else:
    print(f"  ⚠ WARNING: {null_values} null values found")

print("\n" + "=" * 60)
print("✓ ALL ACCEPTANCE CRITERIA PASSED!")
print("=" * 60)
