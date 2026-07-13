#!/usr/bin/env python3
"""
Test verification script for Task 111: Rosstat Multi-Regional Hierarchy & Correlation

Acceptance Criteria:
1. Ingester extracts RF and SKFO data for >90% of files
2. Correlation matrix generated for top 5 indicators
3. Lead/Lag relationship identified (if exists)
"""

import os
import sys
import pandas as pd
import json
from pathlib import Path


def test_ingester_extraction_rate():
    """Test 1: Verify RF and SKFO data extraction >90%"""
    print("TEST 1: Ingester extracts RF and SKFO data for >90% of files")
    print("-" * 60)

    schema_path = "/home/valalav/_projects/sirena-kbr/edge_lab/data/schema_registry.json"

    if not os.path.exists(schema_path):
        print("❌ FAIL: schema_registry.json not found")
        return False

    with open(schema_path, "r") as f:
        registry = json.load(f)

    total_files = len(registry)
    rf_found = sum(1 for v in registry.values() if v.get("rf_found", False))
    skfo_found = sum(1 for v in registry.values() if v.get("skfo_found", False))

    rf_pct = (rf_found / total_files) * 100 if total_files > 0 else 0
    skfo_pct = (skfo_found / total_files) * 100 if total_files > 0 else 0

    print(f"  Total files: {total_files}")
    print(f"  RF found: {rf_found} ({rf_pct:.1f}%)")
    print(f"  SKFO found: {skfo_found} ({skfo_pct:.1f}%)")

    passed = rf_pct > 90 and skfo_pct > 90

    if passed:
        print(f"  ✅ PASS: Both RF and SKFO extraction > 90%")
    else:
        print(f"  ❌ FAIL: Extraction rate below 90%")

    return passed


def test_regional_hierarchy_data_exists():
    """Test: Verify regional_hierarchy_data.csv exists"""
    print("\nTEST 1b: Verify regional_hierarchy_data.csv exists")
    print("-" * 60)

    data_path = "/home/valalav/_projects/sirena-kbr/edge_lab/data/regional_hierarchy_data.csv"

    if not os.path.exists(data_path):
        print("❌ FAIL: regional_hierarchy_data.csv not found")
        return False

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    regions = df["region"].unique() if "region" in df.columns else []

    print(f"  Total rows: {len(df)}")
    print(f"  Regions found: {list(regions)}")

    has_rf = "RF" in regions
    has_skfo = "SKFO" in regions
    has_kbr = "KBR" in regions

    passed = has_rf and has_skfo and has_kbr

    if passed:
        print(f"  ✅ PASS: All three regions (RF, SKFO, KBR) present")
    else:
        print(f"  ❌ FAIL: Missing regions")

    return passed


def test_correlation_matrix():
    """Test 2: Verify correlation matrix generated for top 5 indicators"""
    print("\nTEST 2: Correlation matrix generated for top 5 indicators")
    print("-" * 60)

    report_path = "/home/valalav/_projects/sirena-kbr/edge_lab/data/correlation_report.md"

    if not os.path.exists(report_path):
        print("❌ FAIL: correlation_report.md not found")
        return False

    with open(report_path, "r") as f:
        content = f.read()

    # Check if the report contains indicator analysis
    if "01-01 индекс промышленного производства" in content:
        print("  ✅ Industrial Production indicator found")
        has_production = True
    else:
        print("  ❌ Industrial Production indicator missing")
        has_production = False

    if "09-01 цены на товары и услуги" in content:
        print("  ✅ Consumer Prices indicator found")
        has_cpi = True
    else:
        print("  ❌ Consumer Prices indicator missing")
        has_cpi = False

    # Count indicators analyzed
    indicators_analyzed = content.count("## Key Findings") > 0

    # Check correlation matrix table
    table_rows = content.count("| ") // 4
    print(f"  Correlation table entries: {table_rows}")

    passed = has_production and has_cpi and table_rows >= 15

    if passed:
        print(f"  ✅ PASS: Correlation matrix generated with top indicators")
    else:
        print(f"  ❌ FAIL: Correlation matrix incomplete")

    return passed


def test_lead_lag_relationships():
    """Test 3: Verify lead/lag relationships are identified"""
    print("\nTEST 3: Lead/Lag relationship identified")
    print("-" * 60)

    report_path = "/home/valalav/_projects/sirena-kbr/edge_lab/data/correlation_report.md"

    if not os.path.exists(report_path):
        print("❌ FAIL: correlation_report.md not found")
        return False

    with open(report_path, "r") as f:
        content = f.read()

    # Check if lead-lag section exists
    has_lead_lag_section = "## Lead-Lag Analysis" in content

    # Check if any relationships are found
    has_rf_kbr = "RF-KBR" in content
    has_skfo_kbr = "SKFO-KBR" in content

    # Check for lead/lag interpretation
    has_leads = "leads" in content or "leading" in content
    has_synchronous = "synchronous" in content or "simultaneous" in content

    print(f"  Lead-Lag section: {has_lead_lag_section}")
    print(f"  RF-KBR pair: {has_rf_kbr}")
    print(f"  SKFO-KBR pair: {has_skfo_kbr}")
    print(f"  Lead/Lag interpretation: {has_leads}")
    print(f"  Synchronous relationships: {has_synchronous}")

    passed = has_lead_lag_section and (has_leads or has_synchronous)

    if passed:
        print(f"  ✅ PASS: Lead/Lag relationships identified")
    else:
        print(f"  ❌ FAIL: Lead/Lag analysis incomplete")

    return passed


def run_all_tests():
    """Run all test verification"""
    print("=" * 60)
    print("TASK 111 VERIFICATION TESTS")
    print("=" * 60)

    results = []

    # Test 1a: RF and SKFO extraction rate
    results.append(
        ("Ingester RF/SKFO extraction rate", test_ingester_extraction_rate())
    )

    # Test 1b: Regional hierarchy data
    results.append(
        ("Regional hierarchy data file", test_regional_hierarchy_data_exists())
    )

    # Test 2: Correlation matrix
    results.append(
        ("Correlation matrix for top 5 indicators", test_correlation_matrix())
    )

    # Test 3: Lead/Lag relationships
    results.append(
        ("Lead/Lag relationship identification", test_lead_lag_relationships())
    )

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(r[1] for r in results)

    print("=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - ACCEPTANCE CRITERIA MET")
        return 0
    else:
        print("❌ SOME TESTS FAILED - ACCEPTANCE CRITERIA NOT MET")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
