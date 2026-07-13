#!/usr/bin/env python3
"""
Verification script for Task 253: Reporting: HTML Assembly
Acceptance Criteria:
1. HTML file exists and opens in browser
2. Contains both images and data table
"""

import os
import re
from pathlib import Path


def verify_html_file_exists():
    """Check if HTML file exists"""
    html_path = Path(
        "/home/valalav/_projects/sirena-kbr/edge_lab/assets/reports/model_performance.html"
    )
    return html_path.exists() and html_path.is_file()


def verify_html_content():
    """Check if HTML contains images and data table"""
    html_path = Path(
        "/home/valalav/_projects/sirena-kbr/edge_lab/assets/reports/model_performance.html"
    )

    with open(html_path, "r") as f:
        content = f.read()

    # Check for image tags
    has_mae_image = "../charts/mae_comparison.png" in content
    has_trajectory_image = "../charts/forecast_trajectories.png" in content

    # Check for table structure
    has_table_tag = "<table>" in content
    has_thead_tag = "<thead>" in content
    has_tbody_tag = "<tbody>" in content
    has_table_data = "opr_ridge" in content and "1.1346" in content

    # Check for timestamp and interpretation
    has_timestamp = "2026-01-23" in content or "Generated:" in content
    has_interpretation = "Interpretation" in content or "interpretation" in content

    return {
        "has_mae_image": has_mae_image,
        "has_trajectory_image": has_trajectory_image,
        "has_table_tag": has_table_tag,
        "has_thead_tag": has_thead_tag,
        "has_tbody_tag": has_tbody_tag,
        "has_table_data": has_table_data,
        "has_timestamp": has_timestamp,
        "has_interpretation": has_interpretation,
    }


def main():
    print("=" * 60)
    print("Task 253: Reporting: HTML Assembly - Verification")
    print("=" * 60)

    # Criterion 1: HTML file exists
    print("\n[Criterion 1] HTML file exists and opens in browser")
    if verify_html_file_exists():
        print("✓ PASS: HTML file exists at assets/reports/model_performance.html")
    else:
        print("✗ FAIL: HTML file not found")
        return False

    # Criterion 2: Contains both images and data table
    print("\n[Criterion 2] Contains both images and data table")
    checks = verify_html_content()

    all_passed = True

    # Image checks
    print("\nImage Embedding:")
    if checks["has_mae_image"]:
        print("✓ PASS: MAE comparison image embedded")
    else:
        print("✗ FAIL: MAE comparison image not found")
        all_passed = False

    if checks["has_trajectory_image"]:
        print("✓ PASS: Forecast trajectories image embedded")
    else:
        print("✗ FAIL: Forecast trajectories image not found")
        all_passed = False

    # Table checks
    print("\nData Table:")
    if checks["has_table_tag"] and checks["has_thead_tag"] and checks["has_tbody_tag"]:
        print("✓ PASS: HTML table structure present")
    else:
        print("✗ FAIL: HTML table structure incomplete")
        all_passed = False

    if checks["has_table_data"]:
        print("✓ PASS: Table contains data from consolidated_metrics.csv")
    else:
        print("✗ FAIL: Table data missing")
        all_passed = False

    # Additional content checks
    print("\nAdditional Content:")
    if checks["has_timestamp"]:
        print("✓ PASS: Timestamp included")
    else:
        print("✗ FAIL: Timestamp missing")
        all_passed = False

    if checks["has_interpretation"]:
        print("✓ PASS: Interpretation text included")
    else:
        print("✗ FAIL: Interpretation text missing")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL ACCEPTANCE CRITERIA MET")
        print("=" * 60)
        return True
    else:
        print("✗ SOME ACCEPTANCE CRITERIA NOT MET")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
