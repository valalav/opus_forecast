#!/usr/bin/env python3
"""Verification script for Task 253: HTML Assembly Report."""

import os
import re


def verify_html_report():
    """Verify HTML report meets acceptance criteria."""
    base_dir = "/home/valalav/_projects/sirena-kbr/edge_lab"
    html_path = os.path.join(base_dir, "assets/reports/model_performance.html")
    csv_path = os.path.join(base_dir, "data/consolidated_metrics.csv")
    mae_chart = os.path.join(base_dir, "assets/charts/mae_comparison.png")
    traj_chart = os.path.join(base_dir, "assets/charts/forecast_trajectories.png")

    print("=" * 60)
    print("VERIFICATION: Task 253 - Reporting: HTML Assembly")
    print("=" * 60)

    # Criterion 1: HTML file exists
    criterion_1 = os.path.exists(html_path)
    print(f"\n1. HTML file exists: {criterion_1}")
    if not criterion_1:
        print("   ❌ FAILED: HTML file does not exist")
        return False
    print(f"   ✅ PASS: {html_path}")

    # Criterion 2: Contains images
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    has_mae_img = "../charts/mae_comparison.png" in html_content
    has_traj_img = "../charts/forecast_trajectories.png" in html_content

    criterion_2a = has_mae_img
    criterion_2b = has_traj_img

    print(f"\n2a. Contains MAE comparison image: {criterion_2a}")
    if criterion_2a:
        print("   ✅ PASS: mae_comparison.png embedded")
    else:
        print("   ❌ FAILED: mae_comparison.png not found")

    print(f"\n2b. Contains forecast trajectory image: {criterion_2b}")
    if criterion_2b:
        print("   ✅ PASS: forecast_trajectories.png embedded")
    else:
        print("   ❌ FAILED: forecast_trajectories.png not found")

    # Criterion 3: Contains data table
    has_table_start = "<table>" in html_content
    has_table_end = "</table>" in html_content
    has_thead = "<thead>" in html_content
    has_tbody = "<tbody>" in html_content

    criterion_3a = has_table_start and has_table_end
    criterion_3b = has_thead and has_tbody

    print(f"\n3a. Contains HTML table structure: {criterion_3a}")
    if criterion_3a:
        print("   ✅ PASS: <table> tags found")
    else:
        print("   ❌ FAILED: Table structure not found")

    print(f"\n3b. Contains table header and body: {criterion_3b}")
    if criterion_3b:
        print("   ✅ PASS: <thead> and <tbody> found")
    else:
        print("   ❌ FAILED: Table header/body not found")

    # Criterion 4: Has timestamp
    has_timestamp = re.search(
        r"Generated:\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", html_content
    )
    criterion_4 = has_timestamp is not None

    print(f"\n4. Contains current timestamp: {criterion_4}")
    if criterion_4:
        print(f"   ✅ PASS: {has_timestamp.group()}")
    else:
        print("   ❌ FAILED: Timestamp not found or invalid format")

    # Criterion 5: Has interpretation text
    has_interpretation = "Interpretation" in html_content
    has_executive_summary = "Executive Summary" in html_content
    has_weighted_score_explanation = "Weighted Score Calculation" in html_content

    criterion_5 = (
        has_interpretation and has_executive_summary and has_weighted_score_explanation
    )

    print(f"\n5. Contains interpretation text: {criterion_5}")
    if criterion_5:
        print("   ✅ PASS: Interpretation section found")
    else:
        print("   ❌ FAILED: Interpretation text missing")

    # Criterion 6: Check PNG files exist
    pngs_exist = os.path.exists(mae_chart) and os.path.exists(traj_chart)
    criterion_6 = pngs_exist

    print(f"\n6. PNG chart files exist: {criterion_6}")
    if criterion_6:
        print(f"   ✅ PASS: Both PNG files exist")
        print(f"      - {mae_chart} ({os.path.getsize(mae_chart)} bytes)")
        print(f"      - {traj_chart} ({os.path.getsize(traj_chart)} bytes)")
    else:
        print("   ❌ FAILED: PNG files missing")

    # Final verdict
    print("\n" + "=" * 60)
    all_passed = (
        criterion_1
        and criterion_2a
        and criterion_2b
        and criterion_3a
        and criterion_3b
        and criterion_4
        and criterion_5
        and criterion_6
    )

    if all_passed:
        print("✅ ALL CRITERIA PASSED - Task 253 Complete")
        print("=" * 60)
        return True
    else:
        print("❌ SOME CRITERIA FAILED - Review above")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = verify_html_report()
    exit(0 if success else 1)
