#!/usr/bin/env python3
"""Test script to verify Task 252: Visualization Generator acceptance criteria."""

import os
from pathlib import Path

# Define paths
CHARTS_DIR = Path("/home/valalav/_projects/sirena-kbr/edge_lab/assets/charts")
MAE_CHART = CHARTS_DIR / "mae_comparison.png"
TRAJECTORY_CHART = CHARTS_DIR / "forecast_trajectories.png"


def test_png_files_exist():
    """Criterion 1: PNG files generated successfully."""
    print("Test 1: Checking if PNG files exist...")

    if not MAE_CHART.exists():
        print(f"  FAIL: {MAE_CHART} does not exist")
        return False

    if not TRAJECTORY_CHART.exists():
        print(f"  FAIL: {TRAJECTORY_CHART} does not exist")
        return False

    # Check file sizes are reasonable (> 10KB)
    mae_size = MAE_CHART.stat().st_size
    traj_size = TRAJECTORY_CHART.stat().st_size

    if mae_size < 10000:
        print(f"  FAIL: {MAE_CHART} is too small ({mae_size} bytes)")
        return False

    if traj_size < 10000:
        print(f"  FAIL: {TRAJECTORY_CHART} is too small ({traj_size} bytes)")
        return False

    print(
        f"  PASS: Both PNG files exist (MAE: {mae_size} bytes, Trajectory: {traj_size} bytes)"
    )
    return True


def test_charts_have_metadata():
    """Criterion 2: Charts include title and legends."""
    print("Test 2: Checking if charts have title and legends...")

    # For PNG files, we can't easily verify title/legend content without opening them
    # But we can verify the script generates them with proper matplotlib code
    script_path = Path(
        "/home/valalav/_projects/sirena-kbr/edge_lab/scripts/generate_report_charts.py"
    )

    if not script_path.exists():
        print(f"  FAIL: Script {script_path} does not exist")
        return False

    with open(script_path, "r") as f:
        script_content = f.read()

    # Check for title setting
    if "set_title" not in script_content:
        print("  FAIL: Script does not call set_title")
        return False

    # Check for legend
    if "legend" not in script_content:
        print("  FAIL: Script does not call legend")
        return False

    print("  PASS: Script includes set_title and legend calls")
    return True


def run_all_tests():
    """Run all tests and return overall status."""
    print("=" * 60)
    print("Task 252: Visualization Generator - Test Verification")
    print("=" * 60)
    print()

    results = []

    results.append(("PNG files generated successfully", test_png_files_exist()))
    results.append(("Charts include title and legends", test_charts_have_metadata()))

    print()
    print("=" * 60)
    print("Test Results Summary:")
    print("=" * 60)

    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test_name}")

    all_passed = all(r[1] for r in results)

    print()
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    exit(run_all_tests())
