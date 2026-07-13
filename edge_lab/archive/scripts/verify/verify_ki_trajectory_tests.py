#!/usr/bin/env python3
"""
Verification script for KiTrajectoryForecaster tests.
Run after creating tests to ensure they all pass.
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run pytest on KiTrajectoryForecaster tests."""
    print("=" * 70)
    print("VERIFYING: KiTrajectoryForecaster Tests")
    print("=" * 70)

    test_file = Path(__file__).parent / "tests" / "test_ki_trajectory_forecaster.py"

    if not test_file.exists():
        print(f"❌ FAIL: Test file not found: {test_file}")
        return False

    print(f"Running tests from: {test_file}")
    print()

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    # Check summary line
    summary_lines = result.stdout.split("\n")
    for line in summary_lines:
        if "passed" in line and ("failed" not in line or "0 failed" in line):
            print("=" * 70)
            print("✅ SUCCESS: All tests passed!")
            print("=" * 70)
            return True
        elif "failed" in line:
            print("=" * 70)
            print("❌ FAIL: Some tests failed")
            print("=" * 70)
            return False

    # Check return code
    if result.returncode == 0:
        print("=" * 70)
        print("✅ SUCCESS: All tests passed!")
        print("=" * 70)
        return False

    return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
