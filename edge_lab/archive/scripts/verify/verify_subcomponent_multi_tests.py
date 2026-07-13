#!/usr/bin/env python3
"""
Verification script for Task 11: Test SubcomponentMultiForecaster

This script verifies that:
1. Test file exists
2. All tests pass
3. Key functionality is tested
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run command and return success status."""
    print(f"\n{description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ FAILED: {description}")
        print(result.stdout)
        print(result.stderr)
        return False
    print(f"✅ PASSED: {description}")
    return True


def main():
    print("=" * 70)
    print("TASK 11: Test SubcomponentMultiForecaster - VERIFICATION")
    print("=" * 70)

    all_checks_passed = True

    # Check 1: Test file exists
    test_file = (
        Path(__file__).parent / "tests" / "test_subcomponent_multi_forecaster.py"
    )
    if test_file.exists():
        print(f"✅ Test file exists: {test_file}")
    else:
        print(f"❌ Test file not found: {test_file}")
        all_checks_passed = False

    # Check 2: Run all SubcomponentMultiForecaster tests
    if not run_command(
        "python3 -m pytest tests/test_subcomponent_multi_forecaster.py -v",
        "Running SubcomponentMultiForecaster tests",
    ):
        all_checks_passed = False

    # Check 3: Run quick test (quiet mode)
    if not run_command(
        "python3 -m pytest tests/test_subcomponent_multi_forecaster.py -q",
        "Running SubcomponentMultiForecaster tests (quiet)",
    ):
        all_checks_passed = False

    # Check 4: Verify no regressions in other tests
    if not run_command(
        "python3 -m pytest tests/ -q --tb=no", "Running all tests (no regressions)"
    ):
        all_checks_passed = False

    print("\n" + "=" * 70)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED")
        print("=" * 70)
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
