#!/usr/bin/env python3
"""
Verification script for RidgeShockDummiesForecaster tests
Runs all tests and generates a summary report
"""

import subprocess
import sys
from pathlib import Path


def run_tests():
    """Run pytest for RidgeShockDummiesForecaster tests."""
    test_file = (
        Path(__file__).parent / "tests" / "test_ridge_shock_dummies_forecaster.py"
    )

    if not test_file.exists():
        print(f"ERROR: Test file not found: {test_file}")
        return False

    print("=" * 70)
    print("Running RidgeShockDummiesForecaster Tests")
    print("=" * 70)
    print()

    # Run pytest
    result = subprocess.run(
        ["python3", "-m", "pytest", str(test_file), "-v"],
        capture_output=True,
        text=True,
    )

    # Print output
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    print()
    print("=" * 70)

    # Parse results
    if "passed" in result.stdout:
        # Extract test counts
        lines = result.stdout.split("\n")
        for line in lines:
            if "passed" in line and "==" in line:
                print(f"Result: {line.strip()}")
                break

        if result.returncode == 0:
            print("✅ All tests passed!")
            print("=" * 70)
            return True
        else:
            print("❌ Some tests failed")
            print("=" * 70)
            return False
    else:
        print("❌ No test results found")
        print("=" * 70)
        return False


def print_summary():
    """Print test summary."""
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    print()
    print("File: tests/test_ridge_shock_dummies_forecaster.py")
    print()
    print("Test Coverage: 50 Tests")
    print()
    print("1. Model Import & Parameters (7 tests)")
    print("   - Model import verification")
    print("   - Default parameters (alpha, MIN_TRAIN_SIZE, OUTLIER_YEARS)")
    print("   - Custom parameters (alpha, use_macro, use_2022_dummy)")
    print("   - Base features list validation")
    print("   - Shock dummies list validation")
    print("   - Macro features list validation")
    print("   - ETS weights dictionary")
    print()
    print("2. Fit Functionality (8 tests)")
    print("   - Basic fit without macro")
    print("   - Fit with macro features")
    print("   - Fit without macro features")
    print("   - Fit with 2022 dummy enabled")
    print("   - Fit with 2022 dummy disabled")
    print("   - Fit with custom alpha")
    print("   - Fit with insufficient data (error)")
    print("   - Fit with empty DataFrame (error)")
    print("   - Fit with missing target column (error)")
    print()
    print("3. Predict Functionality (4 tests)")
    print("   - Basic predict functionality")
    print("   - Predict with macro features")
    print("   - Prediction value range validation")
    print("   - Predict without fitting (error)")
    print()
    print("4. Forecast Functionality (3 tests)")
    print("   - Basic forecast (12 months)")
    print("   - Different horizons (1, 6, 12, 24 months)")
    print("   - Forecast without fitting (error)")
    print()
    print("5. Backtest Functionality (4 tests)")
    print("   - Basic backtest")
    print("   - Custom start date")
    print("   - Backtest with macro features")
    print("   - Backtest with shock periods")
    print()
    print("6. Feature Importance (4 tests)")
    print("   - Get feature importance")
    print("   - Importance sorted by absolute coefficient")
    print("   - Feature importance includes shock dummies")
    print("   - Get importance without fitting (error)")
    print()
    print("7. Shock Dummies Specific Tests (2 tests)")
    print("   - Add shock dummies to data")
    print("   - Shock dummy values are correct")
    print()
    print("8. Feature Preparation (4 tests)")
    print("   - Prepare base features")
    print("   - Prepare component lags")
    print("   - Add macro features")
    print("   - Add macro features without macro data")
    print("   - Compute seasonal norm")
    print("   - Compute seasonal norm excludes 2022")
    print()
    print("9. Metrics (2 tests)")
    print("   - Get metrics calculation")
    print("   - Get metrics with empty results")
    print()
    print("10. Edge Cases & Validation (11 tests)")
    print("    - Check fitted validation")
    print("    - Custom alpha parameter")
    print("    - Use macro parameter")
    print("    - Use 2022 dummy parameter")
    print("    - String representation (__repr__)")
    print("    - ETS weight application validation")
    print("    - Outlier years is empty list (key difference!)")
    print("    - Shock dummies in features with use_2022_dummy=True")
    print("    - Only pre-2022 dummies in features without use_2022_dummy")
    print()
    print("=" * 70)
    print()


def main():
    """Main entry point."""
    # Print summary first
    print_summary()

    # Run tests
    success = run_tests()

    if success:
        print("\n✅ RidgeShockDummiesForecaster tests verification PASSED")
        print()
        print("To run tests manually:")
        print("  python3 -m pytest tests/test_ridge_shock_dummies_forecaster.py -v")
        print()
        print("To run a specific test:")
        print(
            "  python3 -m pytest tests/test_ridge_shock_dummies_forecaster.py::TestRidgeShockDummiesForecaster::test_fit_basic -v"
        )
        return 0
    else:
        print("\n❌ RidgeShockDummiesForecaster tests verification FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
