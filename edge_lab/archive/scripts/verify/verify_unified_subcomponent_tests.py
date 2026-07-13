#!/usr/bin/env python3
"""
Verification script for UnifiedSubcomponentForecaster tests.
Runs tests and prints summary.
"""

import subprocess
import sys


def run_tests():
    """Run pytest and return results."""
    result = subprocess.run(
        [
            "python3",
            "-m",
            "pytest",
            "tests/test_unified_subcomponent_forecaster.py",
            "-v",
        ],
        capture_output=True,
        text=True,
    )
    return result


def main():
    print("=" * 70)
    print("UnifiedSubcomponentForecaster Test Verification")
    print("=" * 70)

    result = run_tests()

    print("\n" + result.stdout)

    if result.returncode != 0:
        print("\n" + "=" * 70)
        print("❌ Tests FAILED")
        print("=" * 70)
        print(result.stderr)
        return 1

    # Count passed tests
    passed = result.stdout.count(" PASSED")
    total = result.stdout.split("collected ")[1].split(" items")[0]

    print("\n" + "=" * 70)
    print(f"✅ ALL TESTS PASSED: {passed}/{total}")
    print("=" * 70)
    print("\nUnifiedSubcomponentForecaster tests verified successfully!")
    print("Test coverage includes:")
    print("  - Model import and initialization")
    print("  - Default and custom parameters")
    print("  - UnifiedForecastResult dataclass")
    print("  - Fit functionality (with/without IRF and Ki)")
    print("  - Baseline forecast")
    print("  - Forecast with rate (Ki change/trajectory)")
    print("  - Scenario forecast (base, hike, cut, custom)")
    print("  - Auto Ki trajectory forecast")
    print("  - Predict method")
    print("  - Get info method")
    print("  - Error handling for not fitted state")

    return 0


if __name__ == "__main__":
    sys.exit(main())
