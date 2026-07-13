#!/usr/bin/env python3
"""
Verification script for RidgeExtendedForecaster tests
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("=" * 60)
    print("VERIFICATION: RidgeExtendedForecaster Unit Tests")
    print("=" * 60)

    test_file = Path(__file__).parent / "tests" / "test_ridge_extended_forecaster.py"

    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False

    print(f"📁 Test file: {test_file}")
    print()

    print("Running pytest...")
    print("-" * 60)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
        cwd=Path(__file__).parent,
        capture_output=False,
    )

    print("-" * 60)
    print()

    if result.returncode == 0:
        print("✅ ALL TESTS PASSED")
        print()
        print("Test Coverage:")
        print("- Model import and parameters")
        print("- Feature preparation (base and macro)")
        print("- Fit functionality (with/without macro)")
        print("- Predict functionality")
        print("- Forecast functionality")
        print("- Backtest functionality")
        print("- Feature importance extraction")
        print("- Model info retrieval")
        print("- Error handling for edge cases")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
