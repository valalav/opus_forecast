#!/usr/bin/env python3
"""
Verification script for HierarchicalMicroForecaster tests.
Run this to verify all tests pass.
"""

import subprocess
import sys
from pathlib import Path


def run_test():
    """Run all HierarchicalMicroForecaster tests."""
    print("=" * 60)
    print("Running HierarchicalMicroForecaster tests...")
    print("=" * 60)

    result = subprocess.run(
        [
            "python3",
            "-m",
            "pytest",
            "tests/test_hierarchical_micro_forecaster.py",
            "-v",
        ],
        cwd=Path(__file__).parent,
        capture_output=False,
    )

    print("=" * 60)
    if result.returncode == 0:
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nTest coverage includes:")
        print("  - Model import and initialization")
        print("  - Parameter validation")
        print("  - Prophet subcomponents configuration")
        print("  - Volatile microcomponents set")
        print("  - Component weights validation")
        print("  - Custom region code, horizon, train_start")
        print("  - Feature creation (basic and extended)")
        print("  - Model fitting with mocked data")
        print("  - Prediction and forecast functionality")
        print("  - Error handling for unfitted model")
        print("  - Detailed forecast output")
        print("  - Coverage report generation")
        print("  - Component aggregation logic")
        print("  - String representation")
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(run_test())
