"""
Verification script for HorizonEnsembleForecaster tests
This script validates that all tests pass correctly.
"""

import subprocess
import sys
from pathlib import Path


def run_test(test_file):
    """Run pytest on a specific test file."""
    cmd = [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"]

    result = subprocess.run(cmd, capture_output=True, text=True)

    return result.returncode, result.stdout, result.stderr


def main():
    print("=" * 70)
    print("HORIZON ENSEMBLE FORECASTER TESTS VERIFICATION")
    print("=" * 70)

    test_file = Path(__file__).parent / "tests" / "test_horizon_ensemble_forecaster.py"

    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return 1

    print(f"\n📝 Running tests from: {test_file}")
    print("=" * 70)

    returncode, stdout, stderr = run_test(test_file)

    # Output results
    print(stdout)

    if returncode != 0:
        print("\n" + "=" * 70)
        print("❌ TESTS FAILED")
        print("=" * 70)
        if stderr:
            print("\nERROR OUTPUT:")
            print(stderr)
        return 1

    # Parse results
    lines = stdout.split("\n")
    summary_lines = [l for l in lines if "passed" in l or "failed" in l]

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED")
    print("=" * 70)

    for line in summary_lines:
        if "passed" in line:
            print(line.strip())

    # Check for expected test count
    expected_tests = 30
    test_count_lines = [l for l in lines if f"{expected_tests} passed" in l]

    if test_count_lines:
        print(f"\n📊 Test Coverage: {expected_tests} tests covering:")
        print("   - Model import and parameters")
        print("   - Weight calculation for all horizons (h=1,2,3,6,12)")
        print("   - Weight interpolation for custom horizons")
        print("   - Weight bounds (below min, above max)")
        print("   - Fit functionality with both Huber and Micro models")
        print("   - Predict with ensemble combination")
        print("   - Forecast with trajectory and adaptive weights")
        print("   - Model contributions breakdown")
        print("   - Fallback behavior when models fail")
        print("   - Weight consistency (Huber decreases, Micro increases)")

    print("\n" + "=" * 70)
    print("TASK 14: Test HorizonEnsembleForecaster - ✅ COMPLETED")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
