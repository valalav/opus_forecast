#!/usr/bin/env python3
"""
Verification script for ScenarioRateModel tests

This script verifies that all tests for the ScenarioRateModel pass.
"""

import subprocess
import sys


def main():
    print("=" * 60)
    print("Verifying ScenarioRateModel tests...")
    print("=" * 60)

    # Run pytest on the test file
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_scenario_rate_model.py",
            "-v",
            "--tb=short",
        ],
        capture_output=True,
        text=True,
    )

    # Print output
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Check results
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)

        # Count tests
        output = result.stdout
        if "passed" in output:
            # Extract number of passed tests
            parts = output.split()
            for i, part in enumerate(parts):
                if part == "passed":
                    count = parts[i - 1]
                    print(f"Total tests passed: {count}")
                    break

        return 0
    else:
        print("\n" + "=" * 60)
        print("❌ TESTS FAILED!")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
