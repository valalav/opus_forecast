#!/usr/bin/env python3
"""
Verification script for MicrocomponentForecaster tests.
"""

import subprocess
import sys


def run_tests():
    """Run the MicrocomponentForecaster tests."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_microcomponent_forecaster.py",
            "-v",
        ],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    print(result.stderr)

    if result.returncode == 0:
        print("\n✅ All MicrocomponentForecaster tests passed!")
        return True
    else:
        print("\n❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
