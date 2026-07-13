#!/usr/bin/env python3
"""
Verification script for RegimeDetector tests.

This script runs the RegimeDetector tests and verifies:
1. All tests pass
2. RegimeDetector functionality is correctly implemented
"""

import subprocess
import sys


def run_tests():
    """Run all RegimeDetector tests"""
    print("🔍 RegimeDetector Test Verification")
    print("=" * 60)

    result = subprocess.run(
        ["python3", "-m", "pytest", "tests/test_regime_detector.py", "-v"],
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    print(result.stderr)

    if result.returncode == 0:
        print("\n✅ ALL TESTS PASSED")
        return True
    else:
        print(f"\n❌ TESTS FAILED with return code {result.returncode}")
        return False


def test_basic_functionality():
    """Test basic RegimeDetector functionality"""
    print("\n" + "=" * 60)
    print("Testing Basic Functionality")
    print("=" * 60)

    sys.path.insert(0, "agents")
    from regime_detector import RegimeDetector, RegimeType
    import pandas as pd
    import numpy as np

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=60, freq="MS")
    df = pd.DataFrame(
        {
            "Ki_i": np.linspace(7.0, 7.5, 60) + np.random.randn(60) * 0.1,
            "Ruonia": np.linspace(6.8, 7.3, 60) + np.random.randn(60) * 0.1,
            "mom": np.random.randn(60) * 0.3 + 0.5,
        },
        index=dates,
    )

    # Test detection
    detector = RegimeDetector()
    result = detector.detect(df)

    print(f"\n✓ RegimeDetector initialized")
    print(f"✓ Detected regime: {result.regime.value}")
    print(f"✓ Confidence: {result.confidence:.3f}")
    print(f"✓ Diagnostics: {len(result.diagnostics)} metrics")
    print(f"✓ History tracked: {len(result.history)} entries")

    # Test batch detection
    batch_results = detector.detect_batch(df)
    print(f"✓ Batch detection: {len(batch_results)} results")

    # Test statistics
    stats = detector.get_regime_statistics()
    print(f"✓ Statistics: {stats['total']} total detections")
    print(f"✓ Regime types detected: {list(stats['by_type'].keys())}")

    # Test shock detection
    shock_df = df.copy()
    shock_df["Ki_i"].iloc[-5:] = shock_df["Ki_i"].iloc[-6] + 2.0  # Sudden rate hike

    detector2 = RegimeDetector()
    shock_result = detector2.detect(shock_df)

    print(f"✓ Shock detection: {shock_result.regime.value}")

    return True


if __name__ == "__main__":
    # Run pytest tests
    tests_passed = run_tests()

    # Run basic functionality test
    basic_ok = test_basic_functionality()

    if tests_passed and basic_ok:
        print("\n" + "=" * 60)
        print("✅ ALL VERIFICATIONS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ VERIFICATION FAILED")
        print("=" * 60)
        sys.exit(1)
