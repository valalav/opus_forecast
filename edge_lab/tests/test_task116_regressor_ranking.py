#!/usr/bin/env python3
"""
Test verification for Task 116: Correlation & Regressor Ranking
"""

import pandas as pd
from pathlib import Path


def test_regressor_priority_list():
    """Verify regressor_priority_list.csv meets requirements."""
    path = Path("data/regressor_priority_list.csv")
    assert path.exists(), f"File {path} does not exist"

    df = pd.read_csv(path)

    # Test 1: Priority list contains Top-20 series
    assert len(df) == 20, f"Expected 20 regressors, got {len(df)}"

    # Test 2: All correlations > 0.3
    assert (df["abs_correlation"] > 0.3).all(), (
        "Some regressors have correlation <= 0.3"
    )

    # Test 3: Ranked list includes optimal lag for each regressor
    assert "optimal_lag" in df.columns, "Missing 'optimal_lag' column"
    assert df["optimal_lag"].notna().all(), "Some regressors have missing optimal_lag"

    # Test 4: Ranks are 1-20
    assert list(df["rank"]) == list(range(1, 21)), "Ranks are not 1-20"

    print("✅ Test passed: regressor_priority_list.csv")
    return True


def test_missing_data_report():
    """Verify missing_data_report.csv meets requirements."""
    path = Path("data/missing_data_report.csv")
    assert path.exists(), f"File {path} does not exist"

    df = pd.read_csv(path)

    # Test 1: Report has content
    assert len(df) > 0, "Missing data report is empty"

    # Test 2: All series have missing_pct > 20%
    assert (df["missing_pct"] > 20).all(), (
        "Some rejected series have missing_pct <= 20%"
    )

    # Test 3: Required columns exist
    required_cols = ["feature_id", "missing_pct", "rejection_reason"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    print("✅ Test passed: missing_data_report.csv")
    return True


def main():
    """Run all tests."""
    print("Testing Task 116 Outputs")
    print("=" * 60)

    try:
        test_regressor_priority_list()
        test_missing_data_report()
        print("\n" + "=" * 60)
        print("All tests PASSED ✅")
        return True
    except AssertionError as e:
        print(f"\n❌ Test FAILED: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
