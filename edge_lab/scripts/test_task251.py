#!/usr/bin/env python3
"""
Test verification for Task 251: Metrics Aggregation
===================================================
Verifies:
1. Consolidated CSV contains all models
2. Weighted Score calculated correctly (50% h1 + 30% h2 + 20% h12)
"""

import pandas as pd
import numpy as np
from pathlib import Path


def test_consolidated_file_exists():
    """Test that consolidated_metrics.csv exists."""
    output_path = Path(__file__).parent.parent / "data" / "consolidated_metrics.csv"

    if not output_path.exists():
        print("❌ FAIL: consolidated_metrics.csv does not exist")
        return False

    print(f"✓ PASS: File exists at {output_path}")
    return True


def test_all_models_included():
    """Test that all models are in the consolidated CSV."""
    results_dir = Path(__file__).parent.parent / "archive" / "results"
    output_path = Path(__file__).parent.parent / "data" / "consolidated_metrics.csv"

    consolidated_df = pd.read_csv(output_path)
    models_in_consolidated = set(consolidated_df["Model"].values)

    all_models = set()
    horizons = [1, 2, 12]

    for horizon in horizons:
        metrics_file = results_dir / f"backtest_h{horizon}_metrics.csv"
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            all_models.update(df["model"].values)

    missing_models = all_models - models_in_consolidated

    if missing_models:
        print(f"❌ FAIL: Missing models in consolidated: {missing_models}")
        return False

    print(f"✓ PASS: All models included ({len(models_in_consolidated)} models)")
    return True


def test_weighted_score_calculation():
    """Test that Weighted Score is calculated correctly."""
    output_path = Path(__file__).parent.parent / "data" / "consolidated_metrics.csv"

    df = pd.read_csv(output_path)

    weights = {1: 0.5, 2: 0.3, 12: 0.2}

    all_correct = True

    for idx, row in df.iterrows():
        model = row["Model"]
        expected_score = 0

        for h, weight in weights.items():
            mae_col = f"MAE_h{h}"
            if mae_col in df.columns and pd.notna(row[mae_col]):
                expected_score += row[mae_col] * weight

        actual_score = row["Weighted_Score"]

        if not np.isclose(actual_score, expected_score, rtol=1e-4):
            print(
                f"❌ FAIL: {model} - Expected {expected_score:.4f}, got {actual_score:.4f}"
            )
            all_correct = False
        else:
            print(f"✓ PASS: {model} - Weighted Score = {actual_score:.4f} (correct)")

    return all_correct


def main():
    """Run all tests."""
    print("=" * 60)
    print("Task 251: Metrics Aggregation - Verification Tests")
    print("=" * 60)
    print()

    tests = [
        test_consolidated_file_exists,
        test_all_models_included,
        test_weighted_score_calculation,
    ]

    results = []
    for test in tests:
        print(f"\n{test.__doc__}")
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ FAIL: {test.__name__} raised exception: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    if all(results):
        print("\n✅ All acceptance criteria met!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
