#!/usr/bin/env python3
"""
Verification Script for Task 117: Production - Integrate New Regressors into Sirena

This script verifies that Task 117 acceptance criteria are met:
1. Sirena ensemble uses at least 3 new OPR-based features
2. Backtest shows MAE improvement or documented justification why features didn't help
"""

import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.models.registry import registry
from sirena.models.opr_enhanced_ridge import OPREnhancedRidgeForecaster
from sirena.data.enhanced_loader import load_enhanced_data


def verify_criterion_1():
    """
    Criterion 1: Sirena ensemble uses at least 3 new OPR-based features
    """
    print("=" * 70)
    print("CRITERION 1: Sirena ensemble uses at least 3 new OPR-based features")
    print("=" * 70)

    # Check model is registered
    registered_models = registry.list_models()
    print(f"\n[1/3] Model Registration Check")
    print(f"  - Registered models: {', '.join(registered_models)}")
    print(
        f"  - 'opr_ridge' in registry: {'YES' if 'opr_ridge' in registered_models else 'NO'}"
    )

    if "opr_ridge" not in registered_models:
        print(f"  ❌ FAILED: OPR model not registered in Sirena ensemble")
        return False
    else:
        print(f"  ✅ PASSED: OPR model registered in Sirena ensemble")

    # Check model uses OPR features
    print(f"\n[2/3] OPR Features Check")
    df = load_enhanced_data(top_opr_features=3)

    opr_cols_in_data = [c for c in df.columns if c.startswith("opr_")]
    print(f"  - OPR columns in data: {len(opr_cols_in_data)}")
    print(f"  - Features: {', '.join(opr_cols_in_data)}")

    if len(opr_cols_in_data) < 3:
        print(f"  ❌ FAILED: Less than 3 OPR features in data")
        return False
    else:
        print(f"  ✅ PASSED: {len(opr_cols_in_data)} OPR features in data")

    # Check model actually uses OPR features
    print(f"\n[3/3] Model Integration Check")
    model = OPREnhancedRidgeForecaster(alpha=1.0, use_opr=True, opr_lag=1)
    model.fit(df, "mom")

    opr_features_used = [f for f in model._available_features if f.startswith("opr_")]
    print(f"  - OPR features used by model: {len(opr_features_used)}")
    print(f"  - Features: {', '.join(opr_features_used)}")

    if len(opr_features_used) < 3:
        print(f"  ❌ FAILED: Model uses less than 3 OPR features")
        return False
    else:
        print(f"  ✅ PASSED: Model uses {len(opr_features_used)} OPR features")

    print(f"\n✅ CRITERION 1: PASSED")
    return True


def verify_criterion_2():
    """
    Criterion 2: Backtest shows MAE improvement or documented justification
    """
    print("\n" + "=" * 70)
    print("CRITERION 2: Backtest shows MAE improvement or documented justification")
    print("=" * 70)

    report_path = Path("data/task117_report.md")
    results_path = Path("data/task117_results.json")

    print(f"\n[1/2] Checking report file...")
    if not report_path.exists():
        print(f"  ❌ FAILED: Report not found at {report_path}")
        return False

    print(f"  ✅ Report found at {report_path}")
    report_content = report_path.read_text()

    print(f"\n[2/2] Verifying documented justification...")

    # Check for key elements in report
    has_baseline_mae = "Baseline (No OPR Features)" in report_content
    has_enhanced_mae = "Enhanced (With OPR Features)" in report_content
    has_mae_comparison = "MAE:" in report_content
    has_improvement = "Improvement:" in report_content
    has_justification = "Why Features May Not Have Helped" in report_content

    print(f"  - Has baseline MAE: {has_baseline_mae}")
    print(f"  - Has enhanced MAE: {has_enhanced_mae}")
    print(f"  - Has MAE comparison: {has_mae_comparison}")
    print(f"  - Has improvement metric: {has_improvement}")
    print(f"  - Has justification section: {has_justification}")

    if not (
        has_baseline_mae
        and has_enhanced_mae
        and has_mae_comparison
        and has_improvement
        and has_justification
    ):
        print(f"  ❌ FAILED: Report missing required elements")
        return False

    print(f"  ✅ PASSED: Report documents both results and justification")

    # Load and display results
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"\n[Summary]")
        print(f"  - Baseline MAE: {results['baseline_ridge']['MAE']:.4f}")
        print(f"  - Enhanced MAE: {results['enhanced_ridge']['MAE']:.4f}")
        print(f"  - Improvement: {results['ridge_improvement_pct']:.2f}%")
        print(f"  - OPR features: {results['opr_features_used']} features")

    print(f"\n✅ CRITERION 2: PASSED")
    return True


def main():
    """Main verification function."""
    print("\n" + "=" * 70)
    print("TASK 117 VERIFICATION")
    print("=" * 70)
    print("\nVerifying integration of OPR features into Sirena ensemble...")

    criterion1_passed = verify_criterion_1()
    criterion2_passed = verify_criterion_2()

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(
        f"\nCriterion 1 (3+ OPR features in ensemble): {'✅ PASSED' if criterion1_passed else '❌ FAILED'}"
    )
    print(
        f"Criterion 2 (MAE results documented): {'✅ PASSED' if criterion2_passed else '❌ FAILED'}"
    )

    overall_passed = criterion1_passed and criterion2_passed

    print(
        f"\nOverall: {'✅ ALL CRITERIA PASSED' if overall_passed else '❌ SOME CRITERIA FAILED'}"
    )
    print("=" * 70 + "\n")

    return 0 if overall_passed else 1


if __name__ == "__main__":
    sys.exit(main())
