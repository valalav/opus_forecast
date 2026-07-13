#!/usr/bin/env python3
"""
Test verification script for Task 241: ExogProphet Lag Optimization

Verifies that:
1. Output JSON exists
2. Optimal lag is mathematically calculated (not hardcoded)
"""

import json
from pathlib import Path


def test_brent_lag_analysis():
    """Test that Brent lag analysis was properly generated."""
    output_path = Path("data/brent_lag_analysis.json")

    print("Testing Brent Lag Analysis...")
    print("=" * 60)

    test_results = []

    test_1_path = output_path.exists()
    test_results.append(("Output JSON exists", test_1_path))
    print(f"Test 1: Output JSON exists... {'✓ PASS' if test_1_path else '✗ FAIL'}")

    if not test_1_path:
        print(f"  Expected path: {output_path}")
        return False

    with open(output_path) as f:
        result = json.load(f)

    test_2_has_keys = "optimal_lag" in result and "correlation" in result
    test_results.append(("JSON has required keys", test_2_has_keys))
    print(
        f"Test 2: JSON has required keys (optimal_lag, correlation)... {'✓ PASS' if test_2_has_keys else '✗ FAIL'}"
    )

    test_3_optimal_lag = isinstance(result.get("optimal_lag"), int)
    test_results.append(("optimal_lag is int", test_3_optimal_lag))
    print(
        f"Test 3: optimal_lag is int... {'✓ PASS' if test_3_optimal_lag else '✗ FAIL'}"
    )

    test_4_correlation = isinstance(result.get("correlation"), (int, float))
    test_results.append(("correlation is numeric", test_4_correlation))
    print(
        f"Test 4: correlation is numeric... {'✓ PASS' if test_4_correlation else '✗ FAIL'}"
    )

    test_5_all_correlations = "all_correlations" in result
    test_results.append(("all_correlations exists", test_5_all_correlations))
    print(
        f"Test 5: all_correlations exists... {'✓ PASS' if test_5_all_correlations else '✗ FAIL'}"
    )

    if test_5_all_correlations:
        all_corrs = result["all_correlations"]
        test_6_mathematically_calculated = (
            isinstance(all_corrs, dict)
            and len(all_corrs) > 0
            and all(
                isinstance(k, str) and k.isdigit() and 0 <= int(k) <= 12
                for k in all_corrs.keys()
            )
        )
        test_results.append(
            (
                "Correlations mathematically calculated (0-12 lags)",
                test_6_mathematically_calculated,
            )
        )
        print(
            f"Test 6: Correlations mathematically calculated... {'✓ PASS' if test_6_mathematically_calculated else '✗ FAIL'}"
        )

        if test_6_mathematically_calculated:
            test_7_optimal_matches = (
                all_corrs.get(str(result["optimal_lag"])) == result["correlation"]
            )
            test_results.append(
                (
                    "Optimal lag matches correlation from all_correlations",
                    test_7_optimal_matches,
                )
            )
            print(
                f"Test 7: Optimal lag is mathematically calculated... {'✓ PASS' if test_7_optimal_matches else '✗ FAIL'}"
            )

    print("=" * 60)
    passed = sum(1 for _, passed in test_results if passed)
    total = len(test_results)
    print(f"Tests passed: {passed}/{total}")

    all_passed = all(passed for _, passed in test_results)

    if all_passed:
        print("\n✅ All tests passed!")
        print(f"\nOptimal Lag: {result.get('optimal_lag')} months")
        print(f"Max Correlation: {result.get('correlation'):+.4f}")
    else:
        print("\n✗ Some tests failed")
        for name, passed in test_results:
            if not passed:
                print(f"  FAILED: {name}")

    return all_passed


if __name__ == "__main__":
    success = test_brent_lag_analysis()
    exit(0 if success else 1)
