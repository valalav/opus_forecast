"""
Test cache decorator on forecast functions (Task 29)
====================================================

Verifies that the cache decorator works on forecast functions.
"""

import numpy as np
import pandas as pd
import time
from sirena.cache import cache_decorator


@cache_decorator(key_prefix="forecast", ttl=3600)
def expensive_forecast(model_name: str, horizon: int, data: np.ndarray) -> np.ndarray:
    """
    Simulate expensive forecast computation.

    Args:
        model_name: Name of the model
        horizon: Forecast horizon
        data: Input data

    Returns:
        Forecast array
    """
    time.sleep(0.1)  # Simulate computation
    return np.random.randn(horizon) + data.mean()


@cache_decorator(key_prefix="simple", ttl=60)
def simple_computation(x: float) -> float:
    """Simple function for testing."""
    return x * 2


def test_forecast_caching():
    """Test that cache decorator works on forecast function."""
    print("\n" + "=" * 60)
    print("TEST: Cache Decorator on Forecast Functions")
    print("=" * 60)

    test_data = np.random.randn(100)

    print("\n[1/3] First call (uncached)...")
    start = time.time()
    result1 = expensive_forecast("ridge", 12, test_data)
    elapsed1 = time.time() - start
    print(f"  Computed in {elapsed1:.4f}s")
    print(f"  Result shape: {result1.shape}")

    print("\n[2/3] Second call (from cache)...")
    start = time.time()
    result2 = expensive_forecast("ridge", 12, test_data)
    elapsed2 = time.time() - start
    print(f"  Computed in {elapsed2:.4f}s")
    print(f"  Result shape: {result2.shape}")

    print("\n[3/3] Verify results are identical...")
    identical = np.allclose(result1, result2)
    print(f"  Results identical: {identical}")

    speedup = elapsed1 / elapsed2 if elapsed2 > 0 else 0
    improvement = (1 - elapsed2 / elapsed1) * 100 if elapsed1 > 0 else 0
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Improvement: {improvement:.1f}%")

    success = identical and improvement >= 50
    print(f"\n  Result: {'PASS' if success else 'FAIL'}")
    return success


def test_different_params():
    """Test that different parameters create different cache entries."""
    print("\n" + "=" * 60)
    print("TEST: Different Parameters = Different Cache Entries")
    print("=" * 60)

    print("\n[1/2] Call with param x=5...")
    start = time.time()
    r1 = simple_computation(5)
    elapsed1 = time.time() - start
    print(f"  Result: {r1}, Time: {elapsed1:.4f}s")

    print("\n[2/2] Call with param x=10...")
    start = time.time()
    r2 = simple_computation(10)
    elapsed2 = time.time() - start
    print(f"  Result: {r2}, Time: {elapsed2:.4f}s")

    print("\n[3/3] Verify cache works (second call for x=5)...")
    start = time.time()
    r1_cached = simple_computation(5)
    elapsed3 = time.time() - start
    print(f"  Result: {r1_cached}, Time: {elapsed3:.4f}s")

    same_params_same_result = r1 == r1_cached
    different_params_different_result = r1 != r2
    cached_faster = elapsed3 < elapsed1 / 2  # At least 2x faster

    print(f"  Same params -> same result: {same_params_same_result}")
    print(
        f"  Different params -> different result: {different_params_different_result}"
    )
    print(f"  Cached call is faster: {cached_faster}")

    success = (
        same_params_same_result and different_params_different_result and cached_faster
    )
    print(f"\n  Result: {'PASS' if success else 'FAIL'}")
    return success


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CACHE DECORATOR TEST (Task 29 - Criterion 2)")
    print("=" * 60)
    print("\nVerifying: Cache decorator works on forecast functions")

    results = {
        "forecast_caching": test_forecast_caching(),
        "different_params": test_different_params(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {test_name}: {status}")

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    print(f"\n  Passed: {passed}/{total}")

    all_passed = all(results.values())
    if all_passed:
        print("\n  Overall: PASS - Cache decorator works on forecast functions")
        return 0
    else:
        print("\n  Overall: FAIL")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
