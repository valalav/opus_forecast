"""
Final Verification for Task 29: Performance: Cache
=================================================

Verifies all acceptance criteria:
1. @file: sirena/cache.py exists (>30 lines)
2. @functional: Cache decorator works on forecast functions
3. @metric: Load time reduced by >= 50%
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sirena.cache import cache_decorator, CachedDataLoader


def test_criterion_1_file_exists():
    """Test: sirena/cache.py exists (>30 lines)"""
    print("\n" + "=" * 60)
    print("CRITERION 1: sirena/cache.py exists (>30 lines)")
    print("=" * 60)

    cache_file = Path(__file__).parent / "sirena" / "cache.py"

    if not cache_file.exists():
        print(f"  FAIL: File not found at {cache_file}")
        return False

    line_count = len(cache_file.read_text().splitlines())

    print(f"  File: {cache_file}")
    print(f"  Lines: {line_count}")
    print(f"  Required: >30 lines")
    print(f"  Result: {'PASS' if line_count > 30 else 'FAIL'}")

    return line_count > 30


def test_criterion_2_decorator_works():
    """Test: Cache decorator works on forecast functions"""
    print("\n" + "=" * 60)
    print("CRITERION 2: Cache decorator works on forecast functions")
    print("=" * 60)

    @cache_decorator(key_prefix="test_forecast", ttl=3600)
    def mock_forecast(model: str, horizon: int, data: list) -> list:
        time.sleep(0.05)
        return [x * 1.01 for x in data[:horizon]]

    test_data = list(range(100))

    print("\n[1/2] First call (should compute)...")
    start = time.time()
    result1 = mock_forecast("ridge", 12, test_data)
    elapsed1 = time.time() - start
    print(f"  Time: {elapsed1:.4f}s, Result length: {len(result1)}")

    print("\n[2/2] Second call (should use cache)...")
    start = time.time()
    result2 = mock_forecast("ridge", 12, test_data)
    elapsed2 = time.time() - start
    print(f"  Time: {elapsed2:.4f}s, Result length: {len(result2)}")

    identical = result1 == result2
    improvement = (1 - elapsed2 / elapsed1) * 100 if elapsed1 > 0 else 0

    print(f"\n  Results identical: {identical}")
    print(f"  Speed improvement: {improvement:.1f}%")

    success = identical and improvement > 0
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    return success


def test_criterion_3_performance():
    """Test: Load time reduced by >= 50%"""
    print("\n" + "=" * 60)
    print("CRITERION 3: Load time reduced by >= 50%")
    print("=" * 60)

    loader = CachedDataLoader(use_cache=True, cache_backend="memory")
    loader.clear_cache()

    print("\n[1/2] First load (uncached)...")
    start = time.time()
    data1 = loader.load_monthly_kbr(force_refresh=True)
    elapsed1 = time.time() - start
    print(f"  Time: {elapsed1:.4f}s")

    if data1 is None:
        print("  FAIL: Could not load data")
        return False

    print(f"  Shape: {data1.shape}")

    print("\n[2/2] Second load (from cache)...")
    start = time.time()
    data2 = loader.load_monthly_kbr(force_refresh=False)
    elapsed2 = time.time() - start
    print(f"  Time: {elapsed2:.4f}s")

    improvement = (1 - elapsed2 / elapsed1) * 100 if elapsed1 > 0 else 0
    speedup = elapsed1 / elapsed2 if elapsed2 > 0 else 0

    print(f"\n  Speedup: {speedup:.2f}x")
    print(f"  Improvement: {improvement:.1f}%")
    print(f"  Required: >= 50%")

    success = improvement >= 50
    print(f"  Result: {'PASS' if success else 'FAIL'}")
    return success


def main():
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("TASK 29: PERFORMANCE: CACHE - FINAL VERIFICATION")
    print("=" * 60)

    results = {
        "criterion_1_file_exists": test_criterion_1_file_exists(),
        "criterion_2_decorator_works": test_criterion_2_decorator_works(),
        "criterion_3_performance": test_criterion_3_performance(),
    }

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    for criterion, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {criterion}: {status}")

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    print(f"\n  Passed: {passed}/{total}")

    all_passed = all(results.values())
    if all_passed:
        print("\n" + "=" * 60)
        print("ALL ACCEPTANCE CRITERIA MET")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("SOME ACCEPTANCE CRITERIA NOT MET")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
