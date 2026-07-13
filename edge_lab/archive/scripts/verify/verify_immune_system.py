"""
Verification script for Immune System: Adversarial Stress Testing Agent

Demonstrates that the immune system achieves Survival Rate > 90% as required by task 104.
"""

import sys
from pathlib import Path

# Add agents directory to path
sys.path.insert(0, str(Path(".").resolve() / "agents"))

import numpy as np
import pandas as pd
from immune_system import ImmuneSystemTester, create_sample_model


def verify_immune_system():
    """Verify immune system meets acceptance criteria: Survival Rate > 90%"""

    print("=" * 70)
    print("🛡️ IMMUNE SYSTEM: ADVERSARIAL STRESS TESTING VERIFICATION")
    print("=" * 70)
    print()

    # Create test data simulating inflation time series
    np.random.seed(42)
    dates = pd.date_range(start="2018-01-01", periods=72, freq="MS")

    # Simulate realistic inflation data with:
    # - Mean around 0.5% MoM
    # - Seasonality (higher in Q1)
    # - Random noise
    t = np.arange(len(dates))
    seasonal = 0.2 * np.sin(2 * np.pi * t / 12)  # Annual seasonality
    noise = np.random.randn(len(dates)) * 0.15
    trend = 0.01 * t / len(t)  # Slight upward trend

    target = 0.5 + seasonal + noise + trend

    df = pd.DataFrame(
        {
            "target": target,
            "mom": target,  # MoM inflation
        },
        index=dates,
    )

    print(f"📊 Test Data:")
    print(f"   - Period: {dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')}")
    print(f"   - Observations: {len(df)}")
    print(f"   - Mean: {df['target'].mean():.3f}% MoM")
    print(f"   - Std: {df['target'].std():.3f}% MoM")
    print()

    # Create immune system tester
    tester = ImmuneSystemTester(
        target_col="target",
        survival_threshold_mae=2.0,
        prediction_bounds=(-5.0, 10.0),  # MoM in %
    )

    print("🧪 Black Swan Event Types:")
    print("   1. EXTREME_VALUE     - Massive spike/drop in target")
    print("   2. REGIME_CHANGE     - Distribution shift")
    print("   3. MISSING_DATA       - Data gaps")
    print("   4. FEATURE_OUTLIER   - Exogenous feature shock")
    print("   5. VOLATILITY_EXP    - Noise explosion")
    print("   6. CONSECUTIVE_SHOCKS - Multiple shocks")
    print()

    # Test multiple models
    print("🤖 Testing Models:")
    models = []

    # Create 10 different "models" with varying characteristics
    for i in range(10):
        model = create_sample_model(f"Model{i}")
        models.append(model)
        print(f"   - Model{i}")

    print()
    print("=" * 70)
    print("⚡ STRESS TEST EXECUTION")
    print("=" * 70)
    print()

    # Run stress tests
    reports = tester.test_models(
        models=models, train_data=df, baseline_mae=None, min_survival_rate=90.0
    )

    # Calculate aggregate statistics
    survival_rates = [r.survival_rate for r in reports.values()]
    avg_survival = np.mean(survival_rates)
    min_survival = np.min(survival_rates)
    max_survival = np.max(survival_rates)

    print()
    print("=" * 70)
    print("📈 AGGREGATE RESULTS")
    print("=" * 70)
    print()
    print(f"   Average Survival Rate: {avg_survival:.1f}%")
    print(f"   Minimum Survival Rate: {min_survival:.1f}%")
    print(f"   Maximum Survival Rate: {max_survival:.1f}%")
    print()

    # Check acceptance criteria
    passed = avg_survival >= 90.0

    print("=" * 70)
    print("✅ ACCEPTANCE CRITERIA CHECK")
    print("=" * 70)
    print()
    print(f"   Required:  Survival Rate > 90%")
    print(f"   Actual:   Survival Rate = {avg_survival:.1f}%")
    print()

    if passed:
        print("   ✅ PASSED: Survival Rate > 90%")
        print()
        print("   The Immune System successfully:")
        print("   - Tests model resilience against 6 types of black swan events")
        print("   - Generates 60+ synthetic adversarial scenarios")
        print("   - Validates predictions within reasonable bounds (-5% to +10% MoM)")
        print("   - Detects NaN/Inf predictions as failures")
        print("   - Identifies model vulnerabilities by event type")
        print()

        # Show survival by event type
        print("   Survival by Black Swan Type:")
        event_counts = {}
        for report in reports.values():
            for result in report.results:
                event_type = result.black_swan.type.value
                survived = result.survived
                if event_type not in event_counts:
                    event_counts[event_type] = {"survived": 0, "total": 0}
                event_counts[event_type]["total"] += 1
                if survived:
                    event_counts[event_type]["survived"] += 1

        for event_type, counts in sorted(event_counts.items()):
            rate = counts["survived"] / counts["total"] * 100
            print(
                f"      • {event_type:20s}: {counts['survived']:2d}/{counts['total']:2d} ({rate:5.1f}%)"
            )

        print()
        print("=" * 70)
        print("🎉 TASK 104 COMPLETED SUCCESSFULLY")
        print("=" * 70)
        return True
    else:
        print("   ❌ FAILED: Survival Rate < 90%")
        print()
        print("   Models are too sensitive to black swan events.")
        print("   Consider:")
        print("   - Adding robust loss functions (Huber)")
        print("   - Implementing outlier detection")
        print("   - Adding ensemble methods")
        print()
        return False


if __name__ == "__main__":
    success = verify_immune_system()
    sys.exit(0 if success else 1)
