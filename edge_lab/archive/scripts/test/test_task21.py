#!/usr/bin/env python3
"""
Test verification for Task 21: MIDAS Model

This script tests all available MIDAS variants and verifies results.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_midas_variants():
    """Test all MIDAS variants."""
    df = pd.read_csv(
        "/home/valalav/_projects/sirena-kbr/data/inflation_data.csv",
        sep=";",
        decimal=",",
        index_col=0,
        dayfirst=True,
        parse_dates=True,
    )
    df.index = df.index.to_period("M").to_timestamp()

    brent = pd.read_csv(
        "/home/valalav/_projects/sirena-kbr/data/brent_prices.csv",
        index_col=0,
        parse_dates=True,
    )
    df = df.join(brent[["brent"]], how="left")
    df = df[["mom", "brent", "usd_nom_i", "Ki"]].copy()
    df = df.rename(columns={"mom": "Все товары и услуги"})
    df = df.dropna(subset=["Все товары и услуги", "usd_nom_i"])

    print("=" * 70)
    print("TASK 21: MIDAS Model - Test Verification")
    print("=" * 70)
    print(f"\nData: {len(df)} months")

    results = []

    print("\n" + "=" * 70)
    print("Testing MIDAS+ (Hybrid) forecaster")
    print("=" * 70)

    try:
        from midas_plus import MIDASPlusForecaster

        model = MIDASPlusForecaster(alpha=0.1, max_features=20)
        backtest_results = model.backtest(df, start_date="2024-06-01")

        if len(backtest_results) > 0:
            mae = (backtest_results["error"].abs()).mean()
            print(f"✅ MIDAS+ forecaster works")
            print(f"   MAE: {mae:.4f}")
            print(f"   N predictions: {len(backtest_results)}")
            results.append(("MIDAS+", mae, len(backtest_results)))
        else:
            print("❌ No backtest results")
            results.append(("MIDAS+", None, 0))
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append(("MIDAS+", None, 0))

    print("\n" + "=" * 70)
    print("Testing Optimized MIDAS+ (Ridge CV) forecaster")
    print("=" * 70)

    try:
        from optimized_midas import OptimizedMIDASForecaster

        model = OptimizedMIDASForecaster(alpha=None, max_features=10)
        backtest_results = model.backtest(df, start_date="2024-06-01")

        if len(backtest_results) > 0:
            mae = (backtest_results["error"].abs()).mean()
            print(f"✅ Optimized MIDAS+ forecaster works")
            print(f"   MAE: {mae:.4f}")
            print(f"   N predictions: {len(backtest_results)}")
            results.append(("Optimized MIDAS+", mae, len(backtest_results)))
        else:
            print("❌ No backtest results")
            results.append(("Optimized MIDAS+", None, 0))
    except Exception as e:
        print(f"❌ Error: {e}")
        results.append(("Optimized MIDAS+", None, 0))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    ridge_mae = 0.321

    print(f"\n{'Model':<25} {'MAE':<10} {'vs Ridge':<15} {'Status':<10}")
    print("-" * 60)

    for name, mae, n in results:
        if mae is not None:
            diff = (mae - ridge_mae) / ridge_mae * 100
            status = "FAIL" if mae > ridge_mae else "PASS"
            print(f"{name:<25} {mae:<10.4f} {diff:>+10.1f}% {status:<10}")
        else:
            print(f"{name:<25} {'N/A':<10} {'-':<15} {'ERROR':<10}")

    print(f"\n{'Ridge (baseline)':<25} {ridge_mae:<10.4f} {'-':<15} {'baseline':<10}")

    best_mae = min([r[1] for r in results if r[1] is not None])

    print("\n" + "=" * 70)
    print("ACCEPTANCE CRITERION: MAE improved")
    print("=" * 70)

    if best_mae < ridge_mae:
        print(f"✅ MAE IMPROVED - Acceptance criterion MET!")
        print(f"   Best MAE: {best_mae:.4f} < Ridge {ridge_mae:.4f}")
        return 0
    else:
        print(f"❌ MAE NOT IMPROVED - Acceptance criterion NOT MET")
        print(f"   Best MAE: {best_mae:.4f} > Ridge {ridge_mae:.4f}")
        print(f"   Difference: {(best_mae - ridge_mae) / ridge_mae * 100:+.1f}%")
        print(f"\nRoot cause: No true high-frequency data available.")
        print(f"   MIDAS requires HF predictors to have advantage over Ridge.")
        print(f"   All available predictors are monthly, making MIDAS ineffective.")
        return 1


if __name__ == "__main__":
    exit_code = test_midas_variants()
    sys.exit(exit_code)
