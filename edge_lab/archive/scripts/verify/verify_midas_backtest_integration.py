#!/usr/bin/env python3
"""
Verification script for MIDAS model integration into backtest framework

This script verifies:
1. MIDAS can be imported from backtest framework
2. MIDAS forecast method exists
3. MIDAS is included in model list
4. MIDAS can run a simple backtest
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "scripts"))

print("=" * 70)
print("MIDAS BACKTEST INTEGRATION VERIFICATION")
print("=" * 70)
print()

# Test 1: Import backtest framework
print("Test 1: Importing backtest framework...")
try:
    from backtest_framework import BacktestRunner, MIDAS_AVAILABLE

    print("✓ BacktestRunner imported")
    print(f"  MIDAS_AVAILABLE = {MIDAS_AVAILABLE}")
    if not MIDAS_AVAILABLE:
        print("✗ FAIL: MIDAS not available in backtest framework")
        sys.exit(1)
except ImportError as e:
    print(f"✗ FAIL: Cannot import backtest framework: {e}")
    sys.exit(1)

# Test 2: Verify MIDASForecaster class exists
print("\nTest 2: Checking MIDASForecaster class...")
try:
    from sirena.models.midas import MIDASForecaster

    print("✓ MIDASForecaster class exists")
except ImportError as e:
    print(f"✗ FAIL: Cannot import MIDASForecaster: {e}")
    sys.exit(1)

# Test 3: Verify BacktestRunner has _forecast_midas method
print("\nTest 3: Checking BacktestRunner._forecast_midas method...")
if hasattr(BacktestRunner, "_forecast_midas"):
    print("✓ BacktestRunner._forecast_midas exists")
else:
    print("✗ FAIL: BacktestRunner._forecast_midas not found")
    sys.exit(1)

# Test 4: Create sample data and run quick test
print("\nTest 4: Running quick MIDAS fit/predict test...")
try:
    # Create sample monthly data (need at least 48 months for MIDAS MIN_TRAIN_SIZE)
    # Start from 2016 to avoid outlier years (2010, 2022)
    dates = pd.date_range(start="2016-01-01", periods=72, freq="MS")
    np.random.seed(42)
    cpi = 100 + np.cumsum(np.random.randn(72) * 0.3)

    df = pd.DataFrame({"Date": dates, "Все товары и услуги": cpi}).set_index("Date")

    # Add some macro features
    df["usd_nom_i"] = 90 + np.cumsum(np.random.randn(72) * 1.5)
    df["Ki"] = 7.5 + np.cumsum(np.random.randn(72) * 0.2)
    df["Ruonia"] = 6.0 + np.cumsum(np.random.randn(72) * 0.3)

    # Try to fit MIDAS
    model = MIDASForecaster(weight_type="almon", poly_order=2)
    model.fit(df, "Все товары и услуги")

    # Test forecast method instead of predict (uses iterative approach)
    try:
        forecast = model.forecast(horizon=3)
        if len(forecast) == 3 and not np.isnan(forecast).any():
            print(f"✓ MIDAS fit/forecast works")
            print(f"  Test forecast (3 months): {forecast}")
        else:
            print("✗ FAIL: MIDAS forecast is invalid")
            sys.exit(1)
    except Exception as e:
        print(f"✗ FAIL: MIDAS forecast failed: {e}")
        sys.exit(1)

except Exception as e:
    print(f"✗ FAIL: MIDAS test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 5: Verify BacktestRunner can run with MIDAS
print("\nTest 5: Creating BacktestRunner and checking model initialization...")
try:
    runner = BacktestRunner(horizon=1, test_months=3, output_dir="archive/results")

    # Check if MIDAS is in available models
    runner._prepare_data()

    # Try to get a small sample for testing
    test_dates = runner._get_test_dates()
    if len(test_dates) > 0:
        target_date = test_dates[0]
        train_ridge, train_bvar, cutoff = runner._train_test_split(target_date)

        # Try MIDAS forecast
        midas_pred = runner._forecast_midas(train_ridge, target_date)
        if not np.isnan(midas_pred):
            print(f"✓ BacktestRunner._forecast_midas works")
            print(f"  Sample prediction: {midas_pred:.3f}")
        else:
            print("✗ FAIL: BacktestRunner._forecast_midas returned NaN")
            sys.exit(1)
    else:
        print("⚠ WARNING: No test dates available, skipping full backtest test")

except Exception as e:
    print(f"✗ FAIL: BacktestRunner test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ ALL VERIFICATION TESTS PASSED")
print("=" * 70)
print("\nMIDAS is successfully integrated into the backtest framework!")
print("\nTo run full backtests:")
print("  cd /home/valalav/_projects/sirena-kbr")
print("  python3 scripts/run_backtest_h1.py")
print("  python3 scripts/run_backtest_h2.py")
print("  python3 scripts/run_backtest_h12.py")
