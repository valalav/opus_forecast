#!/usr/bin/env python3
"""
Verify ExogProphet Improved Version
==================================
Tests that the improved ExogProphet achieves MAE <= 0.30
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.models.exog_prophet import ExogProphetForecaster


def test_exog_prophet_mae():
    """Test that improved ExogProphet achieves MAE <= 0.30."""
    print("=" * 60)
    print("Testing Improved ExogProphet")
    print("=" * 60)

    # Create model with optimized parameters
    model = ExogProphetForecaster(
        use_usd=True,
        use_brent=True,
        use_ki=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=0.01,
        seasonality_prior_scale=1.0,
        outlier_years=[2022],
    )

    print(f"\nModel: {model.name}")
    print(f"Regressors: {model.regressors}")
    print(f"Lags: USD={model.USD_LAG}, Brent={model.BRENT_LAG}, Ki={model.KI_LAG}")

    # Run backtest for h=1 (primary horizon)
    print("\nRunning backtest...")
    dummy_df = pd.DataFrame({"dummy": [1]})
    results = model.backtest(
        dummy_df, start_date="2019-01-01", target_col="Все товары и услуги", horizon=1
    )

    if results is None or len(results) == 0:
        print("ERROR: No backtest results!")
        return False

    # Calculate MAE
    results["abs_error"] = np.abs(results["error"])
    mae = results["abs_error"].mean()

    print(f"\nBacktest Results:")
    print(f"  Test period: {results['date'].min()} to {results['date'].max()}")
    print(f"  Number of forecasts: {len(results)}")
    print(f"  MAE: {mae:.4f}")
    print(f"  ME (Mean Error): {results['error'].mean():.4f}")
    print(f"  RMSE: {np.sqrt((results['error'] ** 2).mean()):.4f}")

    # Check acceptance criterion
    criterion = 0.30
    passed = mae <= criterion

    print(f"\nAcceptance Criterion:")
    print(f"  Required: MAE <= {criterion}")
    print(f"  Actual:   MAE = {mae:.4f}")
    print(f"  Status:   {'PASS ✓' if passed else 'FAIL ✗'}")

    # Get regressor importance
    print("\n" + "-" * 60)
    print("Regressor Importance")
    print("-" * 60)

    try:
        model.fit(pd.DataFrame({"dummy": [1]}))
        importance = model.get_regressor_importance()
        if importance:
            for reg, imp in sorted(importance.items(), key=lambda x: -x[1]):
                print(f"  {reg}: {imp:.4f}")
        else:
            print("  No importance data available")
    except Exception as e:
        print(f"  Could not extract importance: {e}")

    print("\n" + "=" * 60)
    if passed:
        print("RESULT: PASSED ✓")
        print("ExogProphet meets MAE <= 0.30 criterion")
    else:
        print("RESULT: FAILED ✗")
        print(f"ExogProphet MAE ({mae:.4f}) exceeds threshold of {criterion}")
    print("=" * 60)

    return passed


if __name__ == "__main__":
    passed = test_exog_prophet_mae()
    sys.exit(0 if passed else 1)
