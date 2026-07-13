#!/usr/bin/env python3
"""
Verify ExogProphet V2 with date alignment fix and optimized parameters
========================================================================

Tests that ExogProphetV2 achieves MAE <= 0.30
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add opus_forecast directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from exog_prophet_v2 import ExogProphetV2


def test_exog_prophet_v2_mae():
    """Test that ExogProphetV2 achieves MAE <= 0.30."""
    print("=" * 60)
    print("Testing ExogProphet V2 (Optimized with Brent)")
    print("=" * 60)

    # Create model with Brent regressor enabled
    model = ExogProphetV2(
        use_usd=True,
        use_brent=True,
        use_ki=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=0.1,  # Optimized
        seasonality_prior_scale=15.0,  # Optimized
        outlier_years=[2022],
    )

    print(f"\nModel: {model.name}")
    print(
        f"Regressors: USD lag-{model.USD_LAG}, Brent lag-{model.BRENT_LAG}, Ki lag-{model.KI_LAG}"
    )
    print(
        f"Params: changepoint_prior_scale={model.changepoint_prior_scale}, seasonality_prior_scale={model.seasonality_prior_scale}"
    )

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

    # Test with Brent disabled to see improvement
    print("\n" + "-" * 60)
    print("Comparing: ExogProphet V2 WITHOUT Brent regressor")
    print("-" * 60)

    model_no_brent = ExogProphetV2(
        use_usd=True,
        use_brent=False,
        use_ki=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=15.0,
        outlier_years=[2022],
    )

    results_no_brent = model_no_brent.backtest(
        pd.DataFrame({"dummy": [1]}),
        start_date="2019-01-01",
        target_col="Все товары и услуги",
        horizon=1,
    )

    if results_no_brent is not None and len(results_no_brent) > 0:
        mae_no_brent = np.abs(results_no_brent["error"]).mean()
        improvement = ((mae_no_brent - mae) / mae_no_brent) * 100

        print(f"  MAE (without Brent): {mae_no_brent:.4f}")
        print(f"  MAE (with Brent):    {mae:.4f}")
        print(f"  Improvement:         {improvement:+.2f}%")

    # Get regressor importance
    print("\n" + "-" * 60)
    print("Regressor Importance")
    print("-" * 60)

    try:
        model.fit(pd.DataFrame({"dummy": [1]}))
        importance = model.get_regressor_importance()
        if importance:
            for reg, imp in importance.items():
                print(f"  {reg}: {imp:.4f}")
        else:
            print("  No regressor importance data available")
    except Exception as e:
        print(f"  Could not extract importance: {e}")

    print("\n" + "=" * 60)
    if passed:
        print("RESULT: PASSED ✓")
        print("ExogProphet V2 meets MAE <= 0.30 criterion")
    else:
        print("RESULT: FAILED ✗")
        print(f"ExogProphet V2 MAE ({mae:.4f}) exceeds threshold of {criterion}")
    print("=" * 60)

    return passed


if __name__ == "__main__":
    passed = test_exog_prophet_v2_mae()
    sys.exit(0 if passed else 1)
