#!/usr/bin/env python3
"""
Debug ExogProphet
=================
Debug script to check data loading and ExogProphet functionality
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add opus_forecast directory to path to import sirena module
sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.models.exog_prophet import ExogProphetForecaster, PROPHET_AVAILABLE


def main():
    print("=" * 60)
    print("ExogProphet Debug")
    print("=" * 60)

    # Check Prophet availability
    print(f"\nProphet Available: {PROPHET_AVAILABLE}")

    # Create model
    model = ExogProphetForecaster(
        use_usd=True,
        use_brent=True,
        use_ki=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        outlier_years=[2022],
    )

    # Test data loading
    print("\n" + "-" * 60)
    print("Testing Data Loading")
    print("-" * 60)

    try:
        macro_df = model._load_macro_data()
        print(f"\nMacro data loaded: {macro_df.shape}")
        print(f"  Columns: {list(macro_df.columns)}")
        print(f"  Date range: {macro_df.index.min()} to {macro_df.index.max()}")
        print(f"  Has 'mom': {'mom' in macro_df.columns}")
        print(f"  Has 'Ki': {'Ki' in macro_df.columns}")
        print(f"  Has 'usd_nom_i': {'usd_nom_i' in macro_df.columns}")
        print("\nFirst 3 rows:")
        print(macro_df.head(3))
    except Exception as e:
        print(f"ERROR loading macro data: {e}")
        return False

    try:
        brent_df = model._load_brent_data()
        if brent_df is not None:
            print(f"\nBrent data loaded: {brent_df.shape}")
            print(f"  Columns: {list(brent_df.columns)}")
            print(f"  Date range: {brent_df.index.min()} to {brent_df.index.max()}")
            print("\nFirst 3 rows:")
            print(brent_df.head(3))
        else:
            print(f"\nBrent data: NOT FOUND")
    except Exception as e:
        print(f"ERROR loading brent data: {e}")

    # Test feature preparation
    print("\n" + "-" * 60)
    print("Testing Feature Preparation")
    print("-" * 60)

    try:
        model.macro_df = macro_df
        model.brent_df = brent_df
        prepared_df = model._prepare_features(macro_df)
        print(f"\nPrepared features: {prepared_df.shape}")
        print(f"  Columns: {list(prepared_df.columns)}")
        print(f"  Has 'usd_lag2': {'usd_lag2' in prepared_df.columns}")
        print(f"  Has 'ki_lag6': {'ki_lag6' in prepared_df.columns}")
        print(f"  Has 'brent_lag5': {'brent_lag5' in prepared_df.columns}")
    except Exception as e:
        print(f"ERROR preparing features: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test Prophet dataframe preparation
    print("\n" + "-" * 60)
    print("Testing Prophet DataFrame Preparation")
    print("-" * 60)

    try:
        prophet_df = model._prepare_prophet_df(prepared_df, "mom")
        print(f"\nProphet dataframe: {prophet_df.shape}")
        print(f"  Columns: {list(prophet_df.columns)}")
        print(f"  Non-null counts:")
        print(prophet_df.notna().sum())
        print(f"\nFirst 3 rows:")
        print(prophet_df.head(3))
    except Exception as e:
        print(f"ERROR preparing prophet df: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test fit
    print("\n" + "-" * 60)
    print("Testing Model Fit")
    print("-" * 60)

    try:
        dummy_df = pd.DataFrame({"dummy": [1]})
        model.fit(dummy_df)
        print(f"\nModel fitted successfully!")
        print(f"  Last date: {model.last_date}")
        print(f"  Regressors used: {model.regressors}")
        print(f"  Is fitted: {model._is_fitted}")
    except Exception as e:
        print(f"ERROR fitting model: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test forecast
    print("\n" + "-" * 60)
    print("Testing Forecast")
    print("-" * 60)

    try:
        horizon = 3
        fc = model.forecast(horizon)
        print(f"\nForecast (h={horizon}):")
        print(f"  Values: {fc}")
    except Exception as e:
        print(f"ERROR forecasting: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test backtest
    print("\n" + "-" * 60)
    print("Testing Backtest")
    print("-" * 60)

    try:
        results = model.backtest(
            pd.DataFrame({"dummy": [1]}),
            start_date="2019-01-01",
            target_col="Все товары и услуги",
            horizon=1,
        )
        if results is not None and len(results) > 0:
            print(f"\nBacktest successful!")
            print(f"  Number of forecasts: {len(results)}")
            print(f"  Date range: {results['date'].min()} to {results['date'].max()}")
            mae = np.abs(results["error"]).mean()
            print(f"  MAE: {mae:.4f}")
        else:
            print(f"\nBacktest returned no results!")
            return False
    except Exception as e:
        print(f"ERROR backtesting: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
