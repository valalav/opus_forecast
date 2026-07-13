#!/usr/bin/env python3
"""
Final verification for Task 21: MIDAS Model - MAE Improvement

This script provides comprehensive analysis of why MAE cannot be improved
with current data constraints and documents the findings.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_data():
    """Load inflation and macro data."""
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

    return df


def analyze_data_frequency():
    """Analyze available data frequency."""
    print("=" * 70)
    print("DATA FREQUENCY ANALYSIS")
    print("=" * 70)

    df = load_data()

    print(f"\nInflation data:")
    print(f"  Rows: {len(df)}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Frequency: MONTHLY (not high-frequency)")

    print(f"\nAvailable predictors (all MONTHLY):")
    for col in ["brent", "usd_nom_i", "Ki"]:
        if col in df.columns:
            print(f"  - {col}: Monthly")

    print(f"\n❌ CRITICAL FINDING:")
    print(f"  No high-frequency (daily/weekly) data available.")
    print(f"  MIDAS requires HF predictors to have advantage.")
    print(f"  Without HF data, MIDAS is just complex lag feature engineering.")


def compare_models(df):
    """Compare all MIDAS variants against Ridge baseline."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    ridge_mae = 0.321

    results = []

    print("\n1. Original MIDAS")
    try:
        from sirena.models.midas import MIDASForecaster

        model = MIDASForecaster(weight_type="almon", poly_order=2)
        midas_results = model.backtest(df, start_date="2024-06-01")
        if len(midas_results) > 0:
            mae = (midas_results["error"].abs()).mean()
            results.append(("Original MIDAS", mae))
            print(
                f"   MAE: {mae:.4f} ({(mae - ridge_mae) / ridge_mae * 100:+.1f}% vs Ridge)"
            )
    except Exception as e:
        print(f"   Error: {e}")

    print("\n2. MIDAS+ (Hybrid)")
    try:
        from midas_plus import MIDASPlusForecaster

        model = MIDASPlusForecaster(alpha=0.1, max_features=20)
        plus_results = model.backtest(df, start_date="2024-06-01")
        if len(plus_results) > 0:
            mae = (plus_results["error"].abs()).mean()
            results.append(("MIDAS+", mae))
            print(
                f"   MAE: {mae:.4f} ({(mae - ridge_mae) / ridge_mae * 100:+.1f}% vs Ridge)"
            )
    except Exception as e:
        print(f"   Error: {e}")

    print("\n3. Optimized MIDAS+ (Ridge CV)")
    try:
        from optimized_midas import OptimizedMIDASForecaster

        model = OptimizedMIDASForecaster(alpha=None, max_features=10)
        opt_results = model.backtest(df, start_date="2024-06-01")
        if len(opt_results) > 0:
            mae = (opt_results["error"].abs()).mean()
            results.append(("Optimized MIDAS+", mae))
            print(
                f"   MAE: {mae:.4f} ({(mae - ridge_mae) / ridge_mae * 100:+.1f}% vs Ridge)"
            )
    except Exception as e:
        print(f"   Error: {e}")

    print("\n4. Ridge Baseline")
    print(f"   MAE: {ridge_mae:.4f} (baseline)")

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<25} {'MAE':<10} {'vs Ridge':<15}")
    print("-" * 50)
    for name, mae in results:
        diff = (mae - ridge_mae) / ridge_mae * 100
        print(f"{name:<25} {mae:<10.4f} {diff:>+10.1f}%")

    best_mae = min([r[1] for r in results]) if results else None

    return best_mae


def analyze_ridge_advantages():
    """Analyze why Ridge baseline is hard to beat."""
    print("\n" + "=" * 70)
    print("RIDGE BASELINE ANALYSIS")
    print("=" * 70)

    print("""
Why Ridge baseline (MAE=0.321) is hard to beat:

1. Sophisticated Feature Engineering:
   - ETS (Exponential Trend Seasonality) with monthly weights
   - Lags: y_lag1, y_lag2, y_lag12
   - Seasonal sin/cos features
   - Subcomponent lags (food, nonfood, services)
   - 3-month moving average (y_ma3)
   - Macro features: ruonia_diff_lag1, spread_lag4, ki_diff_lag6, ki_vol

2. Domain Knowledge:
   - Outlier year handling (exclude 2022, 2010)
   - RobustScaler (resistant to outliers)
   - Seasonal normalization based on historical patterns

3. Proper Regularization:
   - Alpha=0.3 (optimized via CV)
   - RidgeCV for hyperparameter selection
   - Prevents overfitting on small dataset

4. Data-Limitation Resilient:
   - Works well with monthly data
   - Doesn't require high-frequency inputs
   - Optimized for the specific inflation forecasting problem

Without true high-frequency data, any MIDAS approach:
- Reduces to complex lag feature engineering
- Lacks the actual "mixed-data sampling" benefit
- Overfits on small training sets
- Cannot leverage MIDAS's core advantage
""")


def document_root_cause():
    """Document the root cause of failure."""
    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 70)

    print("""
TASK REQUIREMENT:
  - Improve MAE for MIDAS Model
  - Baseline: Ridge MAE = 0.321

FUNDAMENTAL CONSTRAINT:
  - MIDAS (Mixed Data Sampling) requires:
    1. LOW-FREQUENCY target (monthly inflation) ✅ Available
    2. HIGH-FREQUENCY predictors (daily/weekly) ❌ NOT Available

CURRENT DATA SITUATION:
  - inflation_data.csv: MONTHLY
  - brent_prices.csv: MONTHLY (not daily)
  - usd_nom_i: MONTHLY
  - Ki: MONTHLY
  - weekly_prices.csv: Monthly breakdown by products (not HF time series)

WHY MIDAS FAILS:
  MIDAS's advantage is aggregating high-frequency lags (e.g., 28 daily lags)
  into monthly frequency using specialized weighting functions (Almon, Exponential).

  Without true HF data, MIDAS becomes:
  - Complex lag feature engineering without HF benefits
  - Essentially fancy Ridge with more parameters (more overfitting)
  - No actual "mixed data sampling" occurring

ATTEMPTED SOLUTIONS:
  1. Original MIDAS: MAE 0.432 (+34.6%) - Uses Almon weights
  2. MIDAS+ (Hybrid): MAE 0.376 (+17.2%) - Ridge-based with features
  3. Optimized MIDAS+: MAE 0.399 (+24.3%) - RidgeCV with feature selection
  4. Advanced MIDAS (XGBoost): MAE 0.538 (+67.6%) - Overfits on small data

  All approaches fail to beat Ridge baseline.

CONCLUSION:
  The acceptance criterion "MAE improved" CANNOT be met with current data
  because:
  1. No true high-frequency data available
  2. Ridge baseline is highly optimized for monthly data
  3. MIDAS's core advantage (HF aggregation) is unavailable
  4. Small backtest window (19 predictions) limits complex models

RECOMMENDATIONS TO IMPROVE MAE:
  1. Obtain true high-frequency data:
     - Daily USD/RUB exchange rates
     - Daily commodity prices (Brent, Urals)
     - Weekly CBR press releases
     - Daily CBR statistics

  2. Alternative approaches (if HF data unavailable):
     - Improve ETS seasonal weights
     - Add more macro indicators (PPI, wages, etc.)
     - Use hierarchical models with subcomponents
     - Implement ensemble methods (weighted average)
""")


def main():
    """Run final verification."""
    print("=" * 70)
    print("TASK 21: MIDAS Model - Final Verification")
    print("=" * 70)

    df = load_data()
    print(f"\nData: {len(df)} months ({df.index[0].date()} to {df.index[-1].date()})")

    analyze_data_frequency()
    best_mae = compare_models(df)
    analyze_ridge_advantages()
    document_root_cause()

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    ridge_mae = 0.321

    if best_mae and best_mae < ridge_mae:
        print("✅ MAE IMPROVED - Acceptance criterion MET")
        print(f"   Best MAE: {best_mae:.4f} vs Ridge {ridge_mae:.4f}")
        return 0
    else:
        print("❌ MAE NOT IMPROVED - Acceptance criterion NOT MET")
        print(f"\n   The fundamental issue is lack of true high-frequency data.")
        print(f"   MIDAS requires HF predictors to beat Ridge.")
        print(f"   Without HF data, Ridge's optimized approach is superior.")
        best_str = f"{best_mae:.4f}" if best_mae is not None else "N/A"
        print(f"\n   Best achieved MAE: {best_str} vs Ridge {ridge_mae:.4f}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
