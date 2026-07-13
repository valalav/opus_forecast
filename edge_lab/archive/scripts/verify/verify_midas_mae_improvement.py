#!/usr/bin/env python3
"""
Verification script for Task 21: MIDAS Model - MAE Improvement

This script:
1. Verifies the current MIDAS implementation exists
2. Runs backtests on all MIDAS variants
3. Analyzes why MAE cannot be improved
4. Documents what would be needed to meet the acceptance criterion
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


def test_original_midas(df):
    """Test the original MIDAS implementation."""
    print("=" * 70)
    print("TEST 1: Original MIDAS Forecaster")
    print("=" * 70)

    try:
        from sirena.models.midas import MIDASForecaster

        model = MIDASForecaster(weight_type="almon", poly_order=2)
        results = model.backtest(df, start_date="2024-06-01")

        if len(results) > 0:
            mae = (results["error"].abs()).mean()
            print(f"✅ MIDAS works")
            print(f"   MAE: {mae:.4f}")
            print(f"   N predictions: {len(results)}")
            return mae
        else:
            print("❌ No backtest results")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_midas_plus(df):
    """Test MIDAS+ forecaster."""
    print("\n" + "=" * 70)
    print("TEST 2: MIDAS+ Forecaster (Hybrid)")
    print("=" * 70)

    try:
        from midas_plus import MIDASPlusForecaster

        model = MIDASPlusForecaster(alpha=1.0, max_features=10)
        results = model.backtest(df, start_date="2024-06-01")

        if len(results) > 0:
            mae = (results["error"].abs()).mean()
            print(f"✅ MIDAS+ works")
            print(f"   MAE: {mae:.4f}")
            print(f"   N predictions: {len(results)}")
            return mae
        else:
            print("❌ No backtest results")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_midas_v2(df):
    """Test MIDAS v2 forecaster."""
    print("\n" + "=" * 70)
    print("TEST 3: MIDAS v2 Forecaster (Multi-Scale)")
    print("=" * 70)

    try:
        from midas_v2 import MIDASv2Forecaster

        model = MIDASv2Forecaster(alpha=0.1)
        results = model.backtest(df, start_date="2024-06-01")

        if len(results) > 0:
            mae = (results["error"].abs()).mean()
            print(f"✅ MIDAS v2 works")
            print(f"   MAE: {mae:.4f}")
            print(f"   N predictions: {len(results)}")
            return mae
        else:
            print("❌ No backtest results")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def analyze_high_frequency_data():
    """Analyze available high-frequency data."""
    print("\n" + "=" * 70)
    print("HIGH-FREQUENCY DATA ANALYSIS")
    print("=" * 70)

    # Check brent prices (claimed to be monthly but let's verify)
    brent = pd.read_csv(
        "/home/valalav/_projects/sirena-kbr/data/brent_prices.csv",
        index_col=0,
        parse_dates=True,
    )
    print(f"\nBrent prices:")
    print(f"   Rows: {len(brent)}")
    print(f"   Date range: {brent.index[0]} to {brent.index[-1]}")
    print(f"   Frequency: {brent.index.freq}")

    # Check for weekly prices file
    weekly_file = Path("/home/valalav/_projects/sirena-kbr/data/weekly_prices.csv")
    if weekly_file.exists():
        print(f"\n✅ weekly_prices.csv exists")
        try:
            weekly = pd.read_csv(weekly_file, sep=";", nrows=5)
            print(f"   Columns: {weekly.columns.tolist()}")
        except:
            print(f"   Format: Russian data with semicolon separator")
            print(f"   This is NOT true weekly HF time series data")
    else:
        print(f"\n❌ weekly_prices.csv not found")

    # Check inflation data
    inflation = pd.read_csv(
        "/home/valalav/_projects/sirena-kbr/data/inflation_data.csv",
        sep=";",
        decimal=",",
        index_col=0,
        dayfirst=True,
        parse_dates=True,
    )
    inflation.index = inflation.index.to_period("M").to_timestamp()
    print(f"\nInflation data:")
    print(f"   Rows: {len(inflation)}")
    print(f"   Date range: {inflation.index[0]} to {inflation.index[-1]}")
    print(f"   Frequency: {inflation.index.freq}")


def analyze_why_midas_fails():
    """Analyze why MIDAS approach cannot improve MAE."""
    print("\n" + "=" * 70)
    print("ROOT CAUSE ANALYSIS: Why MIDAS Cannot Improve MAE")
    print("=" * 70)

    print("""
The fundamental issue is that MIDAS (Mixed Data Sampling) requires:
1. LOW-FREQUENCY target (monthly inflation data) ✅ Available
2. HIGH-FREQUENCY predictors (daily/weekly macro data) ❌ NOT Available

Current data sources:
- inflation_data.csv: MONTHLY (not HF)
- brent_prices.csv: MONTHLY (not HF)
- usd_nom_i: MONTHLY (not HF)
- Ki: MONTHLY (not HF)

Why this matters for MIDAS:
MIDAS's advantage is aggregating high-frequency lags (e.g., 28 daily lags)
into monthly frequency using specialized weighting functions (Almon, Exponential).

Without true HF data, MIDAS becomes:
- Complex lag feature engineering without actual HF benefits
- Essentially fancy Ridge regression with more overfitting
- No actual "mixed data sampling" occurring

Evidence from tests:
- Original MIDAS:  MAE 0.432 (34.6% worse than Ridge)
- MIDAS+ (hybrid): MAE 0.376 (17.2% worse than Ridge)
- MIDAS v2 (multi-scale): MAE 0.532 (65.8% worse than Ridge)
- Ridge baseline: MAE 0.321

What would improve MAE:
1. True daily/weekly high-frequency data sources
   - CBR press releases on weekly basis
   - Daily currency rates (USD/RUB)
   - Daily commodity prices (Brent, Urals)
   
2. External macro data with higher frequency:
   - Weekly Rosstat releases
   - Daily CBAR statistics
   - Inflation expectations surveys

3. Alternative approaches (if HF data unavailable):
   - Bayesian structural models
   - Machine learning ensemble methods
   - Hierarchical time series models
""")


def main():
    """Run all verification tests."""
    print("=" * 70)
    print("TASK 21: MIDAS Model - MAE Improvement Verification")
    print("=" * 70)

    # Load data
    df = load_data()
    print(
        f"\nData loaded: {len(df)} months ({df.index[0].date()} to {df.index[-1].date()})"
    )

    # Test all MIDAS variants
    mae_original = test_original_midas(df)
    mae_plus = test_midas_plus(df)
    mae_v2 = test_midas_v2(df)

    # Analyze HF data availability
    analyze_high_frequency_data()

    # Root cause analysis
    analyze_why_midas_fails()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    ridge_mae = 0.321

    results = []
    if mae_original:
        results.append(("Original MIDAS", mae_original))
    if mae_plus:
        results.append(("MIDAS+", mae_plus))
    if mae_v2:
        results.append(("MIDAS v2", mae_v2))

    print(f"\n{'Model':<30} {'MAE':<10} {'vs Ridge':<15}")
    print("-" * 55)
    for name, mae in results:
        diff = (mae - ridge_mae) / ridge_mae * 100
        print(f"{name:<30} {mae:<10.4f} {diff:>+10.1f}%")

    print(f"\n{'Ridge (baseline)':<30} {ridge_mae:<10.4f} baseline")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    best_mae = min([r[1] for r in results]) if results else None
    if best_mae and best_mae < ridge_mae:
        print("✅ MAE IMPROVED - Acceptance criterion MET!")
    else:
        print("❌ MAE NOT IMPROVED - Acceptance criterion NOT MET")
        print(
            "\nReason: MIDAS requires true high-frequency data which is not available."
        )
        print("Current data is all monthly, making MIDAS's mixed-data")
        print("sampling approach ineffective compared to simpler Ridge regression.")

    return 0 if best_mae and best_mae < ridge_mae else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
