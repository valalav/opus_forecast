#!/usr/bin/env python3
"""
Inflation Persistence Analysis
============================
Analyze inflation inertia across products by fitting AR(k) models and
calculating sum of AR coefficients (persistence metric).

Persistence closer to 1.0 = high inertia (prices tend to continue trends)
Persistence closer to 0.0 = low inertia (prices revert quickly)

Output: data/inflation_persistence.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

# Configuration
DATA_DIR = Path("/home/valalav/_projects/sirena-kbr/data")
OUTPUT_FILE = DATA_DIR / "inflation_persistence.csv"
START_YEAR = 2016  # Post-Crimea period
MIN_OBS = 48  # Minimum observations for AR model (4 years)


def load_weekly_prices():
    """Load weekly price data for all products."""
    df = pd.read_csv(DATA_DIR / "kbr_weekly_prices_2008_2026.csv", parse_dates=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # Filter to recent period
    df = df[df["year"] >= START_YEAR].copy()

    return df


def aggregate_to_monthly(df):
    """Aggregate weekly prices to monthly (last observation)."""

    # Get last price of each month for each product
    monthly = (
        df.groupby(["product_code", "year", "month"], observed=True)
        .agg({"product_name": "first", "price": "last"})
        .reset_index()
    )

    # Create date column
    monthly["date"] = pd.to_datetime(
        monthly["year"].astype(str) + "-" + monthly["month"].astype(str) + "-01"
    )

    # Sort by product and date
    monthly = monthly.sort_values(["product_code", "date"]).reset_index(drop=True)

    return monthly


def calculate_mom_growth(df):
    """Calculate month-over-month growth rates for each product."""

    results = []

    for product_code in df["product_code"].unique():
        product_df = df[df["product_code"] == product_code].copy()
        product_df = product_df.sort_values("date").reset_index(drop=True)

        # Calculate MoM growth: (price_t / price_{t-1} - 1) * 100
        product_df["mom"] = product_df["price"].pct_change() * 100

        # Keep rows with valid mom values
        product_df = product_df.dropna(subset=["mom"])

        # Check minimum observations
        if len(product_df) < MIN_OBS:
            continue

        results.append(product_df)

    return pd.concat(results, ignore_index=True)


def fit_ar_model(series, max_lags=12):
    """
    Fit AR(k) model and select optimal lag using AIC.

    Returns:
        dict: Contains model fit results and persistence metric
    """
    try:
        # Test stationarity
        adf_result = adfuller(series, maxlag=1)
        is_stationary = adf_result[1] < 0.05  # p-value < 0.05

        # Find optimal lag using AIC
        aic_values = []
        models = []

        for lag in range(1, min(max_lags + 1, len(series) // 4)):
            try:
                model = AutoReg(series, lags=lag, trend="c")
                result = model.fit()
                aic_values.append(result.aic)
                models.append((lag, result))
            except:
                continue

        if not models:
            return None

        # Select model with lowest AIC
        best_idx = np.argmin(aic_values)
        best_lag, best_model = models[best_idx]

        # Calculate persistence: sum of AR coefficients
        ar_params = best_model.params[1:]  # Exclude constant
        persistence = ar_params.sum()

        # Get standard errors
        ar_se = best_model.bse[1:]

        return {
            "selected_lag": best_lag,
            "persistence": persistence,
            "persistence_se": np.sqrt(np.sum(ar_se**2)),  # Error propagation
            "aic": best_model.aic,
            "bic": best_model.bic,
            "is_stationary": is_stationary,
            "adf_pvalue": adf_result[1],
            "n_obs": len(series),
        }

    except Exception as e:
        print(f"Error fitting AR model: {e}")
        return None


def analyze_persistence():
    """Main analysis function."""

    print("=" * 60)
    print("INFLATION PERSISTENCE ANALYSIS")
    print("=" * 60)

    # 1. Load data
    print("\n[1/4] Loading weekly price data...")
    weekly_df = load_weekly_prices()
    print(f"     Loaded {len(weekly_df):,} weekly observations")
    print(f"     Period: {weekly_df['date'].min()} to {weekly_df['date'].max()}")
    print(f"     Products: {weekly_df['product_code'].nunique()}")

    # 2. Aggregate to monthly
    print("\n[2/4] Aggregating to monthly...")
    monthly_df = aggregate_to_monthly(weekly_df)
    print(f"     Monthly observations: {len(monthly_df):,}")

    # 3. Calculate MoM growth
    print("\n[3/4] Calculating MoM growth rates...")
    mom_df = calculate_mom_growth(monthly_df)
    print(f"     Valid MoM observations: {len(mom_df):,}")

    # 4. Fit AR models for each product
    print("\n[4/4] Fitting AR(k) models...")

    results = []

    for product_code in mom_df["product_code"].unique():
        product_df = mom_df[mom_df["product_code"] == product_code]
        product_name = product_df["product_name"].iloc[0]
        series = product_df["mom"].values

        # Skip if not enough data
        if len(series) < MIN_OBS:
            continue

        # Fit AR model
        ar_result = fit_ar_model(series, max_lags=12)

        if ar_result is None:
            continue

        # Calculate additional metrics
        mom_std = series.std()
        mom_mean = series.mean()

        # Volatility classification
        if mom_std < 0.5:
            vol_class = "Low"
        elif mom_std < 1.5:
            vol_class = "Medium"
        else:
            vol_class = "High"

        # Persistence interpretation
        if abs(ar_result["persistence"]) < 0.3:
            persistence_class = "Low Inertia"
        elif abs(ar_result["persistence"]) < 0.7:
            persistence_class = "Medium Inertia"
        else:
            persistence_class = "High Inertia"

        results.append(
            {
                "product_code": product_code,
                "product_name": product_name,
                "persistence": ar_result["persistence"],
                "persistence_se": ar_result["persistence_se"],
                "persistence_class": persistence_class,
                "selected_lag": ar_result["selected_lag"],
                "aic": ar_result["aic"],
                "bic": ar_result["bic"],
                "is_stationary": ar_result["is_stationary"],
                "adf_pvalue": ar_result["adf_pvalue"],
                "n_obs": ar_result["n_obs"],
                "mom_mean": mom_mean,
                "mom_std": mom_std,
                "volatility_class": vol_class,
            }
        )

        print(
            f"     {product_code:3d}: {product_name[:40]:40s} "
            f"Persistence={ar_result['persistence']:6.3f} "
            f"(Lag={ar_result['selected_lag']}, "
            f"AIC={ar_result['aic']:.1f})"
        )

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Sort by persistence (absolute value)
    results_df["persistence_abs"] = results_df["persistence"].abs()
    results_df = results_df.sort_values("persistence_abs", ascending=False)

    # Save to CSV
    results_df = results_df.drop("persistence_abs", axis=1)
    results_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\n" + "=" * 60)
    print(f"RESULTS SAVED TO: {OUTPUT_FILE}")
    print(f"Total products analyzed: {len(results_df)}")
    print("=" * 60)

    # Print summary
    print("\nSUMMARY STATISTICS:")
    print(f"  Mean persistence: {results_df['persistence'].mean():.3f}")
    print(f"  Median persistence: {results_df['persistence'].median():.3f}")
    print(f"  Std persistence: {results_df['persistence'].std():.3f}")
    print(f"  Stationary series: {results_df['is_stationary'].sum()}/{len(results_df)}")

    # Print top 10 most persistent
    print("\nTOP 10 MOST PERSISTENT PRODUCTS (High Inertia):")
    print("-" * 60)
    print(
        f"{'Rank':<5} {'Code':<6} {'Persistence':<12} {'Volatility':<12} {'Product Name'}"
    )
    print("-" * 60)

    for i, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
        print(
            f"{i:<5} {row['product_code']:<6} {row['persistence']:>8.3f} ±{row['persistence_se']:.3f} "
            f"{row['volatility_class']:<12} {row['product_name']}"
        )

    # Print bottom 10 least persistent
    print("\nBOTTOM 10 LEAST PERSISTENT PRODUCTS (Low Inertia):")
    print("-" * 60)
    print(
        f"{'Rank':<5} {'Code':<6} {'Persistence':<12} {'Volatility':<12} {'Product Name'}"
    )
    print("-" * 60)

    for i, (_, row) in enumerate(results_df.tail(10).iterrows(), len(results_df) - 9):
        print(
            f"{i:<5} {row['product_code']:<6} {row['persistence']:>8.3f} ±{row['persistence_se']:.3f} "
            f"{row['volatility_class']:<12} {row['product_name']}"
        )

    return results_df


if __name__ == "__main__":
    results = analyze_persistence()
    print(f"\n✅ Analysis complete. Results saved to: {OUTPUT_FILE}")
