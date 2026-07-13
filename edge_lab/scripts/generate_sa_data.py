#!/usr/bin/env python3
"""
Generate Seasonally Adjusted (SA) Inflation Data
================================================

Creates data/sa_fl.csv from raw inflation data using centered moving average
for seasonal decomposition without statsmodels dependency.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data():
    """Load raw inflation data."""
    data_path = Path(__file__).parent.parent / "data" / "enhanced_inflation_data.csv"
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


def calculate_cma(series, window=12):
    """Calculate centered moving average for trend."""
    cma = series.rolling(window=window, center=True).mean()
    return cma


def decompose_seasonal(df, target_col="mom"):
    """
    Simple seasonal decomposition using CMA method.

    Returns: trend, seasonal, residual components
    """
    series = df[target_col].copy()

    # Calculate trend using centered moving average
    trend = calculate_cma(series, window=12)

    # Calculate detrended series
    detrended = series - trend

    # Calculate seasonal component by averaging detrended values by month
    seasonal = pd.Series(index=series.index, dtype=float)
    for month in range(1, 13):
        # Get all values for this month (excluding NaN in trend)
        mask = (series.index.month == month) & (~trend.isna())
        if mask.sum() > 0:
            seasonal_avg = detrended[mask].mean()
            seasonal[mask] = seasonal_avg

    # Calculate residual
    residual = series - trend - seasonal

    return trend, seasonal, residual


def create_sa_data(df, target_col="mom"):
    """Create seasonally adjusted series."""
    trend, seasonal, _ = decompose_seasonal(df, target_col)

    # Seasonally adjusted = raw - seasonal
    sa_series = df[target_col] - seasonal

    return sa_series


def main():
    """Generate SA data file."""
    print("Generating Seasonally Adjusted Inflation Data...")
    print("=" * 50)

    df = load_data()
    print(
        f"Loaded raw data: {len(df)} observations ({df.index.min()} to {df.index.max()})"
    )

    # Calculate SA for inflation (mom)
    sa_mom = create_sa_data(df, "mom")

    # Create output DataFrame
    sa_df = pd.DataFrame(
        {
            "Date": df.index,
            "mom_raw": df["mom"].values,
            "mom_sa": sa_mom.values,
        }
    )

    # Also adjust other key columns
    for col in ["Nonprod", "Prod", "Serv"]:
        if col in df.columns:
            sa_col = create_sa_data(df, col)
            sa_df[f"{col}_raw"] = df[col].values
            sa_df[f"{col}_sa"] = sa_col.values

    # Save to CSV
    output_path = Path(__file__).parent.parent / "data" / "sa_fl.csv"
    sa_df.to_csv(output_path, index=False)
    print(f"\nSaved SA data to {output_path}")

    # Summary stats
    print("\n--- Summary Statistics ---")
    print(f"Raw MOM mean: {df['mom'].mean():.4f}")
    print(f"SA MOM mean: {sa_mom.mean():.4f}")
    print(f"Seasonal range: {sa_mom.max() - sa_mom.min():.4f}")
    print(f"Raw range: {df['mom'].max() - df['mom'].min():.4f}")

    print("\nSA data generation complete!")


if __name__ == "__main__":
    main()
