#!/usr/bin/env python3
"""
Brent Oil Lag Optimization for ExogProphet

Calculates the optimal lag for Brent oil prices as a leading indicator
for Russian inflation using Cross-Correlation Function (CCF).

Output: data/brent_lag_analysis.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats


def load_inflation_data():
    """Load inflation data (CPI) from parent directory."""
    data_path = Path("../data/inflation_data.csv")

    if not data_path.exists():
        raise FileNotFoundError(f"Inflation data not found: {data_path}")

    df = pd.read_csv(data_path, sep=";")
    df.columns = [col.strip() for col in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
    df["Date"] = df["Date"].apply(lambda x: x.replace(day=1))

    if "mom" not in df.columns:
        raise ValueError("Column 'mom' not found in inflation data")

    df["mom"] = df["mom"].astype(str).str.replace(",", ".").astype(float)

    df = df.set_index("Date").sort_index()

    return df[["mom"]]


def load_brent_data():
    """Load Brent oil prices from parent directory."""
    data_path = Path("../data/brent_prices.csv")

    if not data_path.exists():
        raise FileNotFoundError(f"Brent data not found: {data_path}")

    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"] = df["Date"].apply(lambda x: x.replace(day=1))

    if "brent" not in df.columns:
        raise ValueError("Column 'brent' not found in Brent data")

    df["brent"] = pd.to_numeric(df["brent"], errors="coerce")

    df = df.set_index("Date").sort_index()

    return df[["brent"]]


def calculate_ccf(x, y, max_lag=12):
    """
    Calculate Cross-Correlation Function for lags 0 to max_lag.

    Args:
        x: First time series (Brent)
        y: Second time series (CPI)
        max_lag: Maximum lag to analyze in months

    Returns:
        Dictionary of lag -> correlation values
    """
    correlations = {}

    for lag in range(0, max_lag + 1):
        x_lagged = x.shift(lag)

        valid_idx = x_lagged.notna() & y.notna()
        x_valid = x_lagged[valid_idx]
        y_valid = y[valid_idx]

        if len(x_valid) > 2:
            corr, _ = stats.pearsonr(x_valid, y_valid)
            correlations[lag] = float(corr)
        else:
            correlations[lag] = None

    return correlations


def find_optimal_lag(correlations):
    """
    Find the lag with highest absolute correlation.

    Args:
        correlations: Dict of lag -> correlation

    Returns:
        Tuple of (optimal_lag, max_correlation)
    """
    valid_correlations = {k: v for k, v in correlations.items() if v is not None}

    if not valid_correlations:
        raise ValueError("No valid correlations found")

    optimal_lag = max(valid_correlations.items(), key=lambda x: abs(x[1]))

    return optimal_lag[0], optimal_lag[1]


def main():
    print("=" * 60)
    print("Brent Lag Optimization for ExogProphet")
    print("=" * 60)

    print("\nLoading CPI data...")
    cpi_df = load_inflation_data()
    print(f"  - CPI data shape: {cpi_df.shape}")
    print(f"  - Date range: {cpi_df.index.min()} to {cpi_df.index.max()}")

    print("\nLoading Brent data...")
    brent_df = load_brent_data()
    print(f"  - Brent data shape: {brent_df.shape}")
    print(f"  - Date range: {brent_df.index.min()} to {brent_df.index.max()}")

    print("\nMerging and aligning data...")
    merged_df = pd.merge(
        cpi_df, brent_df, left_index=True, right_index=True, how="inner"
    )
    print(f"  - Merged data shape: {merged_df.shape}")
    print(f"  - Date range: {merged_df.index.min()} to {merged_df.index.max()}")

    print("\nCalculating Cross-Correlation Function (CCF) for lags 0-12...")
    brent_series = merged_df["brent"]
    cpi_series = merged_df["mom"]

    correlations = calculate_ccf(brent_series, cpi_series, max_lag=12)

    print("\nCorrelations by lag:")
    print("  Lag  |  Correlation")
    print("  " + "-" * 25)
    for lag, corr in correlations.items():
        if corr is not None:
            print(f"  {lag:3d}  |  {corr:+.4f}")
        else:
            print(f"  {lag:3d}  |  N/A")

    optimal_lag, max_correlation = find_optimal_lag(correlations)
    print("\n" + "=" * 60)
    print(f"Optimal Lag: {optimal_lag} months")
    print(f"Max Correlation: {max_correlation:+.4f}")
    print("=" * 60)

    result = {
        "optimal_lag": optimal_lag,
        "correlation": max_correlation,
        "all_correlations": {str(k): v for k, v in correlations.items()},
        "data_points": len(merged_df),
        "date_range": {
            "start": str(merged_df.index.min()),
            "end": str(merged_df.index.max()),
        },
    }

    output_path = Path("data/brent_lag_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")

    return result


if __name__ == "__main__":
    main()
