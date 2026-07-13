#!/usr/bin/env python3
"""
Optimal Lag Discovery for Macro Regressors

Systematically finds optimal lags for Ki (Key Rate), USD, and Brent
using Cross-Correlation Function (CCF).

Output: data/optimal_lags.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats


def load_inflation_data():
    """Load inflation data (CPI) from enhanced_inflation_data.csv."""
    data_path = Path("data/enhanced_inflation_data.csv")

    if not data_path.exists():
        raise FileNotFoundError(f"Inflation data not found: {data_path}")

    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    if "mom" not in df.columns:
        raise ValueError("Column 'mom' not found in inflation data")

    df["mom"] = pd.to_numeric(df["mom"], errors="coerce")

    return df[["mom"]]


def load_brent_data():
    """Load Brent oil prices from brent_prices.csv."""
    data_path = Path("data/brent_prices.csv")

    if not data_path.exists():
        data_path = Path("../data/brent_prices.csv")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Brent data not found (tried data/brent_prices.csv and ../data/brent_prices.csv)"
        )

    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"] = df["Date"].apply(lambda x: x.replace(day=1))
    df = df.set_index("Date").sort_index()

    if "brent" not in df.columns:
        raise ValueError("Column 'brent' not found in Brent data")

    df["brent"] = pd.to_numeric(df["brent"], errors="coerce")

    return df[["brent"]]


def load_macro_data():
    """Load macro data (Ki, USD) from enhanced_inflation_data.csv."""
    data_path = Path("data/enhanced_inflation_data.csv")

    if not data_path.exists():
        raise FileNotFoundError(f"Macro data not found: {data_path}")

    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    result = {}

    if "Ki_i" in df.columns:
        result["Ki"] = pd.to_numeric(df["Ki_i"], errors="coerce").to_frame()
    elif "Ki" in df.columns:
        result["Ki"] = pd.to_numeric(df["Ki"], errors="coerce").to_frame()

    if "usd_nom_i" in df.columns:
        result["USD"] = pd.to_numeric(df["usd_nom_i"], errors="coerce").to_frame()
    elif "usd" in df.columns:
        result["USD"] = pd.to_numeric(df["usd"], errors="coerce").to_frame()

    return result


def calculate_ccf(x, y, max_lag=12):
    """
    Calculate Cross-Correlation Function for lags 0 to max_lag.

    Args:
        x: First time series (regressor)
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


def analyze_regressor(cpi_df, regressor_df, regressor_name, max_lag=12):
    """
    Analyze a single regressor to find optimal lag.

    Args:
        cpi_df: DataFrame with CPI data
        regressor_df: DataFrame with regressor data
        regressor_name: Name of the regressor
        max_lag: Maximum lag to analyze

    Returns:
        Dict with analysis results
    """
    merged_df = pd.merge(
        cpi_df, regressor_df, left_index=True, right_index=True, how="inner"
    )

    if len(merged_df) == 0:
        return {
            "regressor": regressor_name,
            "error": "No overlapping data",
            "optimal_lag": None,
            "correlation": None,
            "data_points": 0,
        }

    regressor_col = regressor_df.columns[0]
    regressor_series = merged_df[regressor_col]
    cpi_series = merged_df["mom"]

    correlations = calculate_ccf(regressor_series, cpi_series, max_lag=max_lag)

    try:
        optimal_lag, max_correlation = find_optimal_lag(correlations)
    except ValueError:
        return {
            "regressor": regressor_name,
            "error": "No valid correlations",
            "optimal_lag": None,
            "correlation": None,
            "data_points": len(merged_df),
        }

    return {
        "regressor": regressor_name,
        "optimal_lag": optimal_lag,
        "correlation": max_correlation,
        "all_correlations": {str(k): v for k, v in correlations.items()},
        "data_points": len(merged_df),
        "date_range": {
            "start": str(merged_df.index.min()),
            "end": str(merged_df.index.max()),
        },
    }


def main():
    print("=" * 70)
    print("Optimal Lag Discovery for Macro Regressors")
    print("=" * 70)

    print("\nLoading CPI data...")
    cpi_df = load_inflation_data()
    print(f"  - CPI data shape: {cpi_df.shape}")
    print(f"  - Date range: {cpi_df.index.min()} to {cpi_df.index.max()}")

    print("\nLoading macro data (Ki, USD)...")
    macro_data = load_macro_data()
    for name, df in macro_data.items():
        print(f"  - {name} shape: {df.shape}")
        print(f"    Date range: {df.index.min()} to {df.index.max()}")

    print("\nLoading Brent data...")
    brent_df = load_brent_data()
    print(f"  - Brent data shape: {brent_df.shape}")
    print(f"  - Date range: {brent_df.index.min()} to {brent_df.index.max()}")

    regressors = {}
    regressors.update(macro_data)
    regressors["Brent"] = brent_df

    print("\n" + "=" * 70)
    print("Calculating Cross-Correlation Function (CCF) for lags 0-12...")
    print("=" * 70)

    results = {}

    for regressor_name, regressor_df in regressors.items():
        print(f"\nAnalyzing {regressor_name}...")
        result = analyze_regressor(cpi_df, regressor_df, regressor_name, max_lag=12)

        if result["optimal_lag"] is not None:
            print(f"  Optimal lag: {result['optimal_lag']} months")
            print(f"  Correlation: {result['correlation']:+.4f}")
            print(f"  Data points: {result['data_points']}")

            print(f"\n  Correlations by lag:")
            print("    Lag  |  Correlation")
            print("    " + "-" * 25)
            all_corr = result["all_correlations"]
            for lag in range(0, 13):
                corr = all_corr.get(str(lag))
                if corr is not None:
                    print(f"    {lag:3d}  |  {corr:+.4f}")
                else:
                    print(f"    {lag:3d}  |  N/A")
        else:
            print(f"  Error: {result.get('error', 'Unknown')}")

        regressor_key = regressor_name.lower()
        results[regressor_key] = {
            "optimal_lag": result["optimal_lag"],
            "correlation": result["correlation"],
        }

    print("\n" + "=" * 70)
    print("Summary of Optimal Lags")
    print("=" * 70)
    for regressor_key, values in results.items():
        regressor_name = regressor_key.upper()
        if values["optimal_lag"] is not None:
            print(
                f"  {regressor_name:>5}: lag {values['optimal_lag']:2d}, corr = {values['correlation']:+.4f}"
            )
        else:
            print(f"  {regressor_name:>5}: (no valid correlation found)")

    output_path = Path("data/optimal_lags.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
