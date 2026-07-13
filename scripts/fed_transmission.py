#!/usr/bin/env python3
"""
Federal Reserve Policy Transmission Analysis

Analyzes how US Federal Reserve policy transmits to KBR regional inflation.
Controls for USD/RUB exchange rate and Brent oil prices.
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")


def load_fed_funds_rate(data_dir: str) -> pd.DataFrame:
    """
    Load Federal Funds Rate data.

    Args:
        data_dir: Path to data directory

    Returns:
        DataFrame with Date index and Rate column
    """
    path = Path(data_dir) / "fed_funds_rate.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date")
    df = df.rename(columns={"Rate": "fed_rate"})
    return df


def load_kbr_inflation(data_dir: str) -> pd.DataFrame:
    """
    Load KBR inflation data with macro indicators.

    Args:
        data_dir: Path to data directory

    Returns:
        DataFrame with Date index and mom, usd_nom_i columns
    """
    path = Path(data_dir) / "inflation_data.csv"
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
    )

    # Parse date with dayfirst=True for format "31.01.2010"
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    df = df.set_index("Date")

    # Convert mom from index (101,49) to change (1,49)
    df["mom"] = pd.to_numeric(df["mom"], errors="coerce")
    df["mom_change"] = df["mom"] - 100

    # Convert USD from index to change
    if "usd_nom_i" in df.columns:
        df["usd_nom_i"] = pd.to_numeric(df["usd_nom_i"], errors="coerce")
        df["usd_change"] = df["usd_nom_i"] - 100

    return df


def load_brent(data_dir: str) -> pd.DataFrame:
    """
    Load Brent oil prices.

    Args:
        data_dir: Path to data directory

    Returns:
        DataFrame with Date index and brent column
    """
    path = Path(data_dir) / "brent_prices.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date")
    return df


def create_lags(
    df: pd.DataFrame, columns: List[str], max_lag: int = 12
) -> pd.DataFrame:
    """
    Create lagged versions of specified columns.

    Args:
        df: Input DataFrame
        columns: Columns to create lags for
        max_lag: Maximum lag to create

    Returns:
        DataFrame with lagged columns added
    """
    df_lagged = df.copy()

    for col in columns:
        for lag in range(1, max_lag + 1):
            df_lagged[f"{col}_L{lag}"] = df[col].shift(lag)

    return df_lagged


def align_data(
    fed_df: pd.DataFrame,
    kbr_df: pd.DataFrame,
    brent_df: pd.DataFrame,
    start_year: int = 2010,
) -> pd.DataFrame:
    """
    Align all data sources on common dates.

    Args:
        fed_df: Fed Funds Rate DataFrame
        kbr_df: KBR Inflation DataFrame
        brent_df: Brent DataFrame
        start_year: Start year for analysis

    Returns:
        Aligned DataFrame
    """
    # Forward fill and backward fill to handle different frequencies
    fed_ff = (
        fed_df.reindex(pd.date_range(fed_df.index.min(), fed_df.index.max(), freq="D"))
        .ffill()
        .asfreq("MS")
    )
    kbr_ff = (
        kbr_df.reindex(pd.date_range(kbr_df.index.min(), kbr_df.index.max(), freq="D"))
        .ffill()
        .asfreq("MS")
    )
    brent_ff = (
        brent_df.reindex(
            pd.date_range(brent_df.index.min(), brent_df.index.max(), freq="D")
        )
        .ffill()
        .asfreq("MS")
    )

    # Merge all data
    merged = fed_ff.join(kbr_ff, how="inner")
    merged = merged.join(brent_ff, how="inner")

    # Filter by date
    merged = merged[merged.index.year >= start_year]

    # Remove rows with missing target
    merged = merged.dropna(subset=["mom_change"])

    return merged


def cross_correlation(
    x: pd.Series, y: pd.Series, max_lag: int = 12
) -> Tuple[np.ndarray, int]:
    """
    Calculate cross-correlation between two series.

    Args:
        x: Series 1 (cause)
        y: Series 2 (effect)
        max_lag: Maximum lag to test

    Returns:
        Tuple of (correlations array, optimal lag)
    """
    correlations = []

    for lag in range(max_lag + 1):
        # Shift x by lag
        x_lagged = x.shift(lag)

        # Align and calculate correlation
        aligned = pd.concat([x_lagged, y], axis=1).dropna()
        if len(aligned) < 10:
            corr = np.nan
        else:
            corr = aligned.corr().iloc[0, 1]
        correlations.append(corr)

    correlations = np.array(correlations)

    # Find optimal lag (highest absolute correlation, excluding lag 0)
    abs_corr = np.abs(correlations[1:])  # Exclude lag 0
    if len(abs_corr) > 0:
        optimal_lag = np.argmax(abs_corr) + 1  # +1 because we excluded lag 0
    else:
        optimal_lag = 0

    return correlations, optimal_lag


def run_regression(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    lag: int,
) -> Dict:
    """
    Run linear regression for a specific lag.

    Args:
        df: DataFrame with lagged features
        target_col: Target column name
        feature_cols: Feature column names (without lag suffix)
        lag: Lag to use

    Returns:
        Dictionary with regression results
    """
    # Construct feature columns with lag
    lagged_features = [f"{col}_L{lag}" for col in feature_cols]

    # Prepare data
    X = df[lagged_features].copy()
    y = df[target_col].copy()

    # Drop missing values
    data = pd.concat([X, y], axis=1).dropna()

    if len(data) < 20:
        return {
            "lag": lag,
            "n_obs": 0,
            "r2": np.nan,
            "coefficients": {},
            "p_values": {},
            "fed_coefficient": np.nan,
            "fed_p_value": np.nan,
            "significant": False,
            "intercept": np.nan,
        }

    X = data[lagged_features]
    y = data[target_col]

    # Run regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Calculate R²
    r2 = r2_score(y, y_pred)

    # Calculate p-values using t-test
    n = len(y)
    k = len(feature_cols)
    dof = n - k - 1

    p_values = {}
    residuals = y - y_pred
    mse = np.sum(residuals**2) / dof
    se = np.sqrt(mse * np.diag(np.linalg.inv(X.T @ X)))

    for i, feat in enumerate(lagged_features):
        t_stat = model.coef_[i] / se[i]
        p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), dof))
        p_values[feat] = p_val

    # Check if Fed coefficient is significant
    fed_feat = f"{feature_cols[0]}_L{lag}"
    is_sig = p_values.get(fed_feat, 1.0) < 0.05

    return {
        "lag": lag,
        "n_obs": n,
        "r2": r2,
        "coefficients": dict(zip(lagged_features, model.coef_)),
        "intercept": model.intercept_,
        "p_values": p_values,
        "fed_coefficient": model.coef_[0],
        "fed_p_value": p_values.get(fed_feat, 1.0),
        "significant": is_sig,
    }


def granger_causality_test(
    df: pd.DataFrame,
    cause_col: str,
    effect_col: str,
    max_lag: int = 6,
) -> float:
    """
    Run Granger causality test.

    Args:
        df: DataFrame with aligned data
        cause_col: Cause column
        effect_col: Effect column
        max_lag: Maximum lag to test

    Returns:
        p-value from Granger test
    """
    # Prepare data
    data = df[[cause_col, effect_col]].dropna()

    if len(data) < 20:
        return np.nan

    try:
        result = grangercausalitytests(data.values, maxlag=max_lag, verbose=False)
        # Get p-value from F-test for maxlag
        p_value = result[max_lag][0]["ssr_ftest"][1]
        return p_value
    except Exception:
        return np.nan


def run_analysis(
    df: pd.DataFrame,
    max_lag: int = 12,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run full transmission analysis.

    Tests 3 models:
    1. Simple: CPI ~ Fed(lag)
    2. Controlled: CPI ~ Fed(lag) + USD(lag)
    3. Full: CPI ~ Fed(lag) + USD(lag) + Brent(lag)

    Args:
        df: Aligned DataFrame with all features
        max_lag: Maximum lag to test

    Returns:
        Tuple of (results DataFrame, summary dict)
    """
    # Create lags
    feature_cols = ["fed_rate", "usd_change", "brent"]
    df_lagged = create_lags(df, feature_cols, max_lag)

    results = []

    # Test Model 1: Simple (Fed only)
    print("\nModel 1: Simple (Fed Rate only)")
    print("-" * 50)
    for lag in range(1, max_lag + 1):
        result = run_regression(df_lagged, "mom_change", ["fed_rate"], lag)
        result["model"] = "Simple"
        results.append(result)
        print(
            f"  Lag {lag:2d}: R²={result['r2']:.3f}, "
            f"Coef={result['fed_coefficient']:.6f}, "
            f"p={result['fed_p_value']:.3f} {'✓' if result['significant'] else ''}"
        )

    # Test Model 2: Controlled (Fed + USD)
    print("\nModel 2: Controlled (Fed Rate + USD)")
    print("-" * 50)
    for lag in range(1, max_lag + 1):
        result = run_regression(
            df_lagged, "mom_change", ["fed_rate", "usd_change"], lag
        )
        result["model"] = "Controlled"
        results.append(result)
        print(
            f"  Lag {lag:2d}: R²={result['r2']:.3f}, "
            f"Coef={result['fed_coefficient']:.6f}, "
            f"p={result['fed_p_value']:.3f} {'✓' if result['significant'] else ''}"
        )

    # Test Model 3: Full (Fed + USD + Brent)
    print("\nModel 3: Full (Fed Rate + USD + Brent)")
    print("-" * 50)
    for lag in range(1, max_lag + 1):
        result = run_regression(
            df_lagged, "mom_change", ["fed_rate", "usd_change", "brent"], lag
        )
        result["model"] = "Full"
        results.append(result)
        print(
            f"  Lag {lag:2d}: R²={result['r2']:.3f}, "
            f"Coef={result['fed_coefficient']:.6f}, "
            f"p={result['fed_p_value']:.3f} {'✓' if result['significant'] else ''}"
        )

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Find optimal lag for each model
    summary = {}
    for model_name in ["Simple", "Controlled", "Full"]:
        model_results = results_df[results_df["model"] == model_name]

        # Best R²
        best_r2_idx = model_results["r2"].idxmax()
        best_r2 = model_results.loc[best_r2_idx]

        # Most significant
        sig_results = model_results[model_results["significant"]]
        if len(sig_results) > 0:
            best_sig_idx = sig_results["fed_p_value"].idxmin()
            best_sig = sig_results.loc[best_sig_idx]
        else:
            best_sig = None

        summary[model_name] = {
            "best_r2_lag": int(best_r2["lag"]),
            "best_r2": best_r2["r2"],
            "best_r2_coeff": best_r2["fed_coefficient"],
            "best_r2_p": best_r2["fed_p_value"],
            "best_sig_lag": int(best_sig["lag"]) if best_sig is not None else None,
            "best_sig_p": best_sig["fed_p_value"] if best_sig is not None else None,
        }

    return results_df, summary


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Fed policy transmission to KBR inflation"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Path to data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/fed_transmission_results.csv",
        help="Output path for results",
    )
    parser.add_argument(
        "--max-lag", type=int, default=12, help="Maximum lag to test (months)"
    )
    parser.add_argument(
        "--start-year", type=int, default=2010, help="Start year for analysis"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Federal Reserve Policy Transmission Analysis")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    fed_df = load_fed_funds_rate(args.data_dir)
    print(f"   Fed Funds Rate: {len(fed_df)} observations")

    kbr_df = load_kbr_inflation(args.data_dir)
    print(f"   KBR Inflation: {len(kbr_df)} observations")

    brent_df = load_brent(args.data_dir)
    print(f"   Brent Oil: {len(brent_df)} observations")

    # Align data
    print("\n2. Aligning data...")
    aligned = align_data(fed_df, kbr_df, brent_df, start_year=args.start_year)
    print(f"   Aligned dataset: {len(aligned)} observations")
    print(f"   Date range: {aligned.index.min()} to {aligned.index.max()}")

    # Cross-correlation analysis
    print("\n3. Cross-correlation analysis (Fed Rate → KBR CPI)...")
    corr, opt_lag = cross_correlation(
        aligned["fed_rate"], aligned["mom_change"], args.max_lag
    )
    print(f"   Optimal lag: {opt_lag} months")
    print(f"   Correlation at optimal lag: {corr[opt_lag]:.4f}")

    # Granger causality test
    print("\n4. Granger causality test...")
    granger_p = granger_causality_test(aligned, "fed_rate", "mom_change", max_lag=6)
    print(f"   Granger p-value (lag=6): {granger_p:.4f}")
    print(f"   Significant at 5%: {'Yes ✓' if granger_p < 0.05 else 'No'}")

    # Run regression analysis
    print("\n5. Running regression analysis...")
    results_df, summary = run_analysis(aligned, max_lag=args.max_lag)

    # Save results
    print("\n6. Saving results...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"   Results saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nOptimal Lags by Model:")
    print("-" * 70)
    for model_name, data in summary.items():
        print(f"\n{model_name} Model:")
        print(f"  Best R²: Lag {data['best_r2_lag']}, R²={data['best_r2']:.4f}")
        print(f"  Coefficient at best R²: {data['best_r2_coeff']:.6f}")
        print(f"  P-value at best R²: {data['best_r2_p']:.4f}")
        if data["best_sig_lag"] is not None:
            print(
                f"  Most Significant: Lag {data['best_sig_lag']}, p={data['best_sig_p']:.4f}"
            )
        else:
            print(f"  Most Significant: None (not significant at any lag)")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Find best overall model
    best_model_idx = results_df["r2"].idxmax()
    best_result = results_df.loc[best_model_idx]

    print(f"\nBest Model: {best_result['model']}")
    print(f"Lag: {int(best_result['lag'])} months")
    print(f"R²: {best_result['r2']:.4f}")
    print(f"Fed Rate Coefficient: {best_result['fed_coefficient']:.6f}")
    print(f"Significance: p = {best_result['fed_p_value']:.4f}")

    if best_result["significant"]:
        print(
            "\n✓ Fed rate changes have a statistically significant effect on KBR inflation"
        )
        print(f"  (lag = {int(best_result['lag'])} months, p < 0.05)")
    else:
        print("\n✗ Fed rate changes do NOT have a statistically significant effect")
        print("  on KBR inflation after controlling for USD and Brent")

    print("\n✓ Analysis complete!")

    # Return for import testing
    return results_df, summary


if __name__ == "__main__":
    main()
