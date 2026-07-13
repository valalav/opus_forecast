#!/usr/bin/env python3
"""
Task 116: Correlation & Regressor Ranking
Analyzes 500+ series to find 'Gold Regressors' for inflation forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings("ignore")


def load_data():
    """Load all required data files."""
    data_dir = Path("data")

    # Load macro monolith data
    macro_df = pd.read_csv(data_dir / "kbr_macro_monolith.csv")
    macro_df["Date"] = pd.to_datetime(macro_df["Date"])

    # Load sectoral details data (using test_kbr_sectoral.csv which has more data)
    sectoral_df = pd.read_csv(data_dir / "test_kbr_sectoral.csv")
    sectoral_df["Date"] = pd.to_datetime(sectoral_df["Date"])

    return macro_df, sectoral_df


def extract_target_inflation(sectoral_df):
    """Extract target inflation (ИПЦ - All goods and services)."""
    # Filter for the main CPI indicator (ИПЦ, Все товары и услуги и БИПЦ)
    target_df = sectoral_df[
        sectoral_df["Indicator"].str.contains("ИПЦ.*Все товары", na=False)
        & (sectoral_df["Metric_Type"] == "г/г")
    ].copy()

    # Ensure we have one value per date (keep most recent sheet if duplicates)
    target_df = target_df.sort_values(["Date", "Sheet"], ascending=[True, False])
    target_df = target_df.drop_duplicates(subset=["Date"], keep="first")
    target_df = target_df.sort_values("Date")

    return target_df[["Date", "Value"]].rename(columns={"Value": "target_inflation"})


def pivot_series_to_features(df, value_col="Value", feature_cols=None):
    """
    Pivot time series data into feature columns.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with Date, Value, and feature identifier columns
    value_col : str
        Name of the value column
    feature_cols : list
        List of columns that identify unique features

    Returns:
    --------
    pd.DataFrame
        Pivoted DataFrame with dates as index and features as columns
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in ["Date", value_col]]

    # Create feature identifier
    df = df.copy()
    df["feature_id"] = df[feature_cols].astype(str).agg("::".join, axis=1)

    # Handle duplicates: for each date-feature_id pair, keep the first value
    df_agg = df.groupby(["Date", "feature_id"])[value_col].first().reset_index()

    # Pivot to wide format
    pivoted = df_agg.pivot(index="Date", columns="feature_id", values=value_col)

    return pivoted


def calculate_lagged_correlation(target_series, feature_series, max_lag=6):
    """
    Calculate Pearson correlation for all lags (0 to max_lag).

    Returns the maximum absolute correlation and the optimal lag.
    """
    best_corr = 0
    best_lag = 0
    best_corr_value = 0

    for lag in range(max_lag + 1):
        # Shift feature series by lag months
        shifted_feature = feature_series.shift(lag)

        # Align and remove NaN pairs
        aligned = pd.DataFrame(
            {"target": target_series, "feature": shifted_feature}
        ).dropna()

        if len(aligned) < 10:  # Need at least 10 data points
            continue

        try:
            corr, _ = pearsonr(aligned["target"], aligned["feature"])

            # Track best absolute correlation
            if abs(corr) > abs(best_corr_value):
                best_corr_value = corr
                best_lag = lag
                best_corr = abs(corr)
        except:
            continue

    return best_corr, best_corr_value, best_lag


def analyze_correlations(macro_df, sectoral_df, target_inflation):
    """
    Analyze correlations between all features and target inflation.
    """
    # Pivot macro data
    macro_pivot = pivot_series_to_features(
        macro_df,
        value_col="Value",
        feature_cols=["Indicator", "Category", "Metric_Type"],
    )

    # Pivot sectoral data
    sectoral_pivot = pivot_series_to_features(
        sectoral_df,
        value_col="Value",
        feature_cols=["Indicator", "Metric_Type", "Sheet"],
    )

    # Merge all features with target
    target_inflation = target_inflation.set_index("Date")

    all_features = pd.concat([macro_pivot, sectoral_pivot], axis=1)

    # Remove features that are essentially the same as target (ИПЦ)
    target_cols = [col for col in all_features.columns if "ИПЦ" in col]
    for col in target_cols:
        if col in all_features.columns:
            all_features = all_features.drop(columns=[col])

    results = []
    rejected_series = []

    for feature_name in all_features.columns:
        feature_series = all_features[feature_name]

        # Calculate missing data percentage
        missing_pct = feature_series.isna().sum() / len(feature_series) * 100

        # Filter out series with >20% missing data
        if missing_pct > 20:
            rejected_series.append(
                {
                    "feature_id": feature_name,
                    "missing_pct": missing_pct,
                    "total_obs": len(feature_series),
                    "non_missing_obs": feature_series.notna().sum(),
                    "rejection_reason": "Missing data > 20%",
                }
            )
            continue

        # Calculate lagged correlations
        corr, corr_value, lag = calculate_lagged_correlation(
            target_inflation["target_inflation"], feature_series, max_lag=6
        )

        if corr > 0:  # Only keep features with valid correlations
            results.append(
                {
                    "feature_id": feature_name,
                    "correlation": corr_value,
                    "abs_correlation": corr,
                    "optimal_lag": lag,
                    "missing_pct": missing_pct,
                    "non_missing_obs": feature_series.notna().sum(),
                }
            )

    return pd.DataFrame(results), pd.DataFrame(rejected_series)


def create_priority_list(results_df):
    """Create ranked priority list of regressors."""
    # Filter for correlation > 0.3 and get top 20
    priority = (
        results_df[(results_df["abs_correlation"] > 0.3)]
        .sort_values("abs_correlation", ascending=False)
        .head(20)
    )

    # Add rank
    priority["rank"] = range(1, len(priority) + 1)

    # Reorder columns
    priority = priority[
        [
            "rank",
            "feature_id",
            "correlation",
            "abs_correlation",
            "optimal_lag",
            "missing_pct",
            "non_missing_obs",
        ]
    ]

    return priority


def main():
    """Main execution function."""
    print("Task 116: Intelligence - Correlation & Regressor Ranking")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/5] Loading data...")
    macro_df, sectoral_df = load_data()
    print(f"  - Macro monolith: {len(macro_df)} rows")
    print(f"  - Sectoral details: {len(sectoral_df)} rows")

    # Step 2: Extract target inflation
    print("\n[2/5] Extracting target inflation...")
    target_inflation = extract_target_inflation(sectoral_df)
    print(f"  - Target series: {len(target_inflation)} observations")
    print(
        f"  - Date range: {target_inflation['Date'].min()} to {target_inflation['Date'].max()}"
    )

    # Step 3: Analyze correlations
    print("\n[3/5] Calculating lagged correlations (0-6 months)...")
    results_df, rejected_df = analyze_correlations(
        macro_df, sectoral_df, target_inflation
    )
    print(f"  - Features analyzed: {len(results_df) + len(rejected_df)}")
    print(f"  - Features accepted: {len(results_df)}")
    print(f"  - Features rejected (missing > 20%): {len(rejected_df)}")

    # Step 4: Create priority list
    print("\n[4/5] Creating regressor priority list...")
    priority_df = create_priority_list(results_df)
    print(f"  - Top-20 regressors with correlation > 0.3: {len(priority_df)}")

    # Step 5: Save outputs
    print("\n[5/5] Saving outputs...")

    # Save priority list
    output_path = Path("data/regressor_priority_list.csv")
    priority_df.to_csv(output_path, index=False)
    print(f"  - Saved: {output_path}")

    # Save missing data report
    rejected_path = Path("data/missing_data_report.csv")
    rejected_df.to_csv(rejected_path, index=False)
    print(f"  - Saved: {rejected_path}")

    # Display top 10 regressors
    print("\n" + "=" * 60)
    print("TOP 10 REGRESSORS (by absolute correlation)")
    print("=" * 60)
    print(
        priority_df[["rank", "feature_id", "correlation", "optimal_lag"]]
        .head(10)
        .to_string(index=False)
    )

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total features analyzed: {len(results_df) + len(rejected_df)}")
    print(f"Features with correlation > 0.3: {len(priority_df)}")
    print(f"Highest correlation: {priority_df['abs_correlation'].max():.4f}")
    print(f"Median correlation (top 20): {priority_df['abs_correlation'].median():.4f}")

    return len(priority_df) >= 1


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Task 116 COMPLETED SUCCESSFULLY")
    else:
        print("\n❌ Task 116 FAILED - No high-correlation regressors found")
