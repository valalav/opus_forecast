#!/usr/bin/env python3
"""
Seasonality by Product Group Analysis
Analyzes seasonal patterns for Food, Non-Food, and Services components.
Output: Monthly seasonal factors for each product group.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose
import argparse

# Paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Product groups to analyze
PRODUCT_GROUPS = [
    "Продовольственные товары",
    "Непродовольственные товары",
    "Услуги",
]


def load_data():
    """Load KBR inflation data."""
    df = pd.read_csv(DATA_DIR / "infl_kbr.csv", sep=";", decimal=",")

    # Filter for product groups only
    df = df[df["Товар"].isin(PRODUCT_GROUPS)].copy()

    # Parse date from Day column (DD.MM.YYYY format)
    df["Date"] = pd.to_datetime(df["Day"], format="%d.%m.%Y", errors="coerce")

    # Convert MoM to numeric (handle string values)
    df["MoM"] = pd.to_numeric(df["MoM"], errors="coerce")

    # Convert MoM index to percentage (101.49 -> 1.49%)
    df["inflation_pct"] = df["MoM"] - 100

    # Set date as index
    df = df.set_index("Date").sort_index()

    # Filter to 2010-2025 for complete seasonal patterns
    df = df.loc["2010-01-01":"2025-12-31"]

    return df


def calculate_seasonal_factors(group_df, group_name):
    """Calculate seasonal factors using statsmodels decomposition."""
    # Need at least 24 points for decomposition
    if len(group_df) < 24:
        return None

    # Decompose using additive model
    try:
        decomposition = seasonal_decompose(
            group_df["inflation_pct"], model="additive", period=12
        )

        # Extract seasonal component (first 12 values represent seasonal pattern)
        seasonal = decomposition.seasonal.iloc[:12]

        # Create output dataframe
        indices = []
        for month in range(1, 13):
            # Find seasonal value for this month
            idx = (month - 1) % 12
            if idx < len(seasonal):
                indices.append(
                    {
                        "Group": group_name,
                        "Month": month,
                        "Month_Name": pd.Timestamp(2025, month, 1).strftime("%B"),
                        "Seasonal_Factor": seasonal.iloc[idx],
                    }
                )

        return pd.DataFrame(indices)

    except Exception as e:
        print(f"Error decomposing {group_name}: {e}")
        return None


def calculate_monthly_averages(df):
    """Calculate simple monthly averages for each group."""
    results = []

    for group_name in PRODUCT_GROUPS:
        group_df = df[df["Товар"] == group_name].copy()

        if len(group_df) == 0:
            continue

        # Calculate average inflation by month across all years
        monthly_stats = []
        for month in range(1, 13):
            month_df = group_df[group_df.index.month == month]

            if len(month_df) > 0:
                mean_val = month_df["inflation_pct"].mean()
                median_val = month_df["inflation_pct"].median()
                std_val = month_df["inflation_pct"].std()
                year_count = len(month_df)

                monthly_stats.append(
                    {
                        "Group": group_name,
                        "Month": month,
                        "Month_Name": pd.Timestamp(2025, month, 1).strftime("%B"),
                        "Mean_Inflation": mean_val,
                        "Median_Inflation": median_val,
                        "Std": std_val,
                        "Year_Count": year_count,
                    }
                )

        if monthly_stats:
            results.extend(monthly_stats)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Analyze seasonality by product group")
    parser.add_argument(
        "--output",
        type=str,
        default="data/group_seasonality.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Seasonality by Product Group Analysis")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"  Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")

    # Calculate monthly averages
    print("\nCalculating monthly averages...")
    monthly_avg = calculate_monthly_averages(df)

    # Calculate seasonal factors using decomposition
    print("Calculating seasonal factors...")
    seasonal_results = []

    for group_name in PRODUCT_GROUPS:
        group_df = df[df["Товар"] == group_name].copy()

        if len(group_df) > 0:
            print(f"  Processing {group_name} ({len(group_df)} records)...")
            seasonal_df = calculate_seasonal_factors(group_df, group_name)

            if seasonal_df is not None:
                seasonal_results.append(seasonal_df)

    # Merge seasonal factors with monthly averages
    if seasonal_results:
        seasonal_df = pd.concat(seasonal_results, ignore_index=True)
        final_df = pd.merge(
            monthly_avg,
            seasonal_df[["Group", "Month", "Seasonal_Factor"]],
            on=["Group", "Month"],
            how="left",
        )

        # Add ranking of seasonal intensity
        final_df["Seasonal_Intensity_Rank"] = (
            final_df.groupby("Group")["Seasonal_Factor"]
            .rank(method="dense", ascending=False)
            .astype(int)
        )

        # Save output
        output_path = Path(args.output)
        final_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"\n✓ Saved results to {output_path}")

        # Print summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY: Top Seasonal Months by Group")
        print("=" * 60)

        for group_name in PRODUCT_GROUPS:
            group_df = final_df[final_df["Group"] == group_name]
            top_months = group_df.nlargest(3, "Seasonal_Factor")
            bottom_months = group_df.nsmallest(3, "Seasonal_Factor")

            print(f"\n{group_name}:")
            print(f"  Highest seasonality:")
            for _, row in top_months.iterrows():
                print(f"    {row['Month_Name']:10s} {row['Seasonal_Factor']:7.3f}%")
            print(f"  Lowest seasonality:")
            for _, row in bottom_months.iterrows():
                print(f"    {row['Month_Name']:10s} {row['Seasonal_Factor']:7.3f}%")

        # Overall seasonal intensity
        print("\n" + "=" * 60)
        print("SEASONAL INTENSITY (Std of Seasonal Factors)")
        print("=" * 60)
        for group_name in PRODUCT_GROUPS:
            group_df = final_df[final_df["Group"] == group_name]
            intensity = group_df["Seasonal_Factor"].std()
            mean_seas = group_df["Seasonal_Factor"].mean()
            print(f"  {group_name}: {intensity:.4f} (mean: {mean_seas:.4f})")

    else:
        print("ERROR: No seasonal factors calculated")
        return 1

    return 0


if __name__ == "__main__":
    main()
