#!/usr/bin/env python3
"""
Seasonal Pattern Evolution Analysis
Analyzes how KBR seasonality patterns evolved over 5-year windows.
Eras: 2010-2014, 2015-2019, 2020-2024
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

# Eras to analyze
ERAS = [
    (2010, 2014, "2010-2014 (Pre-Crimea)"),
    (2015, 2019, "2015-2019 (Post-Crimea)"),
    (2020, 2024, "2020-2024 (COVID & Recovery)"),
]


def load_data():
    """Load KBR inflation data."""
    df = pd.read_csv(DATA_DIR / "infl_kbr.csv", sep=";", decimal=",")

    # Filter for 'Все товары и услуги' (total CPI)
    df = df[df["Товар"] == "Все товары и услуги"].copy()

    # Parse date from Day column (DD.MM.YYYY format)
    df["Date"] = pd.to_datetime(df["Day"], format="%d.%m.%Y", errors="coerce")

    # Convert MoM to numeric (handle string values)
    df["MoM"] = pd.to_numeric(df["MoM"], errors="coerce")

    # Convert MoM index to percentage (101.49 -> 1.49%)
    df["inflation_pct"] = df["MoM"] - 100

    # Set date as index
    df = df.set_index("Date").sort_index()

    # Filter to 2010-2024 for complete eras
    df = df.loc["2010-01-01":"2024-12-31"]

    return df


def calculate_mean_indices(df, start_year, end_year):
    """Calculate simple mean seasonal indices for an era."""
    era_df = df.loc[f"{start_year}" : f"{end_year}"].copy()

    indices = []
    for month in range(1, 13):
        month_df = era_df[era_df.index.month == month]
        if len(month_df) > 0:
            mean_val = month_df["inflation_pct"].mean()
            median_val = month_df["inflation_pct"].median()
            std_val = month_df["inflation_pct"].std()
            year_count = len(month_df)
            indices.append(
                {
                    "Month": month,
                    "Mean_Index": mean_val,
                    "Median_Index": median_val,
                    "Std": std_val,
                    "Year_Count": year_count,
                }
            )

    return pd.DataFrame(indices)


def calculate_decomposition_indices(df, start_year, end_year):
    """Calculate seasonal indices using statsmodels decomposition."""
    era_df = df.loc[f"{start_year}" : f"{end_year}"].copy()

    # Need at least 24 points for decomposition
    if len(era_df) < 24:
        return None

    # Decompose using additive model
    decomposition = seasonal_decompose(
        era_df["inflation_pct"], model="additive", period=12
    )

    # Extract seasonal component (first 12 values represent seasonal pattern)
    seasonal = decomposition.seasonal.iloc[:12]

    # Map to months 1-12
    indices = []
    for month in range(1, 13):
        # Find seasonal value for this month
        # Seasonal pattern repeats every 12 months
        idx = (month - 1) % 12
        if idx < len(seasonal):
            indices.append({"Month": month, "Decompose_Index": seasonal.iloc[idx]})

    return pd.DataFrame(indices)


def compare_patterns(era1_df, era2_df, era1_name, era2_name):
    """Compare seasonal patterns between two eras."""
    # Calculate correlation between mean indices
    corr = era1_df["Mean_Index"].corr(era2_df["Mean_Index"])

    # Identify months with biggest changes
    merged = era1_df[["Month", "Mean_Index"]].merge(
        era2_df[["Month", "Mean_Index"]], on="Month", suffixes=("_Era1", "_Era2")
    )
    merged["Change"] = merged["Mean_Index_Era2"] - merged["Mean_Index_Era1"]
    merged["Abs_Change"] = merged["Change"].abs()
    merged = merged.sort_values("Abs_Change", ascending=False)

    return {"correlation": corr, "top_changes": merged.head(5)}


def analyze():
    """Main analysis function."""
    print("=" * 60)
    print("SEASONAL PATTERN EVOLUTION ANALYSIS")
    print("=" * 60)

    df = load_data()
    print(f"\nLoaded {len(df)} months of data (2010-2024)")

    # Store results for each era
    era_results = {}

    # Analyze each era
    for start_year, end_year, era_name in ERAS:
        print(f"\n{'=' * 60}")
        print(f"Era: {era_name}")
        print(f"{'=' * 60}")

        # Calculate mean indices
        mean_indices = calculate_mean_indices(df, start_year, end_year)
        decompose_indices = calculate_decomposition_indices(df, start_year, end_year)

        # Merge if decomposition available
        if decompose_indices is not None:
            indices = mean_indices.merge(decompose_indices, on="Month", how="left")
        else:
            indices = mean_indices

        indices["Era"] = era_name
        indices["Start_Year"] = start_year
        indices["End_Year"] = end_year

        era_results[era_name] = indices

        # Print summary
        print(f"\nTop 3 inflation months (Mean):")
        top3 = indices.nlargest(3, "Mean_Index")
        for _, row in top3.iterrows():
            print(
                f"  Month {row['Month']:2d}: {row['Mean_Index']:.2f}% (Std: {row['Std']:.2f})"
            )

        print(f"\nBottom 3 inflation months (Mean):")
        bottom3 = indices.nsmallest(3, "Mean_Index")
        for _, row in bottom3.iterrows():
            print(
                f"  Month {row['Month']:2d}: {row['Mean_Index']:.2f}% (Std: {row['Std']:.2f})"
            )

        # Special months
        jan = indices[indices["Month"] == 1].iloc[0]
        jul = indices[indices["Month"] == 7].iloc[0]
        dec = indices[indices["Month"] == 12].iloc[0]

        print(f"\nSpecial months:")
        print(f"  January:   {jan['Mean_Index']:.2f}% (Std: {jan['Std']:.2f})")
        print(f"  July (tariffs): {jul['Mean_Index']:.2f}% (Std: {jul['Std']:.2f})")
        print(f"  December:  {dec['Mean_Index']:.2f}% (Std: {dec['Std']:.2f})")

    # Combine all eras
    all_indices = pd.concat(era_results.values(), ignore_index=True)

    # Save to CSV
    output_path = OUTPUT_DIR / "seasonal_indices_by_era.csv"
    all_indices.to_csv(output_path, index=False)
    print(f"\n{'=' * 60}")
    print(f"Saved seasonal indices to {output_path}")
    print(f"{'=' * 60}")

    # Compare patterns between eras
    print("\n" + "=" * 60)
    print("PATTERN COMPARISON BETWEEN ERAS")
    print("=" * 60)

    era_names = [name for _, _, name in ERAS]

    for i in range(len(era_names)):
        for j in range(i + 1, len(era_names)):
            era1_name = era_names[i]
            era2_name = era_names[j]

            result = compare_patterns(
                era_results[era1_name], era_results[era2_name], era1_name, era2_name
            )

            print(f"\n{era1_name} vs {era2_name}:")
            print(f"  Correlation of mean patterns: {result['correlation']:.3f}")
            print(f"  Top 5 months with biggest changes:")
            for _, row in result["top_changes"].iterrows():
                month_name = row["Month"]
                change_val = row["Change"]
                print(
                    f"    Month {month_name}: {change_val:+.2f}pp "
                    f"({row['Mean_Index_Era1']:.2f}% -> {row['Mean_Index_Era2']:.2f}%)"
                )

    # Track evolution of key months across eras
    print("\n" + "=" * 60)
    print("EVOLUTION OF KEY MONTHS")
    print("=" * 60)

    key_months = {
        1: "January (New Year/Tariffs)",
        7: "July (Tariff Indexation)",
        12: "December (Pre-holiday)",
    }

    for month, description in key_months.items():
        print(f"\n{description}:")
        for era_name in era_names:
            era_df = era_results[era_name]
            month_row = era_df[era_df["Month"] == month].iloc[0]
            print(
                f"  {era_name}: {month_row['Mean_Index']:.2f}% (Std: {month_row['Std']:.2f})"
            )

    # Identify months with most/least stable patterns
    print("\n" + "=" * 60)
    print("STABILITY ANALYSIS (Std across eras)")
    print("=" * 60)

    # Calculate std of mean indices across eras for each month
    monthly_means = all_indices.pivot(index="Month", columns="Era", values="Mean_Index")
    monthly_std = monthly_means.std(axis=1)
    monthly_std = monthly_std.sort_values()

    print(f"\nMost stable months (lowest std):")
    for month in monthly_std.head(5).index:
        print(f"  Month {month}: Std={monthly_std[month]:.3f}")

    print(f"\nMost volatile months (highest std):")
    for month in monthly_std.tail(5).index:
        print(f"  Month {month}: Std={monthly_std[month]:.3f}")

    return all_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze seasonal pattern evolution")
    parser.add_argument("--input", default="data/infl_kbr.csv", help="Input CSV file")
    parser.add_argument(
        "--output", default="data/seasonal_indices_by_era.csv", help="Output CSV file"
    )
    args = parser.parse_args()

    analyze()
