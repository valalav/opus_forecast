#!/usr/bin/env python3
"""
Trimmed Mean CPI Calculation for KBR

Excludes top/bottom 10% of price changes each month to reduce noise.
This is a standard Central Bank metric for measuring core inflation.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_subcomponent_data(filepath):
    """
    Load subcomponent MoM data from CSV.

    Args:
        filepath: Path to sub_mom.csv

    Returns:
        DataFrame with Date and subcomponent columns
    """
    # Read with comma as decimal separator
    df = pd.read_csv(filepath, sep=";", encoding="utf-8-sig", decimal=",")

    # Clean column names (strip whitespace)
    df.columns = [col.strip() for col in df.columns]

    # Parse dates (format: 31.01.2010) and convert to month-start
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
    df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    # Convert numeric columns (replace empty strings with NaN)
    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col].replace("", np.nan), errors="coerce")

    return df


def load_headline_cpi(filepath):
    """
    Load Headline CPI data.

    Args:
        filepath: Path to infl_kbr.csv

    Returns:
        DataFrame with Date and MoM for 'Все товары и услуги'
    """
    df = pd.read_csv(filepath, sep=";", encoding="utf-8-sig")
    df.columns = [col.strip() for col in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")

    # Filter to 'Все товары и услуги'
    headline = df[df["Товар"] == "Все товары и услуги"].copy()
    headline = headline[["Date", "MoM"]].rename(columns={"MoM": "Headline_CPI"})

    return headline


def calculate_trimmed_mean(df_subcomp, trim_percent=0.10):
    """
    Calculate Trimmed Mean CPI for each month.

    Args:
        df_subcomp: DataFrame with subcomponent data
        trim_percent: Percentage to trim from each tail (default 10%)

    Returns:
        DataFrame with Date and Trimmed_Mean columns
    """
    results = []

    # Get subcomponent columns (exclude Date)
    subcomp_cols = [col for col in df_subcomp.columns if col.isdigit()]
    n_components = len(subcomp_cols)
    n_to_trim = int(n_components * trim_percent)

    print(f"Total subcomponents: {n_components}")
    print(f"Trimming: {n_to_trim} from top, {n_to_trim} from bottom")

    for idx, row in df_subcomp.iterrows():
        date = row["Date"]

        # Get all subcomponent MoM values for this month as float
        values = row[subcomp_cols].values.astype(float)

        # Values are already in percentage format (no conversion needed)
        values_pct = values

        # Remove NaN values
        mask = ~np.isnan(values_pct)
        values_pct = values_pct[mask]

        if len(values_pct) == 0:
            continue

        # Sort values
        values_sorted = np.sort(values_pct)

        # Trim top and bottom 10%
        trimmed = values_sorted[n_to_trim : len(values_sorted) - n_to_trim]

        # Calculate mean of remaining
        trimmed_mean_pct = np.mean(trimmed)

        results.append({"Date": date, "Trimmed_Mean_CPI": trimmed_mean_pct})

    return pd.DataFrame(results)


def compare_stability(df):
    """
    Compare volatility of Trimmed Mean vs Headline CPI.

    Args:
        df: Merged DataFrame with both series

    Returns:
        dict with comparison metrics
    """
    # Calculate volatility (standard deviation)
    headline_std = df["Headline_CPI_Pct"].std()
    trimmed_std = df["Trimmed_Mean_CPI"].std()

    # Calculate MAE from mean
    headline_mae = np.abs(df["Headline_CPI_Pct"]).mean()
    trimmed_mae = np.abs(df["Trimmed_Mean_CPI"]).mean()

    improvement = ((headline_std - trimmed_std) / headline_std) * 100

    metrics = {
        "headline_std": headline_std,
        "trimmed_std": trimmed_std,
        "headline_mae": headline_mae,
        "trimmed_mae": trimmed_mae,
        "improvement_pct": improvement,
    }

    return metrics


def main():
    """Main execution."""
    # Define paths
    base_dir = Path("/home/valalav/_projects/sirena-kbr")
    sub_mom_file = base_dir / "data/raw/sub_mom.csv"
    infl_file = base_dir / "data/infl_kbr.csv"
    output_file = base_dir / "data/trimmed_mean_cpi.csv"

    print("=" * 60)
    print("Trimmed Mean CPI Calculation for KBR")
    print("=" * 60)

    # Load data
    print(f"\n1. Loading subcomponent data from: {sub_mom_file}")
    df_sub = load_subcomponent_data(sub_mom_file)
    print(
        f"   Loaded {len(df_sub)} months ({df_sub['Date'].min()} to {df_sub['Date'].max()})"
    )

    print(f"\n2. Loading headline CPI data from: {infl_file}")
    df_headline = load_headline_cpi(infl_file)
    print(f"   Loaded {len(df_headline)} months")

    # Calculate trimmed mean
    print(f"\n3. Calculating Trimmed Mean (10% trim)...")
    df_trimmed = calculate_trimmed_mean(df_sub, trim_percent=0.10)
    print(f"   Calculated for {len(df_trimmed)} months")

    # Merge datasets
    print(f"\n4. Merging datasets...")
    df_merged = pd.merge(df_headline, df_trimmed, on="Date", how="inner")

    # Convert headline to percentage for comparison
    df_merged["Headline_CPI_Pct"] = df_merged["Headline_CPI"] - 100

    # Compare stability
    print(f"\n5. Comparing volatility...")
    metrics = compare_stability(df_merged)

    print("\n" + "=" * 60)
    print("VOLATILITY COMPARISON")
    print("=" * 60)
    print(f"Headline CPI Std Dev:  {metrics['headline_std']:.4f}%")
    print(f"Trimmed Mean Std Dev:  {metrics['trimmed_std']:.4f}%")
    print(f"Improvement:           {metrics['improvement_pct']:+.1f}%")
    print(f"\nHeadline CPI MAE:      {metrics['headline_mae']:.4f}%")
    print(f"Trimmed Mean MAE:       {metrics['trimmed_mae']:.4f}%")

    # Check if volatility is lower
    if metrics["trimmed_std"] < metrics["headline_std"]:
        print(f"\n✅ Volatility of Trimmed Mean is LOWER than Headline CPI!")
    else:
        print(f"\n⚠️  Volatility of Trimmed Mean is HIGHER than Headline CPI")

    print("=" * 60)

    # Save results
    print(f"\n6. Saving results to: {output_file}")
    df_out = df_merged[["Date", "Headline_CPI", "Trimmed_Mean_CPI"]].copy()
    df_out.to_csv(output_file, index=False, sep=";", float_format="%.4f")
    print(f"   Saved {len(df_out)} rows")

    # Print sample
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT (last 5 months)")
    print("=" * 60)
    print(df_out.tail().to_string(index=False))

    print("\n✅ Trimmed Mean CPI calculation complete!")

    return metrics


if __name__ == "__main__":
    metrics = main()
