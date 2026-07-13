#!/usr/bin/env python3
"""
Diffusion Index Calculation for KBR

Calculates the percentage of products with rising prices each month.
Diffusion Index measures how broad-based inflation is:
- >50%: Broad-based (prices rising across many categories)
- <50%: Narrow (price increases concentrated in few categories)

This is a leading indicator of inflation persistence.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_subcomponent_mom(filepath):
    """
    Load subcomponent month-over-month data.

    Args:
        filepath: Path to sub_mom.csv

    Returns:
        DataFrame with monthly MoM changes for all subcomponents
    """
    print(f"Loading subcomponent data from: {filepath}")

    # Load with Russian decimal separator and semicolon delimiter
    df = pd.read_csv(filepath, sep=";", decimal=",")

    # Parse date column
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.set_index("Date")

    # Drop empty columns and last column (0 values)
    df = df.loc[:, (df != "").any(axis=0)]
    if "0" in df.columns:
        df = df.drop(columns=["0"])

    # Convert all values to numeric, keeping NaNs
    df = df.apply(pd.to_numeric, errors="coerce")

    print(f"   Loaded {len(df)} months")
    print(f"   Found {len(df.columns)} subcomponents")

    return df


def calculate_diffusion_index(df):
    """
    Calculate diffusion index: % of products with rising prices.

    Args:
        df: DataFrame with monthly MoM changes (components as columns)

    Returns:
        DataFrame with Date, diffusion_index, classification, diffusion_ma3
    """
    print("\nCalculating diffusion index...")

    results = []

    for date, row in df.iterrows():
        # Get non-NaN values
        values = row.dropna()

        if len(values) == 0:
            continue

        # Count products with rising prices (MoM > 0)
        n_rising = (values > 0).sum()
        n_total = len(values)

        # Diffusion index as percentage
        diffusion_idx = (n_rising / n_total) * 100

        # Classification
        classification = "broad-based" if diffusion_idx >= 50 else "narrow"

        results.append(
            {
                "Date": date,
                "diffusion_index": diffusion_idx,
                "n_rising": n_rising,
                "n_total": n_total,
                "classification": classification,
            }
        )

    # Create DataFrame
    result_df = pd.DataFrame(results)
    result_df = result_df.set_index("Date")

    # Calculate 3-month moving average
    result_df["diffusion_ma3"] = (
        result_df["diffusion_index"].rolling(window=3, min_periods=1).mean()
    )

    print(f"   Calculated for {len(result_df)} months")
    print(
        f"   Diffusion range: {result_df['diffusion_index'].min():.1f}% to {result_df['diffusion_index'].max():.1f}%"
    )
    print(f"   Average diffusion: {result_df['diffusion_index'].mean():.1f}%")

    # Count broad-based vs narrow months
    n_broad = (result_df["diffusion_index"] >= 50).sum()
    n_narrow = (result_df["diffusion_index"] < 50).sum()
    print(
        f"   Broad-based months (>50%): {n_broad} ({n_broad / len(result_df) * 100:.1f}%)"
    )
    print(
        f"   Narrow months (<50%): {n_narrow} ({n_narrow / len(result_df) * 100:.1f}%)"
    )

    return result_df


def analyze_inflation_phases(df):
    """
    Analyze how diffusion index relates to inflation phases.

    Args:
        df: DataFrame with diffusion index
    """
    print("\n=== Inflation Phase Analysis ===")

    # Calculate standard deviation of diffusion index
    diffusion_std = df["diffusion_index"].std()
    print(f"Diffusion Index Std Dev: {diffusion_std:.2f}%")

    # Identify periods of high diffusion (> mean + 1 std)
    high_diffusion_threshold = df["diffusion_index"].mean() + diffusion_std
    print(f"High Diffusion Threshold (> {high_diffusion_threshold:.1f}%):")

    high_diffusion_periods = df[df["diffusion_index"] > high_diffusion_threshold]
    if len(high_diffusion_periods) > 0:
        print(f"   {len(high_diffusion_periods)} months of high diffusion")
        print(f"   First: {high_diffusion_periods.index[0].strftime('%Y-%m')}")
        print(f"   Last: {high_diffusion_periods.index[-1].strftime('%Y-%m')}")
    else:
        print("   None")

    # Identify periods of low diffusion (< mean - 1 std)
    low_diffusion_threshold = df["diffusion_index"].mean() - diffusion_std
    print(f"\nLow Diffusion Threshold (< {low_diffusion_threshold:.1f}%):")

    low_diffusion_periods = df[df["diffusion_index"] < low_diffusion_threshold]
    if len(low_diffusion_periods) > 0:
        print(f"   {len(low_diffusion_periods)} months of low diffusion")
        print(f"   First: {low_diffusion_periods.index[0].strftime('%Y-%m')}")
        print(f"   Last: {low_diffusion_periods.index[-1].strftime('%Y-%m')}")
    else:
        print("   None")


def plot_diffusion_index(df, output_path):
    """
    Plot diffusion index over time with classification.

    Args:
        df: DataFrame with diffusion index
        output_path: Path to save plot
    """
    print(f"\nGenerating plot: {output_path}")

    plt.figure(figsize=(14, 6))

    # Plot diffusion index
    plt.plot(
        df.index,
        df["diffusion_index"],
        label="Diffusion Index",
        linewidth=1.5,
        color="#1f77b4",
    )

    # Plot 3-month moving average
    plt.plot(
        df.index,
        df["diffusion_ma3"],
        label="3-Month MA",
        linewidth=2,
        color="#ff7f0e",
        linestyle="--",
    )

    # Plot 50% threshold line
    plt.axhline(
        y=50,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label="50% Threshold (Broad-based vs Narrow)",
    )

    # Highlight broad-based vs narrow
    ax = plt.gca()

    # Color background based on classification
    for i in range(len(df)):
        date = df.index[i]
        diff = df["diffusion_index"].iloc[i]
        if diff >= 50:
            ax.axvspan(date, date + pd.DateOffset(months=1), alpha=0.1, color="red")
        else:
            ax.axvspan(date, date + pd.DateOffset(months=1), alpha=0.1, color="green")

    plt.title(
        "Diffusion Index: % of Products with Rising Prices",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Diffusion Index (%)", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    # Set y-axis limits
    plt.ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"   Plot saved to: {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Calculate Diffusion Index for KBR inflation"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/sub_mom.csv",
        help="Input CSV with subcomponent MoM data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/diffusion_index.csv",
        help="Output CSV with diffusion index results",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="assets/images/diffusion_index.png",
        help="Output path for plot image",
    )

    args = parser.parse_args()

    # Load data
    df = load_subcomponent_mom(args.input)

    # Calculate diffusion index
    diffusion_df = calculate_diffusion_index(df)

    # Analyze inflation phases
    analyze_inflation_phases(diffusion_df)

    # Save results
    output_df = diffusion_df.reset_index()
    output_df["Date"] = output_df["Date"].dt.strftime("%Y-%m-%d")
    output_df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")

    # Generate plot
    plot_output_path = Path(args.plot)
    plot_output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_diffusion_index(diffusion_df, plot_output_path)

    print("\n=== Summary ===")
    print(f"Output CSV: {args.output}")
    print(f"Output Plot: {args.plot}")
    print(
        f"Period: {diffusion_df.index[0].strftime('%Y-%m')} to {diffusion_df.index[-1].strftime('%Y-%m')}"
    )
    print(f"Average Diffusion: {diffusion_df['diffusion_index'].mean():.1f}%")
    print(f"\nInterpretation:")
    print(
        f"  Diffusion Index > 50%: Broad-based inflation (prices rising across many categories)"
    )
    print(
        f"  Diffusion Index < 50%: Narrow inflation (price increases concentrated in few categories)"
    )


if __name__ == "__main__":
    main()
