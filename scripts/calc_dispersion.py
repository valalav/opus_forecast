#!/usr/bin/env python3
"""
Price Dispersion Index Calculation for KBR

Calculates variance of price changes across all products.
High dispersion = relative price shocks (some products rise, others fall/stable).

Price Dispersion measures the heterogeneity of inflation across the basket.
During relative price shocks (e.g., devaluation, tariff hikes),
dispersion increases as prices of different goods adjust at different rates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def load_weekly_prices(filepath):
    """
    Load weekly price data from Rosstat.

    Args:
        filepath: Path to kbr_weekly_prices_2008_2026.csv

    Returns:
        DataFrame with weekly price data
    """
    print(f"Loading weekly prices from: {filepath}")
    df = pd.read_csv(filepath)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"])

    # Filter out rows without wow_growth (first week of data)
    df = df[df["wow_growth"].notna()].copy()

    return df


def calculate_weekly_dispersion(df):
    """
    Calculate variance of price changes across all products for each week.

    Args:
        df: DataFrame with weekly prices and wow_growth

    Returns:
        DataFrame with weekly dispersion
    """
    print(
        "\nCalculating weekly dispersion (variance of wow_growth across all products)..."
    )

    # Group by date and calculate variance of wow_growth
    weekly_dispersion = (
        df.groupby("date")["wow_growth"]
        .agg(
            [
                ("n_products", "count"),
                ("mean_growth", "mean"),
                ("variance", "var"),
                ("std_dev", "std"),
                ("dispersion_idx", lambda x: x.var() * 100),  # Scale for readability
            ]
        )
        .reset_index()
    )

    print(f"   Calculated dispersion for {len(weekly_dispersion)} weeks")
    print(f"   Average products per week: {weekly_dispersion['n_products'].mean():.1f}")
    print(
        f"   Dispersion range: {weekly_dispersion['dispersion_idx'].min():.2f} to {weekly_dispersion['dispersion_idx'].max():.2f}"
    )

    return weekly_dispersion


def aggregate_to_monthly(df_weekly):
    """
    Aggregate weekly dispersion to monthly level.

    Args:
        df_weekly: DataFrame with weekly dispersion

    Returns:
        DataFrame with monthly dispersion (mean of weekly values)
    """
    print("\nAggregating to monthly level...")

    df_weekly["year_month"] = df_weekly["date"].dt.to_period("M")

    monthly_dispersion = (
        df_weekly.groupby("year_month")
        .agg(
            {
                "date": "first",  # Use first date of month
                "n_products": "mean",  # Average number of products
                "mean_growth": "mean",
                "variance": "mean",
                "std_dev": "mean",
                "dispersion_idx": "mean",  # Average weekly dispersion
            }
        )
        .reset_index()
    )

    # Convert period to timestamp
    monthly_dispersion["Date"] = monthly_dispersion["year_month"].dt.to_timestamp()

    print(f"   Aggregated to {len(monthly_dispersion)} months")

    return monthly_dispersion


def load_headline_cpi(filepath):
    """
    Load Headline CPI data.

    Args:
        filepath: Path to inflation_data.csv

    Returns:
        DataFrame with monthly CPI
    """
    print(f"\nLoading headline CPI from: {filepath}")

    # Read with semicolon separator
    df = pd.read_csv(filepath, sep=";", encoding="utf-8-sig")

    # Clean column names
    df.columns = [col.strip() for col in df.columns]

    # Parse dates (format: 31.01.2010)
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
    df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    # Convert mom to numeric (replace comma with decimal point)
    df["mom"] = pd.to_numeric(
        df["mom"].astype(str).str.replace(",", "."), errors="coerce"
    )

    # Convert from index (101.xx) to percentage (1.xx%)
    df["Headline_CPI_Pct"] = df["mom"] - 100

    print(f"   Loaded {len(df)} months")

    return df[["Date", "Headline_CPI_Pct"]]


def merge_data(df_dispersion, df_cpi):
    """
    Merge dispersion data with headline CPI.

    Args:
        df_dispersion: Monthly dispersion data
        df_cpi: Headline CPI data

    Returns:
        Merged DataFrame
    """
    print("\nMerging dispersion with headline CPI...")

    df_merged = pd.merge(df_dispersion, df_cpi, on="Date", how="inner")

    print(f"   Merged {len(df_merged)} months")

    return df_merged


def analyze_dispersion_relationship(df):
    """
    Analyze relationship between dispersion and CPI.

    Args:
        df: Merged DataFrame

    Returns:
        dict with analysis results
    """
    print("\n" + "=" * 60)
    print("DISPERSION ANALYSIS")
    print("=" * 60)

    # Calculate correlation
    correlation = df["dispersion_idx"].corr(df["Headline_CPI_Pct"])

    # Calculate volatility periods
    high_dispersion_threshold = df["dispersion_idx"].quantile(0.75)
    high_disp_periods = df[df["dispersion_idx"] > high_dispersion_threshold]

    avg_cpi_high_disp = high_disp_periods["Headline_CPI_Pct"].mean()
    avg_cpi_normal = df[df["dispersion_idx"] <= high_dispersion_threshold][
        "Headline_CPI_Pct"
    ].mean()

    print(f"\nCorrelation (Dispersion vs CPI): {correlation:.4f}")

    if abs(correlation) > 0.3:
        print(
            f"   {'Strong' if abs(correlation) > 0.5 else 'Moderate'} correlation detected!"
        )
    elif abs(correlation) > 0.1:
        print(f"   Weak correlation detected.")
    else:
        print(f"   No significant correlation.")

    print(f"\nHigh dispersion periods (top 25%):")
    print(f"   Average inflation: {avg_cpi_high_disp:.4f}%")
    print(f"   Normal periods average: {avg_cpi_normal:.4f}%")

    if avg_cpi_high_disp > avg_cpi_normal:
        print(f"   → High dispersion associated with HIGHER inflation")
    else:
        print(f"   → High dispersion associated with LOWER inflation")

    # Find peak dispersion months
    print(f"\nPeak dispersion months (top 5):")
    top_months = df.nlargest(5, "dispersion_idx")
    for _, row in top_months.iterrows():
        print(
            f"   {row['Date'].strftime('%Y-%m')}: Dispersion={row['dispersion_idx']:.2f}, CPI={row['Headline_CPI_Pct']:.4f}%"
        )

    return {
        "correlation": correlation,
        "avg_cpi_high_disp": avg_cpi_high_disp,
        "avg_cpi_normal": avg_cpi_normal,
    }


def plot_dispersion_vs_cpi(df, output_path):
    """
    Plot dispersion vs Headline CPI.

    Args:
        df: Merged DataFrame
        output_path: Path to save plot
    """
    print(f"\nGenerating plot: {output_path}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot 1: Dispersion Index over time
    ax1.plot(df["Date"], df["dispersion_idx"], color="steelblue", linewidth=1.5)
    ax1.axhline(
        df["dispersion_idx"].mean(),
        color="red",
        linestyle="--",
        alpha=0.7,
        label="Mean",
    )
    ax1.set_ylabel("Dispersion Index", fontsize=12)
    ax1.set_title(
        "Price Dispersion Index Across All Products", fontsize=14, fontweight="bold"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Dispersion vs Headline CPI scatter
    ax2.scatter(df["dispersion_idx"], df["Headline_CPI_Pct"], alpha=0.6, s=30)
    ax2.set_xlabel("Dispersion Index", fontsize=12)
    ax2.set_ylabel("Headline CPI (%)", fontsize=12)
    ax2.set_title("Dispersion vs Headline CPI", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(df["dispersion_idx"], df["Headline_CPI_Pct"], 1)
    p = np.poly1d(z)
    ax2.plot(
        df["dispersion_idx"],
        p(df["dispersion_idx"]),
        "r--",
        alpha=0.8,
        label=f"Trend: y={z[0]:.3f}x+{z[1]:.3f}",
    )
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"   Plot saved to {output_path}")
    plt.close()


def main():
    """Main execution."""
    # Define paths
    base_dir = Path("/home/valalav/_projects/sirena-kbr")
    weekly_prices_file = base_dir / "data" / "kbr_weekly_prices_2008_2026.csv"
    cpi_file = base_dir / "data" / "inflation_data.csv"
    output_file = base_dir / "data" / "price_dispersion.csv"
    plot_file = base_dir / "assets" / "charts" / "price_dispersion_vs_cpi.png"

    print("=" * 60)
    print("Price Dispersion Index Calculation for KBR")
    print("=" * 60)

    # Load data
    df_weekly = load_weekly_prices(weekly_prices_file)

    # Calculate weekly dispersion
    df_weekly_disp = calculate_weekly_dispersion(df_weekly)

    # Aggregate to monthly
    df_monthly_disp = aggregate_to_monthly(df_weekly_disp)

    # Load headline CPI
    df_cpi = load_headline_cpi(cpi_file)

    # Merge
    df_merged = merge_data(df_monthly_disp, df_cpi)

    # Analyze
    analysis = analyze_dispersion_relationship(df_merged)

    # Save results
    print("\n" + "=" * 60)
    print(f"Saving results to: {output_file}")
    print("=" * 60)

    df_out = df_merged[
        [
            "Date",
            "dispersion_idx",
            "variance",
            "std_dev",
            "mean_growth",
            "n_products",
            "Headline_CPI_Pct",
        ]
    ].copy()

    df_out.to_csv(output_file, index=False, sep=";", float_format="%.4f")
    print(f"   Saved {len(df_out)} rows")

    # Create plot
    try:
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        plot_dispersion_vs_cpi(df_merged, plot_file)
    except Exception as e:
        print(f"   Warning: Could not generate plot: {e}")
        plot_file = None

    # Print sample
    print("\n" + "=" * 60)
    print("SAMPLE OUTPUT (last 5 months)")
    print("=" * 60)
    print(
        df_out[["Date", "dispersion_idx", "Headline_CPI_Pct"]]
        .tail()
        .to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Date range: {df_out['Date'].min()} to {df_out['Date'].max()}")
    print(f"Average dispersion: {df_out['dispersion_idx'].mean():.4f}")
    print(f"Dispersion std dev: {df_out['dispersion_idx'].std():.4f}")
    print(f"Correlation with CPI: {analysis['correlation']:.4f}")
    print("=" * 60)

    print("\n✅ Price Dispersion Index calculation complete!")

    return df_merged


if __name__ == "__main__":
    df = main()
