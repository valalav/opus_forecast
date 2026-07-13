#!/usr/bin/env python3
"""
Sticky Price Index Calculation for KBR

Identifies products with infrequent price changes (< 4 per year)
and creates a sub-index to measure inflation persistence.
Sticky prices are less responsive to monetary policy shocks.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple


def load_weekly_prices(filepath: str, min_change_pct: float = 0.5) -> pd.DataFrame:
    """
    Load weekly price data.

    Args:
        filepath: Path to kbr_weekly_prices_2008_2026.csv
        min_change_pct: Minimum % change to count as price change (default 0.5%)

    Returns:
        DataFrame with weekly prices
    """
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    # Price change flag: only count significant changes (> min_change_pct%)
    # This filters out micro-adjustments that don't represent real price changes
    df["price_changed"] = df["wow_growth"].abs() > min_change_pct

    return df


def count_price_changes_per_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many times each product changes price per year.

    Args:
        df: Weekly prices DataFrame

    Returns:
        DataFrame with product_code, year, changes_per_year
    """
    changes = df.groupby(["product_code", "year"])["price_changed"].sum().reset_index()
    changes.columns = ["product_code", "year", "changes_per_year"]

    return changes


def calculate_sticky_products(
    df_weekly: pd.DataFrame, max_changes_per_year: float = 4.0
) -> Tuple[List[int], pd.DataFrame]:
    """
    Identify sticky products (price changes < threshold per year).

    Args:
        df_weekly: Weekly prices DataFrame
        max_changes_per_year: Maximum average changes/year (default 4)

    Returns:
        Tuple of (sticky_product_codes, product_stats_df)
    """
    changes = count_price_changes_per_year(df_weekly)

    # Calculate average changes per year for each product
    avg_changes = (
        changes.groupby("product_code")["changes_per_year"]
        .agg(
            [
                ("avg_changes_per_year", "mean"),
                ("total_years", "count"),
                ("max_changes", "max"),
                ("min_changes", "min"),
            ]
        )
        .reset_index()
    )

    # Add product names
    product_names = df_weekly[["product_code", "product_name"]].drop_duplicates()
    avg_changes = avg_changes.merge(product_names, on="product_code", how="left")

    # Identify sticky products
    sticky_mask = avg_changes["avg_changes_per_year"] < max_changes_per_year
    sticky_products = avg_changes[sticky_mask]["product_code"].tolist()

    return sticky_products, avg_changes


def calculate_monthly_index_from_weekly(
    df_weekly: pd.DataFrame, product_codes: List[int], base_date: str = "2016-01-01"
) -> pd.DataFrame:
    """
    Calculate monthly index from weekly data for specific products.

    Args:
        df_weekly: Weekly prices DataFrame
        product_codes: List of product codes to include
        base_date: Base date for index (default 2016-01-01)

    Returns:
        DataFrame with monthly sticky index
    """
    # Filter to sticky products
    df_sticky = df_weekly[df_weekly["product_code"].isin(product_codes)].copy()

    # Calculate monthly average price for each product
    monthly_prices = (
        df_sticky.groupby(["year_month", "product_code"])["price"].mean().reset_index()
    )

    # Pivot to wide format
    pivot_prices = monthly_prices.pivot(
        index="year_month", columns="product_code", values="price"
    )

    # Calculate month-over-month growth for each product
    pivot_growth = pivot_prices.pct_change() * 100  # Convert to percentage

    # Calculate average growth across all sticky products (equal-weighted)
    sticky_mom = pivot_growth.mean(axis=1).to_frame("sticky_mom")

    # Build cumulative index (starting from base_date = 100)
    base_index = 100.0
    sticky_mom["sticky_index"] = np.nan

    # Find base date position
    if base_date in sticky_mom.index:
        base_idx = sticky_mom.index.get_loc(base_date)
        sticky_mom.iloc[base_idx, sticky_mom.columns.get_loc("sticky_index")] = (
            base_index
        )

        # Cumulative index calculation
        for i in range(base_idx + 1, len(sticky_mom)):
            if pd.notna(sticky_mom.iloc[i, sticky_mom.columns.get_loc("sticky_mom")]):
                prev_idx = sticky_mom.iloc[
                    i - 1, sticky_mom.columns.get_loc("sticky_index")
                ]
                curr_mom = sticky_mom.iloc[i, sticky_mom.columns.get_loc("sticky_mom")]
                sticky_mom.iloc[i, sticky_mom.columns.get_loc("sticky_index")] = (
                    prev_idx * (1 + curr_mom / 100)
                )

    return sticky_mom.reset_index().rename(columns={"year_month": "Date"})


def load_headline_cpi(filepath: str) -> pd.DataFrame:
    """
    Load headline CPI data for comparison.

    Args:
        filepath: Path to inflation_data.csv

    Returns:
        DataFrame with Date and headline CPI
    """
    df = pd.read_csv(filepath, sep=";", encoding="utf-8-sig", decimal=",")

    # Parse dates (format: 31.01.2010)
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
    df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    # Get headline CPI (mom column) - convert to numeric
    df["headline_cpi"] = pd.to_numeric(df["mom"], errors="coerce")
    df_out = df[["Date", "headline_cpi"]].copy()

    return df_out


def compare_volatility(df: pd.DataFrame) -> Dict:
    """
    Compare volatility of sticky index vs headline CPI.

    Args:
        df: Merged DataFrame with sticky_index and headline_cpi

    Returns:
        Dictionary with volatility metrics
    """
    # Filter to valid data
    df_valid = df.dropna(subset=["sticky_index", "headline_cpi"])

    # Calculate MoM growth rates
    df_valid["sticky_mom_calc"] = df_valid["sticky_index"].pct_change() * 100
    df_valid["headline_mom"] = df_valid["headline_cpi"] - 100

    # Calculate standard deviation (volatility)
    sticky_std = df_valid["sticky_mom_calc"].std()
    headline_std = df_valid["headline_mom"].std()

    # Calculate MAE
    sticky_mae = np.abs(df_valid["sticky_mom_calc"]).mean()
    headline_mae = np.abs(df_valid["headline_mom"]).mean()

    improvement = ((headline_std - sticky_std) / headline_std) * 100

    return {
        "sticky_std": sticky_std,
        "headline_std": headline_std,
        "sticky_mae": sticky_mae,
        "headline_mae": headline_mae,
        "volatility_reduction_pct": improvement,
    }


def main():
    """Main execution."""
    # Define paths
    base_dir = Path("/home/valalav/_projects/sirena-kbr")
    weekly_file = base_dir / "data/kbr_weekly_prices_2008_2026.csv"
    infl_file = base_dir / "data/inflation_data.csv"
    output_file = base_dir / "data/sticky_price_index.csv"
    sticky_products_file = base_dir / "data/sticky_products_list.csv"

    print("=" * 70)
    print("Sticky Price Index Calculation for KBR")
    print("=" * 70)

    # Load data
    print(f"\n1. Loading weekly price data from: {weekly_file}")
    df_weekly = load_weekly_prices(weekly_file)
    n_products = df_weekly["product_code"].nunique()
    date_range = f"{df_weekly['date'].min().strftime('%Y-%m')} to {df_weekly['date'].max().strftime('%Y-%m')}"
    print(f"   Loaded {len(df_weekly)} weekly observations")
    print(f"   {n_products} unique products ({date_range})")

    # Identify sticky products
    print(f"\n2. Identifying sticky products (< 4 price changes/year)...")
    sticky_products, product_stats = calculate_sticky_products(
        df_weekly, max_changes_per_year=4.0
    )
    n_sticky = len(sticky_products)
    sticky_pct = (n_sticky / n_products) * 100

    print(f"   {n_sticky} sticky products identified ({sticky_pct:.1f}% of total)")
    print(f"   Threshold: < 4.0 average price changes per year")

    # Save sticky products list
    sticky_stats = product_stats[
        product_stats["product_code"].isin(sticky_products)
    ].copy()
    sticky_stats = sticky_stats.sort_values("avg_changes_per_year")
    sticky_stats.to_csv(sticky_products_file, index=False, encoding="utf-8-sig")
    print(f"   Saved sticky products list to: {sticky_products_file}")

    # Calculate sticky index
    print(f"\n3. Calculating monthly sticky price index...")
    df_sticky = calculate_monthly_index_from_weekly(
        df_weekly, sticky_products, base_date="2016-01-01"
    )
    print(f"   Calculated for {len(df_sticky)} months")

    # Load headline CPI
    print(f"\n4. Loading headline CPI data from: {infl_file}")
    df_headline = load_headline_cpi(infl_file)
    print(f"   Loaded {len(df_headline)} months")

    # Merge datasets
    print(f"\n5. Merging datasets...")
    df_merged = pd.merge(df_sticky, df_headline, on="Date", how="inner")
    print(f"   Merged to {len(df_merged)} months (2016-2025)")

    # Compare volatility
    print(f"\n6. Comparing volatility...")
    metrics = compare_volatility(df_merged)

    print("\n" + "=" * 70)
    print("STICKY PRICE INDEX ANALYSIS")
    print("=" * 70)
    print(f"Total products analyzed:          {n_products}")
    print(f"Sticky products identified:        {n_sticky}")
    print(f"Percentage sticky:               {sticky_pct:.1f}%")
    print()
    print("VOLATILITY COMPARISON")
    print("-" * 70)
    print(f"Sticky Index Std Dev:           {metrics['sticky_std']:.4f}%")
    print(f"Headline CPI Std Dev:           {metrics['headline_std']:.4f}%")
    print(
        f"Volatility Reduction:            {metrics['volatility_reduction_pct']:+.1f}%"
    )
    print()
    print(f"Sticky Index MAE:              {metrics['sticky_mae']:.4f}%")
    print(f"Headline CPI MAE:              {metrics['headline_mae']:.4f}%")

    if metrics["sticky_std"] < metrics["headline_std"]:
        print(f"\n✅ Sticky Price Index is LESS volatile than Headline CPI!")
        print(f"   This is expected - sticky prices change infrequently.")
    else:
        print(f"\n⚠️  Sticky Price Index is MORE volatile than Headline CPI")
        print(f"   This is unusual - check threshold selection.")

    print("=" * 70)

    # Show top sticky products
    print("\nTOP 10 STICKIEST PRODUCTS (lowest avg changes/year):")
    print("-" * 70)
    top_sticky = sticky_stats[
        ["product_code", "product_name", "avg_changes_per_year"]
    ].head(10)
    for _, row in top_sticky.iterrows():
        print(
            f"   {row['product_code']:4d} | {row['product_name'][:40]:40s} | {row['avg_changes_per_year']:.2f} changes/year"
        )

    # Show top volatile products
    print("\nTOP 10 MOST VOLATILE PRODUCTS (highest avg changes/year):")
    print("-" * 70)
    top_volatile = product_stats.sort_values(
        "avg_changes_per_year", ascending=False
    ).head(10)
    top_volatile = top_volatile[
        ["product_code", "product_name", "avg_changes_per_year"]
    ]
    for _, row in top_volatile.iterrows():
        print(
            f"   {row['product_code']:4d} | {row['product_name'][:40]:40s} | {row['avg_changes_per_year']:.2f} changes/year"
        )

    print("=" * 70)

    # Save results
    print(f"\n7. Saving results to: {output_file}")
    df_out = df_merged[["Date", "sticky_index", "headline_cpi"]].copy()
    df_out.to_csv(output_file, index=False, sep=";", float_format="%.4f")
    print(f"   Saved {len(df_out)} rows")

    # Print sample
    print("\n" + "=" * 70)
    print("SAMPLE OUTPUT (last 6 months)")
    print("=" * 70)
    df_out["sticky_mom"] = df_out["sticky_index"].pct_change() * 100
    df_out["headline_mom"] = df_out["headline_cpi"] - 100
    print(
        df_out[["Date", "sticky_index", "sticky_mom", "headline_cpi", "headline_mom"]]
        .tail(6)
        .to_string(index=False)
    )

    print("\n✅ Sticky Price Index calculation complete!")

    return metrics, sticky_stats


if __name__ == "__main__":
    metrics, sticky_stats = main()
