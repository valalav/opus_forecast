#!/usr/bin/env python3
"""
Product-Specific Lag Optimization for Weekly Nowcasting
=====================================================

Tests the hypothesis that different products lead/lag CPI by different amounts.
Optimizes lag individually for each of 22 high-quality products.

Methodology:
1. For each product, test lags 1-8 weeks
2. Find optimal lag that maximizes correlation with monthly CPI
3. Create weighted signal using product-specific lags
4. Compare vs uniform lag
5. Document which products lead/lag CPI
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, Tuple, List

warnings.filterwarnings("ignore")

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sirena.data.weekly_loader import (
        load_weekly_prices,
        calculate_product_volatility,
        HIGH_QUALITY_PRODUCTS,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sirena" / "data"))
    from weekly_loader import (
        load_weekly_prices,
        calculate_product_volatility,
        HIGH_QUALITY_PRODUCTS,
    )


def load_monthly_cpi() -> pd.DataFrame:
    """Load monthly CPI inflation data."""
    base_paths = [
        Path.cwd() / "data" / "inflation_data.csv",
        Path.cwd().parent / "data" / "inflation_data.csv",
        Path(__file__).parent.parent / "data" / "inflation_data.csv",
    ]

    for path in base_paths:
        if path.exists():
            df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
            df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", dayfirst=True)
            df = df.set_index("Date").sort_index()

            df.index = df.index.normalize()

            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

            if "mom" in df.columns:
                df["inflation"] = df["mom"] - 100
            else:
                first_numeric = df.select_dtypes(include=[np.number]).columns[0]
                df["inflation"] = df[first_numeric].pct_change() * 100

            return df[["inflation"]].dropna()

    raise FileNotFoundError("Monthly CPI data not found")


def aggregate_weekly_to_monthly(
    weekly_df: pd.DataFrame,
    product_codes: List[int],
    lags: Dict[int, int],
) -> pd.DataFrame:
    """
    Aggregate weekly data to monthly with product-specific lags.

    Args:
        weekly_df: Weekly price data
        product_codes: List of product codes to include
        lags: Dict mapping product_code -> lag_weeks

    Returns:
        DataFrame with monthly aggregated signal indexed by date
    """
    monthly_data = []

    for product_code in product_codes:
        product_df = weekly_df[weekly_df["product_code"] == product_code].copy()

        if len(product_df) == 0:
            continue

        lag = lags.get(product_code, 0)
        if lag > 0:
            product_df["wow_growth"] = product_df["wow_growth"].shift(lag)

        product_df["year_month"] = product_df["date"].dt.to_period("M")

        monthly_product = (
            product_df.groupby("year_month")["wow_growth"].sum().reset_index()
        )
        monthly_product["product_code"] = product_code
        monthly_data.append(monthly_product)

    if not monthly_data:
        return pd.DataFrame()

    combined = pd.concat(monthly_data)
    combined["date"] = combined["year_month"].dt.to_timestamp(how="end")
    combined["date"] = combined["date"].dt.normalize()

    pivot_df = combined.pivot(index="date", columns="product_code", values="wow_growth")

    signal_df = pd.DataFrame(index=pivot_df.index)
    signal_df["signal"] = pivot_df.mean(axis=1)

    return signal_df


def find_optimal_lag_for_product(
    product_df: pd.DataFrame,
    monthly_cpi: pd.DataFrame,
    max_lag: int = 8,
) -> Tuple[int, float]:
    """
    Find the optimal lag for a single product.

    Args:
        product_df: Weekly data for a single product
        monthly_cpi: Monthly CPI data
        max_lag: Maximum lag to test (weeks)

    Returns:
        Tuple of (optimal_lag, max_correlation)
    """
    correlations = {}

    for lag in range(max_lag + 1):
        if lag == 0:
            lagged_growth = product_df["wow_growth"].values
        else:
            lagged_growth = product_df["wow_growth"].shift(lag).values

        product_df["year_month"] = product_df["date"].dt.to_period("M")
        monthly_product = (
            product_df.groupby("year_month")["wow_growth"].sum().reset_index()
        )
        monthly_product["date"] = monthly_product["year_month"].dt.to_timestamp(
            how="end"
        )
        monthly_product["date"] = monthly_product["date"].dt.normalize()

        monthly_product = monthly_product.set_index("date")

        merged = monthly_cpi.join(monthly_product, how="inner")

        if len(merged) < 10:
            correlations[lag] = np.nan
        else:
            corr = merged["inflation"].corr(merged["wow_growth"])
            correlations[lag] = corr

    valid_corrs = {k: v for k, v in correlations.items() if not pd.isna(v)}

    if not valid_corrs:
        return (0, 0.0)

    optimal_lag = max(valid_corrs, key=valid_corrs.get)
    max_correlation = valid_corrs[optimal_lag]

    return (optimal_lag, max_correlation)


def optimize_lags_all_products(
    weekly_df: pd.DataFrame,
    monthly_cpi: pd.DataFrame,
    max_lag: int = 8,
) -> pd.DataFrame:
    """
    Optimize lags for all high-quality products.

    Args:
        weekly_df: Weekly price data
        monthly_cpi: Monthly CPI data
        max_lag: Maximum lag to test (weeks)

    Returns:
        DataFrame with columns: Product_code, Product_name, Optimal_lag, Correlation
    """
    results = []

    high_quality_codes = list(HIGH_QUALITY_PRODUCTS.keys())

    for product_code in high_quality_codes:
        product_name = HIGH_QUALITY_PRODUCTS[product_code]["name"]
        product_df = weekly_df[weekly_df["product_code"] == product_code].copy()

        if len(product_df) == 0:
            results.append(
                {
                    "Product_code": product_code,
                    "Product_name": product_name,
                    "Optimal_lag": 0,
                    "Correlation": 0.0,
                }
            )
            continue

        optimal_lag, max_correlation = find_optimal_lag_for_product(
            product_df, monthly_cpi, max_lag
        )

        results.append(
            {
                "Product_code": product_code,
                "Product_name": product_name,
                "Optimal_lag": optimal_lag,
                "Correlation": max_correlation,
            }
        )

    return pd.DataFrame(results)


def compare_signals(
    weekly_df: pd.DataFrame,
    monthly_cpi: pd.DataFrame,
    optimal_lags_df: pd.DataFrame,
    uniform_lag: int = 0,
    start_date: str = "2019-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """
    Compare product-specific lag signal vs uniform lag signal.

    Args:
        weekly_df: Weekly price data
        monthly_cpi: Monthly CPI data
        optimal_lags_df: DataFrame with optimal lags per product
        uniform_lag: Uniform lag to use for comparison
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        DataFrame with backtest results
    """
    high_quality_codes = list(HIGH_QUALITY_PRODUCTS.keys())

    optimal_lags = dict(
        zip(optimal_lags_df["Product_code"], optimal_lags_df["Optimal_lag"])
    )

    uniform_lags = {code: uniform_lag for code in high_quality_codes}

    signal_optimal = aggregate_weekly_to_monthly(
        weekly_df, high_quality_codes, optimal_lags
    )
    signal_uniform = aggregate_weekly_to_monthly(
        weekly_df, high_quality_codes, uniform_lags
    )

    backtest_df = monthly_cpi.copy()

    backtest_df = backtest_df.join(signal_optimal, how="left", rsuffix="_optimal")
    backtest_df = backtest_df.join(signal_uniform, how="left", rsuffix="_uniform")

    backtest_df.columns = ["inflation", "signal_optimal", "signal_uniform"]

    backtest_df = backtest_df[
        (backtest_df.index >= pd.Timestamp(start_date))
        & (backtest_df.index <= pd.Timestamp(end_date))
    ]

    backtest_df = backtest_df.dropna()

    mae_optimal = (
        (backtest_df["inflation"] - backtest_df["signal_optimal"]).abs().mean()
    )
    mae_uniform = (
        (backtest_df["inflation"] - backtest_df["signal_uniform"]).abs().mean()
    )

    print(f"\nComparison Results ({start_date} to {end_date}):")
    print(f"  Product-specific lag MAE: {mae_optimal:.4f}%")
    print(f"  Uniform lag ({uniform_lag} weeks) MAE: {mae_uniform:.4f}%")
    print(f"  Improvement: {(1 - mae_optimal / mae_uniform) * 100:.2f}%")

    backtest_df["MAE_optimal"] = mae_optimal
    backtest_df["MAE_uniform"] = mae_uniform

    return backtest_df


def main():
    """Main execution function."""
    print("=" * 60)
    print("Product-Specific Lag Optimization for Weekly Nowcasting")
    print("=" * 60)

    print("\n[1/5] Loading weekly price data...")
    weekly_df = load_weekly_prices(start_date="2016-01-01")
    print(f"  Loaded {len(weekly_df):,} weekly observations")

    print("\n[2/5] Loading monthly CPI data...")
    monthly_cpi = load_monthly_cpi()
    print(f"  Loaded {len(monthly_cpi)} monthly observations")

    print("\n[3/5] Optimizing lags for each product...")
    optimal_lags_df = optimize_lags_all_products(weekly_df, monthly_cpi, max_lag=8)
    print(f"  Analyzed {len(optimal_lags_df)} products")

    print("\n[4/5] Saving optimal lags to CSV...")
    output_dir = Path.cwd() / "data"
    if not output_dir.exists():
        output_dir = Path(__file__).parent.parent / "data"

    optimal_lags_df.to_csv(output_dir / "product_optimal_lags.csv", index=False)
    print(f"  Saved to: {output_dir / 'product_optimal_lags.csv'}")

    print("\n[5/5] Comparing product-specific vs uniform lag...")
    backtest_results = compare_signals(
        weekly_df,
        monthly_cpi,
        optimal_lags_df,
        uniform_lag=0,
        start_date="2019-01-01",
        end_date="2025-12-31",
    )

    backtest_output = output_dir / "weekly_lag_comparison.csv"
    backtest_results.to_csv(backtest_output)
    print(f"  Backtest results saved to: {backtest_output}")

    print("\n" + "=" * 60)
    print("Optimal Lags Summary:")
    print("=" * 60)
    print(optimal_lags_df.to_string(index=False))

    lag_stats = optimal_lags_df["Optimal_lag"].value_counts().sort_index()
    print("\nLag Distribution:")
    for lag, count in lag_stats.items():
        print(
            f"  {lag} weeks: {count} products ({count / len(optimal_lags_df) * 100:.1f}%)"
        )

    leading_products = optimal_lags_df[optimal_lags_df["Optimal_lag"] > 0]
    print(f"\nProducts that lead CPI: {len(leading_products)}/{len(optimal_lags_df)}")

    print("\nTop 5 Products by Correlation:")
    top5 = optimal_lags_df.nlargest(5, "Correlation")
    print(top5.to_string(index=False))

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
