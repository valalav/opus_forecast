#!/usr/bin/env python3
"""
Volatility-Weighted Weekly Nowcasting for CPI
==============================================

Tests the hypothesis that weighting products by inverse volatility (1/std)
improves nowcasting accuracy compared to equal-weighted aggregation.

Methodology:
1. Load weekly prices for 22 high-quality products
2. Calculate historical volatility per product (3-year rolling window)
3. Create two signals:
   - Equal-weighted: mean of all product wow_growth
   - Volatility-weighted: sum(wow_growth * 1/std) / sum(1/std)
4. Aggregate weekly signals to monthly frequency
5. Backtest vs actual monthly CPI (2019-2025)
6. Compare MAE for both methods
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

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

            # Normalize dates to midnight for proper join
            df.index = df.index.normalize()

            # Convert numeric columns
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

            # Extract mom inflation (MoM %)
            if "mom" in df.columns:
                df["inflation"] = df["mom"] - 100
            else:
                # Calculate from first column
                first_numeric = df.select_dtypes(include=[np.number]).columns[0]
                df["inflation"] = df[first_numeric].pct_change() * 100

            return df[["inflation"]].dropna()

    raise FileNotFoundError("Monthly CPI data not found")


def compute_equal_weighted_signal(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute equal-weighted signal from weekly prices.

    Args:
        weekly_df: DataFrame with weekly price data

    Returns:
        DataFrame with columns: date, signal_equal
    """
    high_quality_codes = list(HIGH_QUALITY_PRODUCTS.keys())
    filtered = weekly_df[weekly_df["product_code"].isin(high_quality_codes)].copy()

    # Group by date and compute mean of wow_growth
    equal_signal = filtered.groupby("date")["wow_growth"].mean().reset_index()
    equal_signal = equal_signal.rename(columns={"wow_growth": "signal_equal"})

    return equal_signal


def compute_volatility_weighted_signal(
    weekly_df: pd.DataFrame,
    volatility_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute volatility-weighted signal from weekly prices.

    Weighting formula: sum(wow_growth * 1/std) / sum(1/std)

    Args:
        weekly_df: DataFrame with weekly price data
        volatility_stats: DataFrame with product volatility stats

    Returns:
        DataFrame with columns: date, signal_vol_weighted
    """
    high_quality_codes = list(HIGH_QUALITY_PRODUCTS.keys())
    filtered = weekly_df[weekly_df["product_code"].isin(high_quality_codes)].copy()

    # Merge volatility stats
    vol_map = volatility_stats.set_index("product_code")["volatility"].to_dict()
    filtered["volatility"] = filtered["product_code"].map(vol_map)

    # Calculate inverse volatility weights (1/std)
    filtered["inv_vol"] = 1.0 / filtered["volatility"]

    # Compute weighted signal per date
    def compute_weighted_mean(group):
        valid = group.dropna(subset=["wow_growth", "inv_vol"])
        if len(valid) == 0:
            return np.nan
        weighted_sum = (valid["wow_growth"] * valid["inv_vol"]).sum()
        weight_sum = valid["inv_vol"].sum()
        return weighted_sum / weight_sum if weight_sum > 0 else np.nan

    vol_signal = filtered.groupby("date").apply(compute_weighted_mean).reset_index()
    vol_signal = vol_signal.rename(columns={0: "signal_vol_weighted"})

    return vol_signal


def aggregate_weekly_to_monthly(
    weekly_signal: pd.DataFrame,
    signal_col: str,
    method: str = "sum",
    month_end: bool = True,
) -> pd.DataFrame:
    """
    Aggregate weekly signal to monthly frequency.

    Args:
        weekly_signal: DataFrame with weekly signal
        signal_col: Column name of the signal
        method: Aggregation method ('sum' or 'mean')
        month_end: If True, align to month end (for CPI compatibility)

    Returns:
        DataFrame with monthly signal indexed by date
    """
    df = weekly_signal.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M")

    if method == "sum":
        # Sum weekly growth to get monthly growth
        monthly = df.groupby("year_month")[signal_col].sum().reset_index()
    else:
        # Average weekly values
        monthly = df.groupby("year_month")[signal_col].mean().reset_index()

    # Set date to month end to match CPI data
    if month_end:
        monthly["date"] = monthly["year_month"].dt.to_timestamp(how="end")
    else:
        monthly["date"] = monthly["year_month"].dt.to_timestamp()

    # Normalize to midnight for proper join with CPI
    monthly["date"] = monthly["date"].dt.normalize()

    monthly = monthly.set_index("date").sort_index()

    return monthly[[signal_col]]


def run_backtest(
    equal_monthly: pd.DataFrame,
    vol_monthly: pd.DataFrame,
    monthly_cpi: pd.DataFrame,
    start_date: str = "2019-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """
    Run backtest comparing equal-weighted vs volatility-weighted signals.

    Args:
        equal_monthly: Monthly equal-weighted signal
        vol_monthly: Monthly volatility-weighted signal
        monthly_cpi: Monthly actual CPI inflation
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        DataFrame with backtest results
    """
    # Merge all data
    results = monthly_cpi.copy()
    results = results.join(equal_monthly, how="left", rsuffix="_eq")
    results = results.join(vol_monthly, how="left", rsuffix="_vol")

    # Filter to date range
    mask = (results.index >= pd.Timestamp(start_date)) & (
        results.index <= pd.Timestamp(end_date)
    )
    results = results[mask].copy()

    print(f"    Data after date filter: {len(results)} months")
    print(f"    Equal signal non-null: {results['signal_equal'].notna().sum()}")
    print(f"    Vol signal non-null: {results['signal_vol_weighted'].notna().sum()}")

    # Drop rows with missing data
    results = results.dropna(
        subset=["inflation", "signal_equal", "signal_vol_weighted"]
    )

    print(f"    Data after dropping NA: {len(results)} months")

    # Calculate errors
    results["error_equal"] = results["inflation"] - results["signal_equal"]
    results["error_vol"] = results["inflation"] - results["signal_vol_weighted"]

    # Rolling MAE (6-month)
    results["mae_equal_rolling"] = results["error_equal"].abs().rolling(6).mean()
    results["mae_vol_rolling"] = results["error_vol"].abs().rolling(6).mean()

    return results


def calculate_metrics(results: pd.DataFrame) -> dict:
    """
    Calculate performance metrics for both methods.

    Args:
        results: Backtest results DataFrame

    Returns:
        Dict with metrics
    """
    mae_equal = np.mean(np.abs(results["error_equal"]))
    mae_vol = np.mean(np.abs(results["error_vol"]))

    rmse_equal = np.sqrt(np.mean(results["error_equal"] ** 2))
    rmse_vol = np.sqrt(np.mean(results["error_vol"] ** 2))

    # Calculate improvement
    mae_improvement = ((mae_equal - mae_vol) / mae_equal) * 100
    rmse_improvement = ((rmse_equal - rmse_vol) / rmse_equal) * 100

    # Hit rate (directional accuracy)
    results["direction_equal"] = np.sign(results["signal_equal"]) == np.sign(
        results["inflation"]
    )
    results["direction_vol"] = np.sign(results["signal_vol_weighted"]) == np.sign(
        results["inflation"]
    )

    hit_rate_equal = results["direction_equal"].mean() * 100
    hit_rate_vol = results["direction_vol"].mean() * 100

    return {
        "mae_equal": round(mae_equal, 4),
        "mae_vol": round(mae_vol, 4),
        "mae_improvement_pct": round(mae_improvement, 2),
        "rmse_equal": round(rmse_equal, 4),
        "rmse_vol": round(rmse_vol, 4),
        "rmse_improvement_pct": round(rmse_improvement, 2),
        "hit_rate_equal": round(hit_rate_equal, 2),
        "hit_rate_vol": round(hit_rate_vol, 2),
        "n_periods": len(results),
    }


def main():
    print("=" * 60)
    print("Volatility-Weighted Weekly Nowcasting Backtest")
    print("=" * 60)

    # Load data
    print("\n[1] Loading data...")
    weekly_df = load_weekly_prices(start_date="2008-01-01")
    print(f"    Loaded {len(weekly_df):,} weekly price observations")
    print(f"    Date range: {weekly_df['date'].min()} to {weekly_df['date'].max()}")
    print(f"    Products: {weekly_df['product_code'].nunique()}")

    # Calculate volatility
    print("\n[2] Calculating product volatility...")
    volatility_stats = calculate_product_volatility(weekly_df, lookback_years=3)
    print(f"    Volatility stats for {len(volatility_stats)} products")
    print(f"    Mean volatility: {volatility_stats['volatility'].mean():.3f}%")
    print(f"    Min volatility: {volatility_stats['volatility'].min():.3f}%")
    print(f"    Max volatility: {volatility_stats['volatility'].max():.3f}%")

    # Compute signals
    print("\n[3] Computing signals...")
    equal_weekly = compute_equal_weighted_signal(weekly_df)
    vol_weekly = compute_volatility_weighted_signal(weekly_df, volatility_stats)
    print(f"    Equal-weighted signal: {len(equal_weekly)} weekly observations")
    print(f"    Vol-weighted signal: {len(vol_weekly)} weekly observations")

    # Aggregate to monthly
    print("\n[4] Aggregating to monthly...")
    equal_monthly = aggregate_weekly_to_monthly(
        equal_weekly, "signal_equal", method="sum"
    )
    vol_monthly = aggregate_weekly_to_monthly(
        vol_weekly, "signal_vol_weighted", method="sum"
    )
    print(f"    Equal-weighted: {len(equal_monthly)} monthly observations")
    print(f"    Vol-weighted: {len(vol_monthly)} monthly observations")

    # Load monthly CPI
    print("\n[5] Loading monthly CPI...")
    monthly_cpi = load_monthly_cpi()
    print(f"    Monthly CPI: {len(monthly_cpi)} observations")
    print(f"    Date range: {monthly_cpi.index.min()} to {monthly_cpi.index.max()}")

    # Run backtest
    print("\n[6] Running backtest...")
    results = run_backtest(
        equal_monthly,
        vol_monthly,
        monthly_cpi,
        start_date="2019-01-01",
        end_date="2025-12-31",
    )
    print(f"    Backtest periods: {len(results)} months")

    # Calculate metrics
    print("\n[7] Calculating metrics...")
    metrics = calculate_metrics(results)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nEqual-Weighted Method:")
    print(f"  MAE: {metrics['mae_equal']:.4f}%")
    print(f"  RMSE: {metrics['rmse_equal']:.4f}%")
    print(f"  Hit Rate: {metrics['hit_rate_equal']:.2f}%")

    print(f"\nVolatility-Weighted Method:")
    print(f"  MAE: {metrics['mae_vol']:.4f}%")
    print(f"  RMSE: {metrics['rmse_vol']:.4f}%")
    print(f"  Hit Rate: {metrics['hit_rate_vol']:.2f}%")

    print(f"\nImprovement:")
    print(f"  MAE: {metrics['mae_improvement_pct']:+.2f}%")
    print(f"  RMSE: {metrics['rmse_improvement_pct']:+.2f}%")

    if metrics["mae_vol"] < metrics["mae_equal"]:
        print(
            f"\n=> Volatility-weighted method IMPROVES accuracy by {abs(metrics['mae_improvement_pct']):.2f}%"
        )
    else:
        print(
            f"\n=> Equal-weighted method is BETTER by {abs(metrics['mae_improvement_pct']):.2f}%"
        )

    # Save results
    print("\n[8] Saving results...")

    # Save detailed results
    output_dir = Path("/home/valalav/_projects/sirena-kbr/edge_lab/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "weekly_vol_weighted_results.csv"
    results_output = results.reset_index()
    results_output["MAE_equal"] = results["error_equal"].abs()
    results_output["MAE_vol"] = results["error_vol"].abs()
    results_output.to_csv(results_path, index=False)
    print(f"    Saved: {results_path}")

    # Save metrics
    metrics_path = output_dir / "weekly_vol_metrics.json"
    import json

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"    Saved: {metrics_path}")

    # Save volatility stats
    vol_path = output_dir / "product_volatility_stats.csv"
    volatility_stats.to_csv(vol_path, index=False)
    print(f"    Saved: {vol_path}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
