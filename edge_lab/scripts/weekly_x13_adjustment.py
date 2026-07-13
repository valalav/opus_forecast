#!/usr/bin/env python3
"""
X-13 Seasonal Adjustment for Weekly Prices
===========================================

Tests the hypothesis that seasonally-adjusted weekly prices improve
nowcasting accuracy compared to raw prices.

Methodology:
1. Load weekly prices for 22 high-quality products
2. Apply X-13 ARIMA-SEATS seasonal adjustment per product
3. Calculate wow_growth from both raw and SA-adjusted prices
4. Aggregate to monthly frequency
5. Backtest vs actual monthly CPI (2019-2025)
6. Compare MAE for raw vs SA-adjusted methods

Note: If X-13ARIMA-SEATS binary is not available, falls back to
      statsmodels.tsa.seasonal.seasonal_decompose (additive).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sirena.data.weekly_loader import (
        load_weekly_prices,
        HIGH_QUALITY_PRODUCTS,
    )
except ImportError:
    from sirena.data.weekly_loader import (
        load_weekly_prices,
        HIGH_QUALITY_PRODUCTS,
    )

try:
    from statsmodels.tsa.x13 import x13_arima_analysis

    X13_AVAILABLE = True
except ImportError:
    X13_AVAILABLE = False

from statsmodels.tsa.seasonal import seasonal_decompose


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


def apply_x13_adjustment(series: pd.Series, method: str = "auto") -> pd.Series:
    """
    Apply seasonal adjustment to a time series.

    Args:
        series: Time series with datetime index
        method: 'auto', 'x13', or 'decompose'

    Returns:
        Seasonally-adjusted series
    """
    if len(series) < 52:
        return series

    try:
        if method == "x13" or (method == "auto" and X13_AVAILABLE):
            res = x13_arima_analysis(
                series,
                x12path=None,
                prefer_x13=True,
            )
            return res.seasadj
    except Exception:
        pass

    try:
        res = seasonal_decompose(
            series, model="additive", period=52, extrapolate_trend="freq"
        )
        return series - res.seasonal
    except Exception:
        return series


def compute_raw_signal(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute signal from raw weekly prices.

    Args:
        weekly_df: DataFrame with weekly price data

    Returns:
        DataFrame with columns: date, signal_raw
    """
    high_quality_codes = list(HIGH_QUALITY_PRODUCTS.keys())
    filtered = weekly_df[weekly_df["product_code"].isin(high_quality_codes)].copy()

    raw_signal = filtered.groupby("date")["wow_growth"].mean().reset_index()
    raw_signal = raw_signal.rename(columns={"wow_growth": "signal_raw"})

    return raw_signal


def compute_sa_signal(weekly_df: pd.DataFrame, method: str = "auto") -> pd.DataFrame:
    """
    Compute signal from seasonally-adjusted weekly prices.

    Args:
        weekly_df: DataFrame with weekly price data
        method: Seasonal adjustment method

    Returns:
        DataFrame with columns: date, signal_sa
    """
    high_quality_codes = list(HIGH_QUALITY_PRODUCTS.keys())
    filtered = weekly_df[weekly_df["product_code"].isin(high_quality_codes)].copy()

    filtered = filtered.sort_values(["product_code", "date"]).reset_index(drop=True)

    product_sa_growth = []

    for code in high_quality_codes:
        product_data = filtered[filtered["product_code"] == code].copy()

        if len(product_data) < 52:
            product_data["wow_growth_sa"] = product_data["wow_growth"]
        else:
            product_data = product_data.set_index("date")
            product_data = product_data.sort_index()

            sa_prices = apply_x13_adjustment(product_data["price"], method)

            product_data["price_sa"] = sa_prices.values
            product_data = product_data.reset_index()

            product_data["price_prev_sa"] = product_data["price_sa"].shift(1)
            product_data["wow_growth_sa"] = (
                (product_data["price_sa"] - product_data["price_prev_sa"])
                / product_data["price_prev_sa"]
            ) * 100

        product_sa_growth.append(
            product_data[["date", "product_code", "wow_growth_sa"]]
        )

    if not product_sa_growth:
        return pd.DataFrame(columns=["date", "signal_sa"])

    all_sa = pd.concat(product_sa_growth, ignore_index=True)

    sa_signal = all_sa.groupby("date")["wow_growth_sa"].mean().reset_index()
    sa_signal = sa_signal.rename(columns={"wow_growth_sa": "signal_sa"})

    return sa_signal


def aggregate_weekly_to_monthly(
    weekly_signal: pd.DataFrame,
    signal_col: str,
) -> pd.DataFrame:
    """
    Aggregate weekly signal to monthly frequency.

    Args:
        weekly_signal: DataFrame with weekly signal
        signal_col: Column name of the signal

    Returns:
        DataFrame with monthly signal indexed by date
    """
    df = weekly_signal.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M")

    monthly = df.groupby("year_month")[signal_col].sum().reset_index()

    monthly["date"] = monthly["year_month"].dt.to_timestamp(how="end")
    monthly["date"] = monthly["date"].dt.normalize()

    monthly = monthly.set_index("date").sort_index()

    return monthly[[signal_col]]


def run_backtest(
    raw_monthly: pd.DataFrame,
    sa_monthly: pd.DataFrame,
    monthly_cpi: pd.DataFrame,
    start_date: str = "2019-01-01",
    end_date: str = "2025-12-31",
) -> pd.DataFrame:
    """
    Run backtest comparing raw vs SA-adjusted signals.

    Args:
        raw_monthly: Monthly raw signal
        sa_monthly: Monthly SA-adjusted signal
        monthly_cpi: Monthly actual CPI inflation
        start_date: Backtest start date
        end_date: Backtest end date

    Returns:
        DataFrame with backtest results
    """
    results = monthly_cpi.copy()
    results = results.join(raw_monthly, how="left", rsuffix="_raw")
    results = results.join(sa_monthly, how="left", rsuffix="_sa")

    mask = (results.index >= pd.Timestamp(start_date)) & (
        results.index <= pd.Timestamp(end_date)
    )
    results = results[mask].copy()

    print(f"    Data after date filter: {len(results)} months")
    print(f"    Raw signal non-null: {results['signal_raw'].notna().sum()}")
    print(f"    SA signal non-null: {results['signal_sa'].notna().sum()}")

    results = results.dropna(subset=["inflation", "signal_raw", "signal_sa"])

    print(f"    Data after dropping NA: {len(results)} months")

    results["error_raw"] = results["inflation"] - results["signal_raw"]
    results["error_sa"] = results["inflation"] - results["signal_sa"]

    results["MAE_raw"] = np.abs(results["error_raw"])
    results["MAE_SA"] = np.abs(results["error_sa"])

    return results


def calculate_metrics(results: pd.DataFrame) -> dict:
    """
    Calculate performance metrics for both methods.

    Args:
        results: Backtest results DataFrame

    Returns:
        Dict with metrics
    """
    mae_raw = np.mean(np.abs(results["error_raw"]))
    mae_sa = np.mean(np.abs(results["error_sa"]))

    rmse_raw = np.sqrt(np.mean(results["error_raw"] ** 2))
    rmse_sa = np.sqrt(np.mean(results["error_sa"] ** 2))

    mae_improvement = ((mae_raw - mae_sa) / mae_raw) * 100
    rmse_improvement = ((rmse_raw - rmse_sa) / rmse_raw) * 100

    results["direction_raw"] = np.sign(results["signal_raw"]) == np.sign(
        results["inflation"]
    )
    results["direction_sa"] = np.sign(results["signal_sa"]) == np.sign(
        results["inflation"]
    )

    hit_rate_raw = results["direction_raw"].mean() * 100
    hit_rate_sa = results["direction_sa"].mean() * 100

    return {
        "mae_raw": round(mae_raw, 4),
        "mae_sa": round(mae_sa, 4),
        "mae_improvement_pct": round(mae_improvement, 2),
        "rmse_raw": round(rmse_raw, 4),
        "rmse_sa": round(rmse_sa, 4),
        "rmse_improvement_pct": round(rmse_improvement, 2),
        "hit_rate_raw": round(hit_rate_raw, 2),
        "hit_rate_sa": round(hit_rate_sa, 2),
        "n_periods": len(results),
    }


def main():
    print("=" * 60)
    print("X-13 Seasonal Adjustment for Weekly Prices")
    print("=" * 60)

    if X13_AVAILABLE:
        print("\n[X-13 Status]: X-13 ARIMA-SEATS is available")
    else:
        print(
            "\n[X-13 Status]: X-13 ARIMA-SEATS not available, using seasonal_decompose fallback"
        )

    print("\n[1] Loading data...")
    weekly_df = load_weekly_prices(start_date="2008-01-01")
    print(f"    Loaded {len(weekly_df):,} weekly price observations")
    print(f"    Date range: {weekly_df['date'].min()} to {weekly_df['date'].max()}")
    print(f"    Products: {weekly_df['product_code'].nunique()}")

    print("\n[2] Computing signals...")
    raw_weekly = compute_raw_signal(weekly_df)
    sa_weekly = compute_sa_signal(weekly_df, method="auto")
    print(f"    Raw signal: {len(raw_weekly)} weekly observations")
    print(f"    SA signal: {len(sa_weekly)} weekly observations")

    print("\n[3] Aggregating to monthly...")
    raw_monthly = aggregate_weekly_to_monthly(raw_weekly, "signal_raw")
    sa_monthly = aggregate_weekly_to_monthly(sa_weekly, "signal_sa")
    print(f"    Raw monthly: {len(raw_monthly)} monthly observations")
    print(f"    SA monthly: {len(sa_monthly)} monthly observations")

    print("\n[4] Loading monthly CPI...")
    monthly_cpi = load_monthly_cpi()
    print(f"    Monthly CPI: {len(monthly_cpi)} observations")
    print(f"    Date range: {monthly_cpi.index.min()} to {monthly_cpi.index.max()}")

    print("\n[5] Running backtest...")
    results = run_backtest(
        raw_monthly,
        sa_monthly,
        monthly_cpi,
        start_date="2019-01-01",
        end_date="2025-12-31",
    )
    print(f"    Backtest periods: {len(results)} months")

    print("\n[6] Calculating metrics...")
    metrics = calculate_metrics(results)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nRaw Prices Method:")
    print(f"  MAE: {metrics['mae_raw']:.4f}%")
    print(f"  RMSE: {metrics['rmse_raw']:.4f}%")
    print(f"  Hit Rate: {metrics['hit_rate_raw']:.2f}%")

    print(f"\nSeasonally-Adjusted Method:")
    print(f"  MAE: {metrics['mae_sa']:.4f}%")
    print(f"  RMSE: {metrics['rmse_sa']:.4f}%")
    print(f"  Hit Rate: {metrics['hit_rate_sa']:.2f}%")

    print(f"\nImprovement:")
    print(f"  MAE: {metrics['mae_improvement_pct']:+.2f}%")
    print(f"  RMSE: {metrics['rmse_improvement_pct']:+.2f}%")

    if metrics["mae_sa"] < metrics["mae_raw"]:
        print(
            f"\n=> Seasonal adjustment IMPROVES accuracy by {abs(metrics['mae_improvement_pct']):.2f}%"
        )
    else:
        print(
            f"\n=> Raw prices method is BETTER by {abs(metrics['mae_improvement_pct']):.2f}%"
        )

    print("\n[7] Saving results...")
    output_dir = Path("/home/valalav/_projects/sirena-kbr/edge_lab/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_output = results.reset_index()
    results_output.to_csv(output_dir / "weekly_sa_comparison.csv", index=False)
    print(f"    Saved: {output_dir / 'weekly_sa_comparison.csv'}")

    import json

    metrics_path = output_dir / "weekly_sa_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"    Saved: {metrics_path}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
