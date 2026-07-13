#!/usr/bin/env python3
"""
Leading Indicator Backtest for CPI Nowcasting
==========================================

Rigorously backtest 33 identified leading indicators.

Methodology:
1. For each indicator, run out-of-sample backtest 2019-2025
2. Calculate hit rate (% times direction correct)
3. Calculate value-add (MAE improvement vs baseline)
4. Rank indicators by predictive power
5. Select top 5-10 for production use

Training period: 2016-2018 (pre-backtest)
Backtest period: 2019-2025 (84 months)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Any
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sirena.data.weekly_loader import (
        load_weekly_prices,
        compute_basket_signal,
    )
    from sirena.models.leading_indicators import LeadingIndicatorDetector
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sirena" / "data"))
    from weekly_loader import (
        load_weekly_prices,
        compute_basket_signal,
    )

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sirena" / "models"))
    from leading_indicators import LeadingIndicatorDetector


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
    weekly_df: pd.DataFrame, product_code: int, lag_months: int = 0
) -> pd.Series:
    """
    Aggregate weekly product prices to monthly signal.

    Args:
        weekly_df: Weekly price data
        product_code: Product to aggregate
        lag_months: Months to lag the signal (leading indicator)

    Returns:
        Monthly time series of product price growth
    """
    product_df = weekly_df[weekly_df["product_code"] == product_code].copy()

    if len(product_df) == 0:
        return pd.Series(dtype=float)

    product_df["year_month"] = product_df["date"].dt.to_period("M")

    monthly_product = product_df.groupby("year_month")["wow_growth"].sum().reset_index()
    monthly_product["date"] = monthly_product["year_month"].dt.to_timestamp(how="end")
    monthly_product["date"] = monthly_product["date"].dt.normalize()

    monthly_product = monthly_product.set_index("date")

    if lag_months > 0:
        monthly_product["wow_growth"] = monthly_product["wow_growth"].shift(lag_months)

    return monthly_product["wow_growth"]


def calculate_hit_rate(actual: pd.Series, predicted: pd.Series) -> float:
    """
    Calculate hit rate (% times direction correct).

    Args:
        actual: Actual values
        predicted: Predicted values

    Returns:
        Hit rate (0-1)
    """
    direction_actual = np.sign(actual.values)
    direction_pred = np.sign(predicted)

    hits = (direction_actual == direction_pred).sum()
    total = len(direction_actual)

    return hits / total if total > 0 else 0.0


def backtest_indicator(
    monthly_cpi: pd.DataFrame,
    indicator_series: pd.Series,
    product_code: int,
    product_name: str,
    lag_months: int,
    train_start: str = "2016-01-01",
    train_end: str = "2018-12-31",
    test_start: str = "2019-01-01",
    test_end: str = "2025-12-31",
) -> Dict[str, Any]:
    """
    Backtest a single leading indicator.

    Args:
        monthly_cpi: Monthly CPI data
        indicator_series: Monthly indicator signal
        product_code: Product code
        product_name: Product name
        lag_months: Lag for the indicator
        train_start: Training start date
        train_end: Training end date
        test_start: Test start date
        test_end: Test end date

    Returns:
        Dict with backtest results
    """
    # Merge indicator with CPI
    df = monthly_cpi.copy()
    df = df.join(indicator_series.rename("indicator"), how="left")

    # Drop rows with missing data
    df = df.dropna()

    # Create features
    df["y_L1"] = df["inflation"].shift(1)
    df["y_L2"] = df["inflation"].shift(2)
    df["y_L3"] = df["inflation"].shift(3)

    # Remove initial NaNs
    df = df.dropna()

    # Split train/test
    train_mask = (df.index >= pd.Timestamp(train_start)) & (
        df.index <= pd.Timestamp(train_end)
    )
    test_mask = (df.index >= pd.Timestamp(test_start)) & (
        df.index <= pd.Timestamp(test_end)
    )

    train = df[train_mask].copy()
    test = df[test_mask].copy()

    if len(train) < 12 or len(test) < 12:
        return {
            "product_code": product_code,
            "product_name": product_name,
            "lag_months": lag_months,
            "hit_rate": np.nan,
            "mae_baseline": np.nan,
            "mae_with_indicator": np.nan,
            "mae_improvement": np.nan,
            "n_predictions": 0,
        }

    # Baseline: AR model without indicator
    baseline_features = ["y_L1", "y_L2", "y_L3"]
    model_baseline = Ridge(alpha=1.0)
    model_baseline.fit(train[baseline_features], train["inflation"])
    pred_baseline = model_baseline.predict(test[baseline_features])

    # Test model: AR model + indicator
    test_features = baseline_features + ["indicator"]
    model_test = Ridge(alpha=1.0)
    model_test.fit(train[test_features], train["inflation"])
    pred_test = model_test.predict(test[test_features])

    # Calculate metrics
    mae_baseline = mean_absolute_error(test["inflation"], pred_baseline)
    mae_test = mean_absolute_error(test["inflation"], pred_test)

    hit_rate_baseline = calculate_hit_rate(test["inflation"], pred_baseline)
    hit_rate_test = calculate_hit_rate(test["inflation"], pred_test)

    # MAE improvement
    if mae_baseline > 0:
        mae_improvement = (mae_baseline - mae_test) / mae_baseline * 100
    else:
        mae_improvement = 0.0

    return {
        "product_code": product_code,
        "product_name": product_name,
        "lag_months": lag_months,
        "hit_rate_baseline": hit_rate_baseline,
        "hit_rate_with_indicator": hit_rate_test,
        "hit_rate_improvement": (hit_rate_test - hit_rate_baseline) * 100,
        "mae_baseline": mae_baseline,
        "mae_with_indicator": mae_test,
        "mae_improvement": mae_improvement,
        "n_predictions": len(test),
        "weight_indicator": abs(model_test.coef_[-1])
        if len(model_test.coef_) > 0
        else 0.0,
    }


def run_all_indicator_backtests() -> pd.DataFrame:
    """
    Run backtest for all identified leading indicators.

    Returns:
        DataFrame with backtest results for all indicators
    """
    print("=" * 70)
    print("Leading Indicator Backtest")
    print("=" * 70)

    print("\n[1/6] Loading data...")
    weekly_df = load_weekly_prices(start_date="2016-01-01")
    monthly_cpi = load_monthly_cpi()
    print(f"  Weekly observations: {len(weekly_df):,}")
    print(f"  Monthly CPI observations: {len(monthly_cpi)}")

    print("\n[2/6] Detecting leading indicators...")
    detector = LeadingIndicatorDetector(significance_level=0.10)
    indicators_df = detector.analyze()
    significant = indicators_df[indicators_df["is_significant"] == True]
    print(f"  Total products analyzed: {len(indicators_df)}")
    print(f"  Significant leading indicators (p<0.10): {len(significant)}")

    if len(significant) < 30:
        print("  WARNING: Less than 30 indicators detected!")
    else:
        print(f"  Target met: >=30 indicators")

    print("\n[3/6] Running backtests...")
    results = []

    for idx, row in significant.iterrows():
        product_code = int(row["product_code"])
        product_name = row["product_name"]
        lag_months = int(row["best_lead_lag"])

        # Aggregate weekly to monthly
        indicator_series = aggregate_weekly_to_monthly(
            weekly_df, product_code, lag_months
        )

        # Skip if no data
        if len(indicator_series) < 10:
            continue

        # Backtest
        result = backtest_indicator(
            monthly_cpi,
            indicator_series,
            product_code,
            product_name,
            lag_months,
        )

        results.append(result)

        progress = len(results) / len(significant) * 100
        if len(results) % 5 == 0:
            print(f"  Progress: {len(results)}/{len(significant)} ({progress:.0f}%)")

    print(f"  Completed: {len(results)} backtests")

    print("\n[4/6] Creating results DataFrame...")
    results_df = pd.DataFrame(results)
    results_df = results_df.dropna(subset=["mae_improvement"])

    print(f"  Valid results: {len(results_df)}")

    print("\n[5/6] Ranking indicators...")

    # Calculate combined score
    results_df["combined_score"] = (
        results_df["hit_rate_with_indicator"] * 0.4
        + (1 - results_df["mae_with_indicator"] / 2.0) * 0.4
        + results_df["mae_improvement"].clip(-10, 50) / 50 * 0.2
    )

    results_df = results_df.sort_values("combined_score", ascending=False)
    results_df["Rank"] = range(1, len(results_df) + 1)

    # Create output DataFrame with ONLY required columns
    output_df = pd.DataFrame(
        {
            "Product_code": results_df["product_code"].values,
            "Hit_rate": results_df["hit_rate_with_indicator"].values,
            "MAE_improvement": results_df["mae_improvement"].values,
            "Rank": results_df["Rank"].values,
        }
    )

    print("\n[6/6] Saving results...")
    output_dir = Path.cwd() / "data"
    if not output_dir.exists():
        output_dir = Path(__file__).parent.parent / "data"

    output_df.to_csv(output_dir / "leading_indicator_performance.csv", index=False)
    print(f"  Saved to: {output_dir / 'leading_indicator_performance.csv'}")
    print(f"  Columns: Product_code, Hit_rate, MAE_improvement, Rank")

    return results_df


def print_summary(results_df: pd.DataFrame):
    """Print summary of backtest results."""
    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)

    print(f"\nTotal indicators tested: {len(results_df)}")
    print(f"Test period: 2019-2025 ({results_df['n_predictions'].iloc[0]} months)")

    print("\nBaseline Model (AR without indicators):")
    baseline_mae = results_df["mae_baseline"].mean()
    baseline_hit = results_df["hit_rate_baseline"].mean()
    print(f"  Average MAE: {baseline_mae:.4f}%")
    print(f"  Average hit rate: {baseline_hit * 100:.1f}%")

    print("\nWith Leading Indicators:")
    avg_mae = results_df["mae_with_indicator"].mean()
    avg_improvement = results_df["mae_improvement"].mean()
    avg_hit = (results_df["hit_rate_with_indicator"]).mean()
    print(f"  Average MAE: {avg_mae:.4f}%")
    print(f"  Average hit rate: {avg_hit * 100:.1f}%")
    print(f"  Average MAE improvement: {avg_improvement:+.2f}%")

    improving = (results_df["mae_improvement"] > 0).sum()
    print(
        f"  Indicators that improve MAE: {improving}/{len(results_df)} ({improving / len(results_df) * 100:.0f}%)"
    )

    print("\n" + "-" * 70)
    print("TOP 10 LEADING INDICATORS")
    print("-" * 70)
    print(f"{'Rank':<4} {'Product':<40} {'Lag':<4} {'Hit':<5} {'MAE':<7} {'Imprv':<7}")
    print("-" * 70)

    top10 = results_df.head(10)
    for _, row in top10.iterrows():
        print(
            f"{int(row['Rank']):<4} {row['product_name'][:40]:<40} "
            f"{int(row['lag_months']):<4} {row['hit_rate_with_indicator'] * 100:<5.1f} "
            f"{row['mae_with_indicator']:<7.4f} {row['mae_improvement']:+.2f}%"
        )

    print("\n" + "-" * 70)
    print("WORST 10 INDICATORS")
    print("-" * 70)
    worst10 = results_df.tail(10)
    for _, row in worst10.iterrows():
        print(
            f"{int(row['Rank']):<4} {row['product_name'][:40]:<40} "
            f"{int(row['lag_months']):<4} {row['hit_rate_with_indicator'] * 100:<5.1f} "
            f"{row['mae_with_indicator']:<7.4f} {row['mae_improvement']:+.2f}%"
        )

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR PRODUCTION")
    print("=" * 70)

    top5 = results_df.head(5)
    top5_codes = top5["product_code"].tolist()

    print(f"\nTop 5 indicators for production:")
    for _, row in top5.iterrows():
        print(f"  - {row['product_name']}")
        print(
            f"    Code: {int(row['product_code'])}, Lag: {int(row['lag_months'])} months"
        )
        print(
            f"    Hit rate: {row['hit_rate_with_indicator'] * 100:.1f}%, "
            f"MAE: {row['mae_with_indicator']:.4f}%"
        )

    top10 = results_df.head(10)
    print(f"\nExtended list (top 10):")
    for _, row in top10.iterrows():
        print(f"  {int(row['product_code']):4d}: {row['product_name']}")

    print("\n" + "=" * 70)


def update_docs():
    """Update WEEKLY_RESEARCH.md with leading indicators section."""
    docs_path = Path.cwd() / "docs" / "WEEKLY_RESEARCH.md"
    if not docs_path.exists():
        docs_path = Path(__file__).parent.parent / "docs" / "WEEKLY_RESEARCH.md"

    output_dir = Path.cwd() / "data"
    if not output_dir.exists():
        output_dir = Path(__file__).parent.parent / "data"

    section = "\n\n## Leading Indicators Backtest\n\n"
    section += "### Methodology\n\n"
    section += "Backtest of leading indicators (p<0.10 Granger causality) "
    section += "using out-of-sample validation (2019-2025).\n\n"
    section += "- **Baseline**: AR model on CPI lagged values only\n"
    section += "- **Test**: AR model + leading indicator signal\n"
    section += "- **Metrics**: Hit rate (direction correctness), MAE improvement\n\n"

    # Load results from CSV with correct columns
    results_path = output_dir / "leading_indicator_performance.csv"
    if results_path.exists():
        results_df = pd.read_csv(results_path)

        section += "### Results\n\n"
        section += "| Rank | Product_code | Hit_rate | MAE_improvement |\n"
        section += "|------|-------------|----------|---------------|\n"

        for _, row in results_df.iterrows():
            section += f"| {int(row['Rank'])} | {int(row['Product_code'])} | "
            section += f"{row['Hit_rate']:.3f} | "
            section += f"{row['MAE_improvement']:+.2f}% |\n"

        section += f"\n**Key findings:**\n"
        section += f"- {len(results_df)} indicators backtested\n"
        improving = (results_df["MAE_improvement"] > 0).sum()
        section += f"- {improving}/{len(results_df)} indicators improve MAE ({improving / len(results_df) * 100:.0f}%)\n\n"

        section += "### Production Recommendations\n\n"
        section += "**Top indicators for production use (by Rank):**\n\n"
        for _, row in results_df.head(10).iterrows():
            section += f"  {int(row['Rank'])}. Product_code {int(row['Product_code'])} (Hit: {row['Hit_rate']:.1%}, MAE imprv: {row['MAE_improvement']:+.2f}%)\n"
    else:
        section += "### Results\n\n"
        section += f"Run script to generate results: `python3 scripts/leading_indicator_backtest.py`\n"

    if docs_path.exists():
        with open(docs_path, "r", encoding="utf-8") as f:
            existing = f.read()

        if "## Leading Indicators Backtest" not in existing:
            existing += section
            with open(docs_path, "w", encoding="utf-8") as f:
                f.write(existing)
            print(f"  Updated: {docs_path}")
        else:
            print(f"  Section already exists in {docs_path}")
    else:
        with open(docs_path, "w", encoding="utf-8") as f:
            f.write("# WEEKLY_RESEARCH\n\n" + section)
        print(f"  Created: {docs_path}")


def main():
    """Main execution function."""
    results_df = run_all_indicator_backtests()
    print_summary(results_df)
    update_docs()

    print("\nBacktest complete!")


if __name__ == "__main__":
    main()
