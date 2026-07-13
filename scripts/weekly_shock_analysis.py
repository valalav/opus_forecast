"""
Historical Shock Analysis for Weekly Price Data
============================================

Analyzes weekly price behavior during historical economic shocks to:
1. Identify which products deviate most during each shock period
2. Build shock-detection features from weekly data
3. Test if weekly data can provide early warning
4. Document shock signatures for each product

Shock periods analyzed:
- 2008-09 to 2009-02: Global financial crisis (6 months)
- 2014-12 to 2015-02: Sanctions + currency crisis (3 months)
- 2020-03 to 2020-05: COVID lockdown (3 months)
- 2022-03 to 2022-06: War + sanctions (4 months)

Output:
- data/weekly_shock_signatures.csv: Shock_period, Product_code, Z_score, Lead_weeks
- data/shock_early_warning.csv: Shock, Warning_date, Actual_start, Lead_days
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

# Add parent directory to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.data.weekly_loader import load_weekly_prices, HIGH_QUALITY_PRODUCTS


# Define shock periods based on historical data
SHOCK_PERIODS = {
    "2008_crisis": {
        "name": "2008 Global Financial Crisis",
        "start": pd.Timestamp("2008-09-01"),
        "end": pd.Timestamp("2009-02-28"),
        "description": "Global financial crisis, currency devaluation",
    },
    "2014_sanctions": {
        "name": "2014 Sanctions + Currency Crisis",
        "start": pd.Timestamp("2014-12-01"),
        "end": pd.Timestamp("2015-02-28"),
        "description": "Sanctions imposition, ruble collapse",
    },
    "2020_covid": {
        "name": "2020 COVID Lockdown",
        "start": pd.Timestamp("2020-03-01"),
        "end": pd.Timestamp("2020-05-31"),
        "description": "COVID pandemic, economic shutdown",
    },
    "2022_war": {
        "name": "2022 War + Sanctions",
        "start": pd.Timestamp("2022-03-01"),
        "end": pd.Timestamp("2022-06-30"),
        "description": "Military operation, severe sanctions",
    },
}


def load_monthly_inflation() -> pd.DataFrame:
    """
    Load monthly inflation data to identify shock periods.

    Returns:
        DataFrame with Date, Inflation columns
    """
    # Try multiple possible paths
    paths = [
        Path(__file__).parent.parent / "data" / "inflation_data.csv",
        Path.cwd() / "data" / "inflation_data.csv",
        Path(__file__).parent.parent / "data" / "infl_kbr.csv",
    ]

    for path in paths:
        if path.exists():
            if path.name == "inflation_data.csv":
                df = pd.read_csv(path, sep=";")
                # Parse date (format: dd.mm.yyyy)
                df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
                # Extract inflation (mom is 100.xx format, so -100 = %)
                df["Inflation"] = (
                    pd.to_numeric(
                        df["mom"].astype(str).str.replace(",", "."), errors="coerce"
                    )
                    - 100
                )
                return df[["Date", "Inflation"]].dropna()
            else:  # infl_kbr.csv
                df = pd.read_csv(path, sep=";")
                df["Date"] = pd.to_datetime(df["Day"])
                df["Inflation"] = (
                    pd.to_numeric(
                        df["MoM"].astype(str).str.replace(",", "."), errors="coerce"
                    )
                    - 100
                )
                return df[["Date", "Inflation"]].dropna()

    raise FileNotFoundError("Monthly inflation data not found")


def calculate_shock_z_scores(
    weekly_df: pd.DataFrame,
    shock_start: pd.Timestamp,
    shock_end: pd.Timestamp,
    lookback_weeks: int = 52,
) -> pd.DataFrame:
    """
    Calculate Z-scores for products during a shock period.

    Args:
        weekly_df: Weekly prices DataFrame
        shock_start: Start date of shock
        shock_end: End date of shock
        lookback_weeks: Historical window for baseline stats

    Returns:
        DataFrame with Product_code, Z_score, Z_abs, Lead_weeks
    """
    # Filter data for shock period + baseline
    baseline_start = shock_start - timedelta(weeks=lookback_weeks)

    df = weekly_df[
        (weekly_df["date"] >= baseline_start) & (weekly_df["date"] <= shock_end)
    ].copy()

    results = []

    for product_code in df["product_code"].unique():
        product_data = df[df["product_code"] == product_code].sort_values("date")

        # Calculate baseline stats (pre-shock)
        baseline = product_data[product_data["date"] < shock_start]
        shock = product_data[
            (product_data["date"] >= shock_start) & (product_data["date"] <= shock_end)
        ]

        if len(baseline) < 12 or len(shock) < 1:
            continue

        # Calculate Z-score for each week in shock
        baseline_mean = baseline["wow_growth"].mean()
        baseline_std = baseline["wow_growth"].std()

        if baseline_std == 0 or np.isnan(baseline_std):
            baseline_std = 1  # Avoid division by zero

        # Calculate mean Z-score during shock
        shock_z_scores = ((shock["wow_growth"] - baseline_mean) / baseline_std).abs()
        mean_z = shock_z_scores.mean()
        max_z = shock_z_scores.max()

        # Calculate lead time: first week with Z > 2.0
        first_warning = None
        warning_dates = shock[shock_z_scores > 2.0]
        if len(warning_dates) > 0:
            first_warning = warning_dates.iloc[0]["date"]
            lead_weeks = max(0, (first_warning - shock_start).days // 7)
        else:
            lead_weeks = 0

        results.append(
            {
                "Product_code": product_code,
                "Product_name": product_data["product_name"].iloc[0],
                "Z_score_mean": mean_z,
                "Z_score_max": max_z,
                "Z_abs": mean_z,
                "Lead_weeks": lead_weeks,
                "First_warning": first_warning,
                "n_shock_weeks": len(shock),
            }
        )

    return pd.DataFrame(results)


def calculate_early_warning_signals(
    weekly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    shock_key: str,
    shock_info: Dict,
) -> Dict:
    """
    Test if weekly data provides early warning of shock.

    Methodology:
    1. Look for Z-score spikes > 2.0 BEFORE official shock start
    2. Compare to monthly inflation spike timing
    3. Calculate lead time in days

    Args:
        weekly_df: Weekly prices DataFrame
        monthly_df: Monthly inflation DataFrame
        shock_key: Key for shock (e.g., '2022_war')
        shock_info: Dictionary with start/end dates

    Returns:
        Dict with warning information
    """
    shock_start = shock_info["start"]
    shock_end = shock_info["end"]

    # Look back 8 weeks before shock
    lookback_start = shock_start - timedelta(weeks=8)

    # Aggregate weekly to monthly for comparison
    weekly_df["year_month"] = weekly_df["date"].dt.to_period("M")
    monthly_aggregated = (
        weekly_df.groupby(["year_month", "product_code"])
        .agg({"wow_growth": "mean"})
        .reset_index()
    )
    monthly_aggregated["date"] = monthly_aggregated["year_month"].dt.to_timestamp()

    # Calculate rolling Z-scores
    all_products_z = []
    for product_code in HIGH_QUALITY_PRODUCTS.keys():
        product_data = weekly_df[weekly_df["product_code"] == product_code].sort_values(
            "date"
        )

        # Calculate rolling Z
        product_data["rolling_mean"] = (
            product_data["wow_growth"]
            .shift(1)
            .rolling(window=52, min_periods=12)
            .mean()
        )
        product_data["rolling_std"] = (
            product_data["wow_growth"].shift(1).rolling(window=52, min_periods=12).std()
        )
        product_data["z_score"] = (
            (product_data["wow_growth"] - product_data["rolling_mean"])
            / product_data["rolling_std"]
        ).fillna(0)

        # Look for Z > 2.5 in pre-shock period
        pre_shock = product_data[
            (product_data["date"] >= lookback_start)
            & (product_data["date"] < shock_start)
        ]

        warnings = pre_shock[pre_shock["z_score"].abs() > 2.5]

        if len(warnings) > 0:
            first_warning = warnings.iloc[0]["date"]
            all_products_z.append(
                {
                    "Product_code": product_code,
                    "First_warning": first_warning,
                    "Max_z": warnings["z_score"].abs().max(),
                }
            )

    # Find earliest warning across all products
    if all_products_z:
        warning_df = pd.DataFrame(all_products_z)
        earliest_warning = warning_df["First_warning"].min()

        # Check if monthly inflation spiked after shock start
        monthly_shock = monthly_df[
            (monthly_df["Date"] >= shock_start) & (monthly_df["Date"] <= shock_end)
        ]

        if len(monthly_shock) > 0:
            actual_spike_month = monthly_shock.loc[
                monthly_shock["Inflation"].abs().idxmax()
            ]["Date"]
        else:
            actual_spike_month = shock_start

        lead_days = (actual_spike_month - earliest_warning).days

        return {
            "Shock": shock_info["name"],
            "Warning_date": earliest_warning,
            "Actual_start": actual_spike_month,
            "Lead_days": max(0, lead_days),
            "Lead_weeks": max(0, lead_days // 7),
            "Products_with_warning": len(all_products_z),
            "Top_warning_product": warning_df.loc[warning_df["First_warning"].idxmin()][
                "Product_code"
            ],
        }
    else:
        return {
            "Shock": shock_info["name"],
            "Warning_date": None,
            "Actual_start": shock_start,
            "Lead_days": 0,
            "Lead_weeks": 0,
            "Products_with_warning": 0,
            "Top_warning_product": None,
        }


def analyze_all_shocks() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze all historical shocks.

    Returns:
        signatures_df: DataFrame with shock signatures per product
        early_warnings_df: DataFrame with early warning capability
    """
    # Load data
    weekly_df = load_weekly_prices(start_date="2008-01-01")
    monthly_df = load_monthly_inflation()

    # Filter to high-quality products only
    weekly_df = weekly_df[
        weekly_df["product_code"].isin(HIGH_QUALITY_PRODUCTS.keys())
    ].copy()

    all_signatures = []
    all_warnings = []

    print("Analyzing historical shocks...\n")
    print("=" * 80)

    for shock_key, shock_info in SHOCK_PERIODS.items():
        print(f"\n{shock_info['name']}")
        print(f"Period: {shock_info['start'].date()} to {shock_info['end'].date()}")
        print(f"Description: {shock_info['description']}")

        # Calculate shock signatures (product-level Z-scores)
        signatures = calculate_shock_z_scores(
            weekly_df, shock_info["start"], shock_info["end"]
        )
        signatures["Shock_period"] = shock_key
        signatures["Shock_name"] = shock_info["name"]
        all_signatures.append(signatures)

        # Calculate early warning capability
        warning = calculate_early_warning_signals(
            weekly_df, monthly_df, shock_key, shock_info
        )
        all_warnings.append(warning)

        # Print summary
        print(f"\nTop 5 products by Z-score:")
        top_products = signatures.nlargest(5, "Z_abs")
        for _, row in top_products.iterrows():
            print(
                f"  {row['Product_name']}: Z={row['Z_abs']:.2f}, Lead={row['Lead_weeks']}w"
            )

        if warning["Warning_date"] is not None:
            print(f"\nEarly warning detected:")
            print(f"  First warning: {warning['Warning_date'].date()}")
            print(f"  Actual spike: {warning['Actual_start'].date()}")
            print(f"  Lead time: {warning['Lead_weeks']} weeks")
            print(f"  Products with warning: {warning['Products_with_warning']}")
        else:
            print(f"\nNo early warning detected from weekly data")

        print("-" * 80)

    # Combine results
    signatures_df = pd.concat(all_signatures, ignore_index=True)
    warnings_df = pd.DataFrame(all_warnings)

    return signatures_df, warnings_df


def main():
    """Main execution function."""
    print("\nHistorical Shock Analysis for Weekly Price Data")
    print("=" * 80)

    # Analyze all shocks
    signatures_df, warnings_df = analyze_all_shocks()

    # Save results
    output_dir = Path(__file__).parent.parent / "data"
    signatures_path = output_dir / "weekly_shock_signatures.csv"
    warnings_path = output_dir / "shock_early_warning.csv"

    # Format signatures output (rename to match acceptance criteria exactly)
    signatures_output = signatures_df[
        [
            "Shock_period",
            "Product_code",
            "Product_name",
            "Z_score_mean",
            "Z_score_max",
            "Lead_weeks",
        ]
    ].copy()
    # Rename Z_score_mean to Z_score to match acceptance criteria
    signatures_output = signatures_output.rename(columns={"Z_score_mean": "Z_score"})
    # Keep only required columns: Shock_period, Product_code, Z_score, Lead_weeks
    signatures_output = signatures_output[
        ["Shock_period", "Product_code", "Z_score", "Lead_weeks"]
    ]
    signatures_output.to_csv(signatures_path, index=False)

    # Format warnings output
    warnings_output = warnings_df[
        ["Shock", "Warning_date", "Actual_start", "Lead_days"]
    ].copy()
    warnings_output["Warning_date"] = warnings_output["Warning_date"].dt.strftime(
        "%Y-%m-%d"
    )
    warnings_output["Actual_start"] = warnings_output["Actual_start"].dt.strftime(
        "%Y-%m-%d"
    )
    warnings_output.to_csv(warnings_path, index=False)

    print(f"\nResults saved:")
    print(f"  {signatures_path} ({len(signatures_output)} rows)")
    print(f"  {warnings_path} ({len(warnings_output)} rows)")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nProducts analyzed: {len(HIGH_QUALITY_PRODUCTS)}")
    print(f"Shocks analyzed: {len(SHOCK_PERIODS)}")

    # Average lead time
    avg_lead = warnings_df["Lead_weeks"].mean()
    print(f"\nAverage early warning: {avg_lead:.1f} weeks")

    # Shocks with early warning
    shocks_with_warning = (warnings_df["Lead_weeks"] > 0).sum()
    print(f"Shocks with early warning: {shocks_with_warning}/{len(SHOCK_PERIODS)}")

    # Top products by average Z-score across shocks
    print("\nTop 10 products by average Z-score across all shocks:")
    product_avg_z = (
        signatures_df.groupby("Product_code")["Z_abs"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    for product_code, avg_z in product_avg_z.items():
        product_name = signatures_df[signatures_df["Product_code"] == product_code][
            "Product_name"
        ].iloc[0]
        print(f"  {product_name}: Z_avg={avg_z:.2f}")


if __name__ == "__main__":
    main()
