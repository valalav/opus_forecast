#!/usr/bin/env python3
"""
Weekly-Monthly Ensemble Optimization
==================================

Tests optimal blend ratios between weekly nowcast and monthly model forecasts.

Task 415: Ensemble with Monthly Models
1. Generate forecasts from top 5 monthly models (Huber, Ridge, NGBoost, Subcomp, EBM)
2. Test blend ratios: 20/80, 40/60, 60/40, 80/20 (weekly/monthly)
3. Test dynamic blend based on weeks elapsed in month
4. Find optimal blend that minimizes MAE
5. Document optimal strategy

Output: data/weekly_monthly_blend_results.csv
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.models.weekly_prices import WeeklyPriceNowcaster
from sirena.models import (
    HuberForecaster,
    RidgeForecaster,
    SubcomponentForecaster,
)


def load_monthly_data() -> pd.DataFrame:
    """
    Load monthly CPI data for backtesting.

    Returns:
        DataFrame with Date index and MoM column
    """
    base_paths = [
        Path.cwd() / "data" / "infl_kbr.csv",
        Path(__file__).parent.parent / "data" / "infl_kbr.csv",
    ]

    for path in base_paths:
        if path.exists():
            # Load with proper format for Sirena models
            df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8")

            # Fix dates
            if "Day" in df.columns:
                df["Date"] = pd.to_datetime(
                    df["Day"], format="%d.%m.%Y", errors="coerce"
                )
            elif "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

            # Fix MoM numeric format
            if "MoM" in df.columns:
                if df["MoM"].dtype == object:
                    df["MoM"] = df["MoM"].astype(str).str.replace(",", ".")
                df["MoM"] = pd.to_numeric(df["MoM"], errors="coerce")

            # Pivot to match Sirena model format
            if "Товар" in df.columns and "MoM" in df.columns:
                df = df.pivot_table(
                    index="Date", columns="Товар", values="MoM", aggfunc="first"
                )

            df = df.sort_index()

            return df

    raise FileNotFoundError("Monthly CPI data not found")


def get_week_number(date: pd.Timestamp) -> int:
    """
    Get week number in month (1-4).

    Args:
        date: Date to check

    Returns:
        Week number (1-4)
    """
    # Day of month (1-31)
    day = date.day

    # Divide into weeks: 1-7, 8-14, 15-21, 22-31
    if day <= 7:
        return 1
    elif day <= 14:
        return 2
    elif day <= 21:
        return 3
    else:
        return 4


def run_weekly_backtest(
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-01",
) -> pd.DataFrame:
    """
    Run backtest for weekly nowcaster.

    Args:
        start_date: Start date for backtest
        end_date: End date for backtest

    Returns:
        DataFrame with date, actual, prediction
    """
    print("\n" + "=" * 60)
    print("Running Weekly Nowcaster Backtest...")
    print("=" * 60)

    model = WeeklyPriceNowcaster(use_macro=True, use_components=True)

    try:
        # Fit and backtest
        backtest_results = model.backtest(start_date=start_date, end_date=end_date)

        # Calculate metrics
        metrics = model.get_metrics(backtest_results)

        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  KPI violations: {metrics['kpi_violations']}/{metrics['n_samples']}")

        # Return clean results
        results = backtest_results[["date", "actual", "prediction"]].copy()
        results["error"] = results["actual"] - results["prediction"]

        return results

    except Exception as e:
        print(f"  Error: {e}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame()


def run_monthly_backtest(
    model_name: str,
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-01",
) -> pd.DataFrame:
    """
    Run backtest for a monthly model.

    Args:
        model_name: Name of model to test
        start_date: Start date for backtest
        end_date: End date for backtest

    Returns:
        DataFrame with date, actual, prediction
    """
    print(f"\n{'=' * 60}")
    print(f"Running {model_name} Monthly Backtest...")
    print("=" * 60)

    # Load data
    monthly_df = load_monthly_data()

    # Filter to test period
    test_start = pd.Timestamp(start_date)
    test_end = pd.Timestamp(end_date)
    test_dates = monthly_df[
        (monthly_df.index >= test_start) & (monthly_df.index <= test_end)
    ].index

    if len(test_dates) == 0:
        print(f"  No test dates found in range {start_date} - {end_date}")
        return pd.DataFrame()

    # Initialize model
    if model_name == "Huber":
        from sirena.models import HuberForecaster

        model_class = HuberForecaster
    elif model_name == "Ridge":
        from sirena.models import RidgeForecaster

        model_class = RidgeForecaster
    elif model_name == "Subcomp":
        from sirena.models import SubcomponentForecaster

        model_class = SubcomponentForecaster
    elif model_name == "EBM":
        from sirena.models import EBMForecaster

        model_class = EBMForecaster
    elif model_name == "NGBoost":
        try:
            from sirena.models import NGBoostForecaster

            model_class = NGBoostForecaster
        except ImportError:
            print(f"  NGBoost not available, skipping")
            return pd.DataFrame()
    else:
        print(f"  Unknown model: {model_name}")
        return pd.DataFrame()

    results = []
    failed_preds = 0

    # Rolling backtest
    for target_date in test_dates:
        try:
            # Train on data before target_date
            train_df = monthly_df[monthly_df.index < target_date].copy()

            # Exclude outlier years
            train_df = train_df[~train_df.index.year.isin([2022])]

            if len(train_df) < 24:
                continue

            # Train model
            model = model_class()
            model.fit(train_df, "Все товары и услуги")

            # Need to extend train_df with target_date row for prediction
            # Create extended DataFrame with target_date
            train_df_ext = train_df.copy()
            train_df_ext.loc[target_date] = np.nan

            # Predict for target month
            pred_result = model.predict(train_df_ext, target_date)

            if "prediction" in pred_result and not np.isnan(pred_result["prediction"]):
                prediction = pred_result["prediction"] - 100  # Convert to MoM %
                actual = monthly_df.loc[target_date, "Все товары и услуги"] - 100

                results.append(
                    {
                        "date": target_date,
                        "actual": actual,
                        "prediction": prediction,
                    }
                )
            else:
                failed_preds += 1

        except Exception as e:
            failed_preds += 1
            # Continue to next date
            continue

    if len(results) == 0:
        print(f"  No successful predictions for {model_name} (failed: {failed_preds})")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df["error"] = results_df["actual"] - results_df["prediction"]

    mae = np.mean(np.abs(results_df["error"]))
    print(f"  MAE: {mae:.4f}")
    print(f"  Samples: {len(results_df)} (failed: {failed_preds})")

    return results_df


def test_fixed_blend(
    weekly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    weekly_weight: float,
) -> float:
    """
    Test fixed blend ratio.

    Args:
        weekly_df: Weekly backtest results
        monthly_df: Monthly backtest results
        weekly_weight: Weight for weekly (0.0-1.0)

    Returns:
        MAE of blended predictions
    """
    monthly_weight = 1.0 - weekly_weight

    # Merge on date
    merged = pd.merge(
        weekly_df[["date", "prediction"]],
        monthly_df[["date", "prediction"]],
        on="date",
        suffixes=("_weekly", "_monthly"),
        how="inner",
    )

    if len(merged) == 0:
        return np.nan

    # Calculate blend
    merged["blend_pred"] = (
        merged["prediction_weekly"] * weekly_weight
        + merged["prediction_monthly"] * monthly_weight
    )

    merged["actual"] = weekly_df.loc[
        weekly_df["date"].isin(merged["date"]), "actual"
    ].values

    # Calculate MAE
    mae = np.mean(np.abs(merged["actual"] - merged["blend_pred"]))

    return mae


def test_dynamic_blend(
    weekly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    week_weights: Dict[int, float],
) -> float:
    """
    Test dynamic blend based on week in month.

    Args:
        weekly_df: Weekly backtest results
        monthly_df: Monthly backtest results
        week_weights: Dict mapping week (1-4) to weekly weight

    Returns:
        MAE of blended predictions
    """
    # Merge on date
    merged = pd.merge(
        weekly_df[["date", "prediction"]],
        monthly_df[["date", "prediction"]],
        on="date",
        suffixes=("_weekly", "_monthly"),
        how="inner",
    )

    if len(merged) == 0:
        return np.nan

    # Add week number
    merged["week"] = merged["date"].apply(get_week_number)

    # Apply dynamic weights
    merged["weekly_weight"] = merged["week"].map(week_weights)
    merged["monthly_weight"] = 1.0 - merged["weekly_weight"]

    # Calculate blend
    merged["blend_pred"] = (
        merged["prediction_weekly"] * merged["weekly_weight"]
        + merged["prediction_monthly"] * merged["monthly_weight"]
    )

    merged["actual"] = weekly_df.loc[
        weekly_df["date"].isin(merged["date"]), "actual"
    ].values

    # Calculate MAE
    mae = np.mean(np.abs(merged["actual"] - merged["blend_pred"]))

    return mae


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("WEEKLY-MONTHLY ENSEMBLE OPTIMIZATION")
    print("=" * 80)

    # Define test period
    start_date = "2024-01-01"
    end_date = "2025-12-01"

    # Run weekly backtest
    weekly_results = run_weekly_backtest(start_date, end_date)

    if len(weekly_results) == 0:
        print("\nERROR: Weekly backtest failed!")
        return

    # Define monthly models to test
    monthly_models = ["Subcomp", "EBM", "NGBoost", "Ridge", "Huber"]

    # Store monthly-only MAE for each model
    monthly_mae_dict = {}

    # Test blend ratios
    blend_ratios = [0.20, 0.40, 0.60, 0.80]  # weekly weights

    all_results = []

    # Test each monthly model
    for model_name in monthly_models:
        print(f"\n\n{'#' * 80}")
        print(f"# Testing Ensemble: Weekly + {model_name}")
        print(f"{'#' * 80}")

        # Run monthly backtest
        monthly_results = run_monthly_backtest(model_name, start_date, end_date)

        if len(monthly_results) == 0:
            print(f"Skipping {model_name} (no predictions)")
            continue

        # Calculate monthly-only MAE
        monthly_mae = np.mean(np.abs(monthly_results["error"]))
        print(f"\nMonthly-only MAE: {monthly_mae:.4f}")

        # Store for summary later
        monthly_mae_dict[model_name] = monthly_mae

        # Test fixed blends
        fixed_results = []
        for weekly_weight in blend_ratios:
            blend_mae = test_fixed_blend(weekly_results, monthly_results, weekly_weight)

            if not np.isnan(blend_mae):
                fixed_results.append(
                    {
                        "Model": model_name,
                        "Blend_type": "Fixed",
                        "Week_in_month": "N/A",
                        "Weekly_weight": weekly_weight,
                        "Monthly_weight": 1.0 - weekly_weight,
                        "MAE": blend_mae,
                    }
                )
                print(
                    f"  Blend {weekly_weight:.0%}/{1.0 - weekly_weight:.0%}: MAE = {blend_mae:.4f}"
                )

        # Test dynamic blends
        # Strategy: Higher weekly weight later in month (more data available)
        dynamic_configs = [
            "Progressive",  # Week 1: 20%, 2: 40%, 3: 60%, 4: 80%
            "Conservative",  # Week 1-3: 40%, 4: 60%
            "Aggressive",  # Week 1-2: 40%, 3: 80%, 4: 100%
        ]

        for config_name in dynamic_configs:
            if config_name == "Progressive":
                week_weights = {1: 0.20, 2: 0.40, 3: 0.60, 4: 0.80}
            elif config_name == "Conservative":
                week_weights = {1: 0.40, 2: 0.40, 3: 0.40, 4: 0.60}
            elif config_name == "Aggressive":
                week_weights = {1: 0.40, 2: 0.40, 3: 0.80, 4: 1.00}

            blend_mae = test_dynamic_blend(
                weekly_results, monthly_results, week_weights
            )

            if not np.isnan(blend_mae):
                all_results.append(
                    {
                        "Model": model_name,
                        "Blend_type": f"Dynamic_{config_name}",
                        "Week_in_month": str(week_weights),
                        "Weekly_weight": np.nan,
                        "Monthly_weight": np.nan,
                        "MAE": blend_mae,
                    }
                )
                print(f"  Dynamic {config_name}: MAE = {blend_mae:.4f}")

        # Find best fixed blend
        if len(fixed_results) > 0:
            best_fixed = min(fixed_results, key=lambda x: x["MAE"])

        all_results.extend(fixed_results)

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    if len(results_df) == 0:
        print("\nERROR: No results generated!")
        return

    # Save results
    output_path = (
        Path(__file__).parent.parent / "data" / "weekly_monthly_blend_results.csv"
    )
    results_df.to_csv(output_path, index=False)

    print(f"\n\n{'=' * 80}")
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nResults saved to: {output_path}")
    print(f"Total combinations tested: {len(results_df)}")

    # Find overall best
    best_overall = results_df.loc[results_df["MAE"].idxmin()]

    print(f"\nBest overall combination:")
    print(f"  Model: {best_overall['Model']}")
    print(f"  Blend type: {best_overall['Blend_type']}")
    print(f"  MAE: {best_overall['MAE']:.4f}")

    # Summary by model (fixed blends only)
    fixed_only = results_df[results_df["Blend_type"] == "Fixed"].copy()

    print("\n\nBest Fixed Blend by Model:")
    for model in monthly_models:
        model_data = fixed_only[fixed_only["Model"] == model]
        if len(model_data) > 0:
            best = model_data.loc[model_data["MAE"].idxmin()]
            monthly_mae_for_model = monthly_mae_dict.get(model, np.nan)
            improvement = (
                (monthly_mae_for_model - best["MAE"]) / monthly_mae_for_model * 100
                if not np.isnan(monthly_mae_for_model) and monthly_mae_for_model > 0
                else 0
            )
            print(f"\n  {model}:")
            print(
                f"    Best blend: {best['Weekly_weight']:.0%} weekly, {best['Monthly_weight']:.0%} monthly"
            )
            print(f"    Blend MAE: {best['MAE']:.4f}")
            print(f"    Monthly-only MAE: {monthly_mae_for_model:.4f}")
            print(f"    Improvement: {improvement:.1f}%")

    # Compare fixed vs dynamic
    print("\n\nFixed vs Dynamic Comparison:")
    fixed_avg = fixed_only["MAE"].mean()
    dynamic_only = results_df[results_df["Blend_type"].str.startswith("Dynamic")]
    dynamic_avg = dynamic_only["MAE"].mean() if len(dynamic_only) > 0 else np.nan

    print(f"  Fixed blend avg MAE: {fixed_avg:.4f}")
    print(f"  Dynamic blend avg MAE: {dynamic_avg:.4f}")

    if not np.isnan(dynamic_avg):
        diff = (dynamic_avg - fixed_avg) / fixed_avg * 100
        print(f"  Difference: {diff:.1f}%")
        print(f"  Winner: {'Fixed' if fixed_avg < dynamic_avg else 'Dynamic'}")


if __name__ == "__main__":
    main()
