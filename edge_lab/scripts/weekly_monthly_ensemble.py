#!/usr/bin/env python3
"""
Weekly-Monthly Ensemble Optimization
===================================

Optimizes blend of weekly nowcast with monthly model forecasts.

Methodology:
1. Load backtest predictions from top 5 monthly models
2. Generate weekly nowcast predictions for same period
3. Test blend ratios: 20/80, 40/60, 60/40, 80/20 (weekly/monthly)
4. Test dynamic blend based on weeks elapsed in month
5. Find optimal blend that minimizes MAE

Monthly models: Huber, Ridge, NGBoost, Subcomp, EBM
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, List, Tuple
from datetime import datetime

warnings.filterwarnings("ignore")

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sirena.models.weekly_prices import WeeklyPriceNowcaster
    from sirena.data.weekly_loader import load_weekly_prices
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sirena"))
    from models.weekly_prices import WeeklyPriceNowcaster
    from data.weekly_loader import load_weekly_prices


def load_monthly_backtest() -> pd.DataFrame:
    """Load h=1 backtest predictions for top 5 monthly models."""
    # Try multiple possible paths
    base_paths = [
        Path(__file__).parent.parent
        / "archive"
        / "results"
        / "backtest_h1_predictions.csv",
        Path(__file__).parent.parent.parent
        / "archive"
        / "results"
        / "backtest_h1_predictions.csv",
        Path.cwd() / "archive" / "results" / "backtest_h1_predictions.csv",
    ]

    backtest_path = None
    for path in base_paths:
        if path.exists():
            backtest_path = path
            break

    if backtest_path is None:
        raise FileNotFoundError(
            f"Backtest file not found. Tried:\n"
            + "\n".join(f"  - {p}" for p in base_paths)
        )
    if not backtest_path.exists():
        raise FileNotFoundError(f"Backtest file not found: {backtest_path}")

    df = pd.read_csv(backtest_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    # Select top 5 models and Actual
    models = ["Actual", "Huber", "Ridge", "NGBoost", "Subcomp", "EBM"]

    for col in models:
        if col not in df.columns:
            print(
                f"Warning: {col} not found in backtest, available: {df.columns.tolist()[:5]}"
            )

    return df[[c for c in models if c in df.columns]].dropna(subset=["Actual"])


def generate_weekly_nowcasts(backtest_dates: pd.DatetimeIndex) -> pd.Series:
    """Generate weekly nowcast predictions for backtest period.

    Using WeeklyPriceNowcaster trained on historical data.
    """
    # Determine training period: all data before first backtest date
    first_backtest = backtest_dates.min()
    train_end = first_backtest - pd.DateOffset(months=1)

    print(f"Training weekly model: 2016-01-01 to {train_end.strftime('%Y-%m-%d')}")

    # Initialize and fit weekly model
    weekly_model = WeeklyPriceNowcaster(
        alpha=1.0,
        use_macro=True,
        use_components=True,
        regressor="ridge",
    )

    try:
        weekly_model.fit(
            start_date="2016-01-01",
            end_date=str(train_end.date()),
            exclude_years=[2022],
        )
    except Exception as e:
        print(f"Warning: Failed to fit weekly model: {e}")
        return pd.Series(np.nan, index=backtest_dates)

    # Generate nowcasts for each backtest date
    weekly_predictions = []
    for target_date in backtest_dates:
        try:
            pred = weekly_model.nowcast(target_date)
            weekly_predictions.append(
                {
                    "Date": target_date,
                    "weekly_pred": pred["prediction"],
                    "coverage": pred.get("coverage", 0.95),
                }
            )
        except Exception as e:
            print(f"Warning: Nowcast failed for {target_date}: {e}")
            weekly_predictions.append(
                {
                    "Date": target_date,
                    "weekly_pred": np.nan,
                    "coverage": 0.0,
                }
            )

    weekly_df = pd.DataFrame(weekly_predictions).set_index("Date")
    return weekly_df["weekly_pred"]


def compute_week_of_month(date: pd.Timestamp) -> int:
    """Compute week of month (1-5) for a given date."""
    # First day of month
    first_day = date.replace(day=1)

    # Week number (1-5)
    week = ((date.day - 1) // 7) + 1
    return min(week, 5)  # Cap at 5


def test_fixed_blends(
    monthly_df: pd.DataFrame,
    weekly_series: pd.Series,
    blend_ratios: List[float],
) -> pd.DataFrame:
    """Test fixed blend ratios (weekly/monthly).

    Args:
        monthly_df: DataFrame with actual and model predictions
        weekly_series: Weekly nowcast predictions
        blend_ratios: List of weekly weights (0.0-1.0)

    Returns:
        DataFrame with MAE for each blend ratio and each monthly model
    """
    results = []

    for weekly_weight in blend_ratios:
        monthly_weight = 1.0 - weekly_weight

        for model in ["Huber", "Ridge", "NGBoost", "Subcomp", "EBM"]:
            if model not in monthly_df.columns:
                continue

            # Compute ensemble: weekly_weight * weekly + monthly_weight * monthly
            ensemble = (
                weekly_weight * weekly_series + monthly_weight * monthly_df[model]
            )

            # Calculate MAE
            mae = np.mean(np.abs(monthly_df["Actual"] - ensemble))

            # Format blend ratio as "80/20" (weekly/monthly)
            # Round to handle floating point precision issues
            weekly_pct = int(round(weekly_weight * 100))
            monthly_pct = int(round(monthly_weight * 100))
            blend_ratio = f"{weekly_pct}/{monthly_pct}"

            results.append(
                {
                    "Blend_type": "fixed",
                    "Blend_ratio": blend_ratio,
                    "Week_in_month": "N/A",  # Not applicable for fixed blend
                    "Weekly_weight": weekly_weight,
                    "Monthly_weight": monthly_weight,
                    "Model": model,
                    "MAE": mae,
                }
            )

    return pd.DataFrame(results)


def test_dynamic_blends(
    monthly_df: pd.DataFrame,
    weekly_series: pd.Series,
) -> pd.DataFrame:
    """Test dynamic blend based on week of month.

    Logic: Later in month = more weight to weekly nowcast
    - Week 1: 10% weekly, 90% monthly
    - Week 2: 20% weekly, 80% monthly
    - Week 3: 40% weekly, 60% monthly
    - Week 4: 60% weekly, 40% monthly
    - Week 5: 80% weekly, 20% monthly

    Args:
        monthly_df: DataFrame with actual and model predictions
        weekly_series: Weekly nowcast predictions

    Returns:
        DataFrame with MAE for dynamic blend strategy
    """
    # Map week of month to weights
    week_weights = {
        1: (0.10, 0.90),  # (weekly, monthly)
        2: (0.20, 0.80),
        3: (0.40, 0.60),
        4: (0.60, 0.40),
        5: (0.80, 0.20),
    }

    results = []

    for model in ["Huber", "Ridge", "NGBoost", "Subcomp", "EBM"]:
        if model not in monthly_df.columns:
            continue

        # Calculate dynamic ensemble for each date
        ensemble_values = []
        for date in monthly_df.index:
            week = compute_week_of_month(date)
            w_weekly, w_monthly = week_weights.get(week, (0.40, 0.60))

            weekly_pred = weekly_series.get(date, np.nan)
            monthly_pred = monthly_df.at[date, model]

            if pd.isna(weekly_pred) or pd.isna(monthly_pred):
                ensemble_values.append(np.nan)
            else:
                ensemble = w_weekly * weekly_pred + w_monthly * monthly_pred
                ensemble_values.append(ensemble)

        # Calculate MAE (excluding NaN)
        ensemble_series = pd.Series(ensemble_values, index=monthly_df.index)
        valid_mask = ~(pd.isna(ensemble_series) | pd.isna(monthly_df["Actual"]))
        mae = np.mean(
            np.abs(monthly_df["Actual"][valid_mask] - ensemble_series[valid_mask])
        )

        results.append(
            {
                "Blend_type": "dynamic",
                "Blend_ratio": "dynamic",  # Varies by week
                "Week_in_month": "1-5",  # Uses all 5 weeks
                "Weekly_weight": "dynamic",
                "Monthly_weight": "dynamic",
                "Model": model,
                "MAE": mae,
                "Strategy": "week_based",
            }
        )

    return pd.DataFrame(results)


def calculate_monthly_ensemble(
    monthly_df: pd.DataFrame,
    weekly_series: pd.Series,
    weekly_weight: float = 0.15,
) -> pd.Series:
    """Calculate ensemble of monthly models + weekly nowcast.

    Monthly ensemble = average of top 5 monthly models
    Final = weekly_weight * weekly + (1-weekly_weight) * monthly_ensemble

    Args:
        monthly_df: DataFrame with actual and model predictions
        weekly_series: Weekly nowcast predictions
        weekly_weight: Weight for weekly nowcast (default 0.15 = 15%)

    Returns:
        Series with ensemble predictions
    """
    monthly_models = ["Huber", "Ridge", "NGBoost", "Subcomp", "EBM"]
    available_models = [m for m in monthly_models if m in monthly_df.columns]

    if not available_models:
        raise ValueError(f"No monthly models available: {monthly_df.columns.tolist()}")

    # Average of monthly models
    monthly_ensemble = monthly_df[available_models].mean(axis=1)

    # Blend with weekly
    final_ensemble = (
        weekly_weight * weekly_series + (1 - weekly_weight) * monthly_ensemble
    )

    return final_ensemble


def main():
    """Main execution."""
    print("=" * 60)
    print("Weekly-Monthly Ensemble Optimization")
    print("=" * 60)

    # 1. Load monthly backtest predictions
    print("\n[1/5] Loading monthly backtest predictions...")
    monthly_df = load_monthly_backtest()
    print(
        f"Loaded {len(monthly_df)} predictions from {monthly_df.index.min()} to {monthly_df.index.max()}"
    )

    # Check which models are available
    available_models = [
        c
        for c in ["Huber", "Ridge", "NGBoost", "Subcomp", "EBM"]
        if c in monthly_df.columns
    ]
    print(f"Available monthly models: {available_models}")

    if len(available_models) < 3:
        print("Warning: Less than 3 monthly models available")

    # 2. Generate weekly nowcast predictions
    print("\n[2/5] Generating weekly nowcast predictions...")
    weekly_series = generate_weekly_nowcasts(monthly_df.index)
    print(f"Generated {len(weekly_series.dropna())} weekly nowcasts")

    # Align data
    combined_df = monthly_df.copy()
    combined_df["weekly"] = weekly_series
    combined_df = combined_df.dropna(subset=["weekly"])

    print(f"Aligned dataset: {len(combined_df)} observations")

    if len(combined_df) < 3:
        print("Error: Not enough data for analysis")
        return

    # 3. Test fixed blend ratios
    print("\n[3/5] Testing fixed blend ratios...")
    blend_ratios = [0.20, 0.40, 0.60, 0.80]  # Weekly weights
    fixed_results = test_fixed_blends(combined_df, combined_df["weekly"], blend_ratios)
    print(f"Tested {len(fixed_results)} fixed blend configurations")

    # 4. Test dynamic blend
    print("\n[4/5] Testing dynamic blend (week-based)...")
    dynamic_results = test_dynamic_blends(combined_df, combined_df["weekly"])
    print(f"Tested {len(dynamic_results)} dynamic blend configurations")

    # 5. Combine and analyze results
    print("\n[5/5] Analyzing results...")
    all_results = pd.concat([fixed_results, dynamic_results], ignore_index=True)

    # Save results
    output_path = (
        Path(__file__).parent.parent / "data" / "weekly_monthly_blend_results.csv"
    )
    all_results.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Find optimal blend for each model
    print("\n" + "=" * 60)
    print("OPTIMAL BLENDS PER MODEL")
    print("=" * 60)

    for model in available_models:
        model_results = all_results[all_results["Model"] == model]

        if len(model_results) == 0:
            continue

        best_idx = model_results["MAE"].idxmin()
        best = model_results.loc[best_idx]

        print(f"\n{model}:")
        print(f"  Best Blend: {best['Blend_type']}")
        if best["Blend_type"] == "fixed":
            print(f"  Weekly Weight: {best['Weekly_weight']:.0%}")
            print(f"  Monthly Weight: {best['Monthly_weight']:.0%}")
        print(f"  MAE: {best['MAE']:.4f}")

    # Baseline comparison (monthly only vs optimal blend)
    print("\n" + "=" * 60)
    print("BASELINE vs OPTIMAL")
    print("=" * 60)

    for model in available_models:
        # Monthly-only baseline
        monthly_only_mae = np.mean(np.abs(combined_df["Actual"] - combined_df[model]))

        # Best blend
        model_results = all_results[all_results["Model"] == model]
        best_mae = model_results["MAE"].min()

        # Improvement
        improvement = (monthly_only_mae - best_mae) / monthly_only_mae * 100

        print(
            f"{model:12s}: MAE={monthly_only_mae:.4f} → {best_mae:.4f} ({improvement:+.1f}%)"
        )

    # Overall best
    print("\n" + "=" * 60)
    print("OVERALL BEST")
    print("=" * 60)

    best_overall_idx = all_results["MAE"].idxmin()
    best_overall = all_results.loc[best_overall_idx]

    print(f"Model: {best_overall['Model']}")
    print(f"Blend: {best_overall['Blend_type']}")
    if best_overall["Blend_type"] == "fixed":
        print(f"Weekly Weight: {best_overall['Weekly_weight']:.0%}")
        print(f"Monthly Weight: {best_overall['Monthly_weight']:.0%}")
    print(f"MAE: {best_overall['MAE']:.4f}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    # Compare fixed vs dynamic
    fixed_avg = all_results[all_results["Blend_type"] == "fixed"]["MAE"].mean()
    dynamic_avg = all_results[all_results["Blend_type"] == "dynamic"]["MAE"].mean()

    print(f"Average MAE (fixed blends): {fixed_avg:.4f}")
    print(f"Average MAE (dynamic blend): {dynamic_avg:.4f}")
    print(
        f"Difference: {fixed_avg - dynamic_avg:.4f} ({((fixed_avg - dynamic_avg) / fixed_avg * 100):+.1f}%)"
    )

    # Count of models improved by ensemble
    improved = 0
    for model in available_models:
        monthly_only_mae = np.mean(np.abs(combined_df["Actual"] - combined_df[model]))
        best_mae = all_results[all_results["Model"] == model]["MAE"].min()
        if best_mae < monthly_only_mae:
            improved += 1

    print(f"\nModels improved by ensemble: {improved}/{len(available_models)}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
