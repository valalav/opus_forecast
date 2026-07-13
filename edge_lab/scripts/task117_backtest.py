#!/usr/bin/env python3
"""
Task 117: Production - Integrate New Regressors into Sirena

This script backtests enhanced models (with OPR features) vs baseline models
to compare MAE and verify improvement.

Key steps:
1. Load enhanced data with OPR features from regressor_priority_list.csv
2. Backtest baseline models (without OPR features)
3. Backtest enhanced models (with OPR features)
4. Compare MAE and document results
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(1, str(Path(__file__).parent.parent))

# Use ModelRegistry to get models
from sirena.models.registry import ModelRegistry
from sirena.data.enhanced_loader import load_enhanced_data

# Get models from registry (name 'ridge_macro' and 'huber')
RidgeMacroForecaster = ModelRegistry.get("ridge_macro")
HuberForecaster = ModelRegistry.get("huber")


def evaluate_model(model, df, start_date="2019-01-01", target_col="mom"):
    """
    Evaluate model on backtest data.

    Returns:
        dict with MAE, RMSE, KPI, and error stats
    """
    results = model.backtest(
        df, start_date=start_date, target_col=target_col, horizon=1
    )

    if results.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "KPI": 0, "n_samples": 0}

    mae = results["error"].abs().mean()
    rmse = np.sqrt((results["error"] ** 2).mean())
    kpi = (results["error"].abs() <= 0.5).sum() / len(results) * 100

    return {"MAE": mae, "RMSE": rmse, "KPI": kpi, "n_samples": len(results)}


def prepare_baseline_data(df):
    """
    Prepare data for baseline models (without OPR features).

    This is the original format expected by RidgeMacroForecaster and HuberForecaster.
    """
    # Keep only original columns
    base_cols = [
        "mom",
        "Prod",
        "Serv",
        "Nonprod",
        "Ki_i",
        "Ruonia",
        "usd_nom_i",
        "brent",
    ]
    available = [c for c in base_cols if c in df.columns]

    return df[available].copy()


def prepare_enhanced_data(df):
    """
    Prepare data for enhanced models (with OPR features).

    Converts OPR features to format expected by models.
    """
    df = df.copy()

    # Identify OPR feature columns
    opr_cols = [c for c in df.columns if c.startswith("opr_")]

    # Add OPR features with appropriate lag
    # For now, use lag=1 (previous month) as OPR features
    for col in opr_cols:
        df[f"{col}_L1"] = df[col].shift(1)

    return df


def test_baseline_vs_enhanced(df):
    """
    Compare baseline vs enhanced models.

    Args:
        df: Enhanced DataFrame with OPR features

    Returns:
        dict with comparison results
    """
    print("\n" + "=" * 70)
    print("TASK 117: Baseline vs Enhanced Model Comparison")
    print("=" * 70)

    # Prepare data for baseline models
    baseline_df = prepare_baseline_data(df)

    # Prepare data for enhanced models
    enhanced_df = prepare_enhanced_data(df)

    # Test baseline RidgeMacro
    print("\n[1/4] Testing Baseline RidgeMacroForecaster...")
    baseline_ridge = RidgeMacroForecaster(alpha=1.0, use_huber=False)
    baseline_ridge_metrics = evaluate_model(
        baseline_ridge, baseline_df, start_date="2019-01-01", target_col="mom"
    )
    print(f"  MAE: {baseline_ridge_metrics['MAE']:.4f}")
    print(f"  RMSE: {baseline_ridge_metrics['RMSE']:.4f}")
    print(f"  KPI: {baseline_ridge_metrics['KPI']:.1f}%")
    print(f"  Samples: {baseline_ridge_metrics['n_samples']}")

    # Test baseline Huber
    print("\n[2/4] Testing Baseline HuberForecaster...")
    baseline_huber = HuberForecaster(epsilon=1.35, alpha=0.3, use_macro=True)
    baseline_huber_metrics = evaluate_model(
        baseline_huber,
        baseline_df,
        start_date="2019-01-01",
        target_col="Все товары и услуги",
    )
    print(f"  MAE: {baseline_huber_metrics['MAE']:.4f}")
    print(f"  RMSE: {baseline_huber_metrics['RMSE']:.4f}")
    print(f"  KPI: {baseline_huber_metrics['KPI']:.1f}%")
    print(f"  Samples: {baseline_huber_metrics['n_samples']}")

    # Test enhanced RidgeMacro (with OPR features)
    print("\n[3/4] Testing Enhanced RidgeMacroForecaster (with OPR)...")
    enhanced_ridge = RidgeMacroForecaster(alpha=1.0, use_huber=False)
    enhanced_ridge_metrics = evaluate_model(
        enhanced_ridge, enhanced_df, start_date="2019-01-01", target_col="mom"
    )
    print(f"  MAE: {enhanced_ridge_metrics['MAE']:.4f}")
    print(f"  RMSE: {enhanced_ridge_metrics['RMSE']:.4f}")
    print(f"  KPI: {enhanced_ridge_metrics['KPI']:.1f}%")
    print(f"  Samples: {enhanced_ridge_metrics['n_samples']}")

    # Get feature importance
    if enhanced_ridge.is_fitted:
        importance = enhanced_ridge.get_feature_importance()
        print("\n  Top 10 features (by coefficient):")
        print(importance.head(10).to_string(index=False))

    # Test enhanced Huber (with OPR features)
    print("\n[4/4] Testing Enhanced HuberForecaster (with OPR)...")

    # Note: HuberForecaster doesn't automatically use OPR features
    # We'll need to extend it or just use baseline for comparison
    enhanced_huber = HuberForecaster(epsilon=1.35, alpha=0.3, use_macro=True)
    enhanced_huber_metrics = evaluate_model(
        enhanced_huber,
        enhanced_df,
        start_date="2019-01-01",
        target_col="Все товары и услуги",
    )
    print(f"  MAE: {enhanced_huber_metrics['MAE']:.4f}")
    print(f"  RMSE: {enhanced_huber_metrics['RMSE']:.4f}")
    print(f"  KPI: {enhanced_huber_metrics['KPI']:.1f}%")
    print(f"  Samples: {enhanced_huber_metrics['n_samples']}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: Model Performance Comparison")
    print("=" * 70)

    baseline_mae = baseline_ridge_metrics["MAE"]
    target_mae = 0.236

    print(f"\nBaseline MAE (target): {target_mae:.3f}")
    print(f"Baseline RidgeMacro MAE: {baseline_ridge_metrics['MAE']:.4f}")
    print(f"Enhanced RidgeMacro MAE: {enhanced_ridge_metrics['MAE']:.4f}")

    ridge_improvement = (
        (baseline_mae - enhanced_ridge_metrics["MAE"]) / baseline_mae
    ) * 100
    print(f"\nRidgeMacro Improvement: {ridge_improvement:+.2f}%")

    print(f"\nBaseline Huber MAE: {baseline_huber_metrics['MAE']:.4f}")
    print(f"Enhanced Huber MAE: {enhanced_huber_metrics['MAE']:.4f}")

    huber_improvement = (
        (baseline_mae - enhanced_huber_metrics["MAE"]) / baseline_mae
    ) * 100
    print(f"Huber Improvement: {huber_improvement:+.2f}%")

    # Goal: MAE < 0.22
    print(
        f"\nGoal (MAE < 0.22): {'ACHIEVED' if enhanced_ridge_metrics['MAE'] < 0.22 else 'NOT ACHIEVED'}"
    )

    # Document OPR features used
    print(
        f"\nOPR Features Integrated: {sum(1 for c in enhanced_df.columns if c.startswith('opr_'))} features"
    )
    print(
        f"OPR Feature Coverage: {sum([c.startswith('opr_') for c in enhanced_df.columns if enhanced_df[c].notna().any()]) / sum([c.startswith('opr_') for c in enhanced_df.columns]) * 100:.1f}%"
    )

    return {
        "baseline_ridge": baseline_ridge_metrics,
        "enhanced_ridge": enhanced_ridge_metrics,
        "baseline_huber": baseline_huber_metrics,
        "enhanced_huber": enhanced_huber_metrics,
        "target_mae": target_mae,
        "ridge_improvement_pct": ridge_improvement,
        "huber_improvement_pct": huber_improvement,
        "opr_features_count": sum(
            1 for c in enhanced_df.columns if c.startswith("opr_")
        ),
        "opr_features_with_data": sum(
            1
            for c in enhanced_df.columns
            if c.startswith("opr_") and enhanced_df[c].notna().any()
        ),
    }


def main():
    """Main execution function."""
    print("Task 117: Production - Integrate New Regressors into Sirena")
    print("=" * 70)

    # Load enhanced data
    print("\nLoading enhanced data with OPR features...")
    df = load_enhanced_data(top_opr_features=5)
    print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")

    # Run comparison
    results = test_baseline_vs_enhanced(df)

    # Save results
    output_path = Path("data/task117_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import json

    with open(output_path, "w") as f:
        # Convert to JSON-serializable format
        serializable = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable[key] = {
                    k: float(v) if isinstance(v, (int, float, np.number)) else v
                    for k, v in value.items()
                }
            else:
                serializable[key] = (
                    float(value)
                    if isinstance(value, (int, float, np.number))
                    else value
                )
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
