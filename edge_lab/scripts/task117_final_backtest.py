#!/usr/bin/env python3
"""
Task 117: Production - Integrate New Regressors into Sirena

This script backtests enhanced models (with monthly OPR features) vs baseline models
to compare MAE and verify improvement.

Key steps:
1. Load enhanced data with monthly OPR features from regressor_priority_list.csv
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

from sirena.data.enhanced_loader import load_enhanced_data
from sirena.models.opr_enhanced_ridge import OPREnhancedRidgeForecaster


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

    # Identify OPR feature columns
    opr_cols = [c for c in df.columns if c.startswith("opr_")]
    print(f"\nAvailable OPR features: {len(opr_cols)}")
    for col in opr_cols:
        coverage = (df[col].notna().sum() / len(df)) * 100
        print(f"  - {col}: {coverage:.1f}% coverage")

    # Test baseline Ridge (no OPR)
    print("\n[1/2] Testing Baseline RidgeForecaster (no OPR)...")
    baseline_ridge = OPREnhancedRidgeForecaster(
        alpha=1.0, use_huber=False, use_opr=False
    )
    baseline_ridge_metrics = evaluate_model(
        baseline_ridge, baseline_df, start_date="2019-01-01", target_col="mom"
    )
    print(f"  MAE: {baseline_ridge_metrics['MAE']:.4f}")
    print(f"  RMSE: {baseline_ridge_metrics['RMSE']:.4f}")
    print(f"  KPI: {baseline_ridge_metrics['KPI']:.1f}%")
    print(f"  Samples: {baseline_ridge_metrics['n_samples']}")

    # Test enhanced Ridge (with OPR features)
    print("\n[2/2] Testing Enhanced RidgeForecaster (with OPR)...")
    enhanced_ridge = OPREnhancedRidgeForecaster(
        alpha=1.0, use_huber=False, use_opr=True, opr_lag=1
    )
    enhanced_ridge_metrics = evaluate_model(
        enhanced_ridge, df, start_date="2019-01-01", target_col="mom"
    )
    print(f"  MAE: {enhanced_ridge_metrics['MAE']:.4f}")
    print(f"  RMSE: {enhanced_ridge_metrics['RMSE']:.4f}")
    print(f"  KPI: {enhanced_ridge_metrics['KPI']:.1f}%")
    print(f"  Samples: {enhanced_ridge_metrics['n_samples']}")

    # Get feature importance if model fitted
    if (
        hasattr(enhanced_ridge, "_is_fitted")
        and enhanced_ridge._is_fitted
        and enhanced_ridge_metrics["n_samples"] > 0
    ):
        importance = enhanced_ridge.get_feature_importance()
        print("\n  Top 10 features (by coefficient):")
        print(importance.head(10).to_string(index=False))

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: Model Performance Comparison")
    print("=" * 70)

    baseline_mae = baseline_ridge_metrics["MAE"]
    target_mae = 0.236
    goal_mae = 0.22

    print(f"\nBaseline MAE (target): {target_mae:.3f}")
    print(f"Goal MAE: {goal_mae:.3f}")
    print(f"Baseline Ridge MAE: {baseline_ridge_metrics['MAE']:.4f}")
    print(f"Enhanced Ridge MAE: {enhanced_ridge_metrics['MAE']:.4f}")

    ridge_improvement = (
        ((baseline_mae - enhanced_ridge_metrics["MAE"]) / baseline_mae) * 100
        if baseline_mae > 0
        else 0
    )
    print(f"\nRidge Improvement: {ridge_improvement:+.2f}%")

    # Goal achievement
    print(
        f"\nGoal (MAE < 0.22): {'ACHIEVED' if enhanced_ridge_metrics['MAE'] < goal_mae else 'NOT ACHIEVED'}"
    )

    # Document OPR features used
    print(
        f"\nOPR Features Integrated: {sum(1 for c in df.columns if c.startswith('opr_'))} features"
    )

    # Determine if OPR features were actually used
    opr_features_used = 0
    # Fit a single model to check which features would be used
    test_model = OPREnhancedRidgeForecaster(
        alpha=1.0, use_huber=False, use_opr=True, opr_lag=1
    )
    try:
        test_model.fit(df, "mom")
        if hasattr(test_model, "_available_features"):
            opr_features_used = sum(
                1 for f in test_model._available_features if f.startswith("opr_")
            )
    except:
        pass

    print(f"OPR Features Used in Model: {opr_features_used} features")

    return {
        "baseline_ridge": baseline_ridge_metrics,
        "enhanced_ridge": enhanced_ridge_metrics,
        "target_mae": target_mae,
        "goal_mae": goal_mae,
        "ridge_improvement_pct": ridge_improvement,
        "opr_features_count": sum(1 for c in df.columns if c.startswith("opr_")),
        "opr_features_used": opr_features_used,
        "opr_cols": opr_cols,
    }


def generate_report(results, df):
    """Generate markdown report with results."""
    report = f"""# Task 117: Production - Integrate New Regressors into Sirena

## Summary

This report documents the integration of OPR (Official Price Reporting) features into Ridge regression models for inflation forecasting.

## OPR Features Used

Features from `regressor_priority_list.csv` (Task 116):
- Count: {results["opr_features_count"]} features
- Type: Monthly (м/м) indices only
- Features used in model: {results["opr_features_used"]} features

OPR Features:
"""

    for col in results["opr_cols"]:
        coverage = (df[col].notna().sum() / len(df)) * 100
        report += f"- `{col}`: {coverage:.1f}% coverage\n"

    report += f"""
## Model Performance Comparison

### Baseline (No OPR Features)
- MAE: {results["baseline_ridge"]["MAE"]:.4f}
- RMSE: {results["baseline_ridge"]["RMSE"]:.4f}
- KPI: {results["baseline_ridge"]["KPI"]:.1f}%
- Samples: {results["baseline_ridge"]["n_samples"]}

### Enhanced (With OPR Features)
- MAE: {results["enhanced_ridge"]["MAE"]:.4f}
- RMSE: {results["enhanced_ridge"]["RMSE"]:.4f}
- KPI: {results["enhanced_ridge"]["KPI"]:.1f}%
- Samples: {results["enhanced_ridge"]["n_samples"]}

### Analysis
- Baseline Target MAE: {results["target_mae"]:.3f}
- Goal MAE: {results["goal_mae"]:.3f}
- Improvement: {results["ridge_improvement_pct"]:+.2f}%
- Goal Achieved: {"YES" if results["enhanced_ridge"]["MAE"] < results["goal_mae"] else "NO"}

"""

    # Add analysis of why features helped or didn't help
    if results["ridge_improvement_pct"] > 0:
        report += """## Results

✅ OPR features provided a measurable improvement in forecasting accuracy.

### Why Features Helped

Monthly OPR features (sectoral CPI components) provide:
1. **Leading indicators**: Sector-specific price movements often precede overall inflation
2. **Granular information**: Captures price dynamics in specific categories
3. **Different timing**: Some components (e.g., fuel, food) react faster to supply shocks
4. **No look-ahead bias**: Using lag=1 ensures features are available at forecast time

"""
    else:
        report += f"""## Results

⚠️ OPR features did not provide measurable improvement (change: {results["ridge_improvement_pct"]:+.2f}%).

### Why Features May Not Have Helped

1. **High Correlation with Target**: Monthly OPR features are CPI components, so they correlate strongly with overall inflation (correlation > 0.92)
2. **Redundancy**: Sectoral CPI information may already be captured in autoregressive features (mom_L1, mom_L2, mom_L3)
3. **Lag Structure**: The optimal lag (1) means features are very recent, limiting their predictive power
4. **Feature Selection**: The top 3 monthly features from Task 116 all measure CPI components, not independent predictors like:
   - Production indices (correlation ~ 0.02)
   - Economic leading indicators
   - Commodity prices

### Alternative Approaches

To improve forecasting beyond current baselines, consider:
1. **Production/Construction indices**: These have economic fundamentals but low correlation with CPI
2. **Non-CPI indicators**: Survey data, business confidence, expectations
3. **Advanced lag analysis**: Cross-correlation at longer horizons (6-12 months)
4. **Non-linear models**: Tree-based models or neural networks that capture interactions

"""

    return report


def main():
    """Main execution function."""
    print("Task 117: Production - Integrate New Regressors into Sirena")
    print("=" * 70)

    # Load enhanced data
    print("\nLoading enhanced data with monthly OPR features...")
    df = load_enhanced_data(top_opr_features=3)  # Only 3 monthly features available
    print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")

    # Run comparison
    results = test_baseline_vs_enhanced(df)

    # Generate and save report
    report = generate_report(results, df)
    report_path = Path("data/task117_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")

    # Save results as JSON
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
            elif isinstance(value, list):
                serializable[key] = value
            else:
                serializable[key] = (
                    float(value)
                    if isinstance(value, (int, float, np.number))
                    else value
                )
        json.dump(serializable, f, indent=2)

    print(f"Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    main()
