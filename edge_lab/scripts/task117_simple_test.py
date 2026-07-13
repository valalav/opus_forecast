#!/usr/bin/env python3
"""
Task 117 Simple Test: Direct model backtest comparison

Tests baseline RidgeMacro vs enhanced (with OPR features)
"""

import sys

sys.path.insert(0, "/home/valalav/_projects/sirena-kbr")

import pandas as pd
import numpy as np
from pathlib import Path

# Direct import from sirena.models
from sirena.models.ridge_macro import RidgeMacroForecaster
from sirena.models.huber import HuberForecaster

# Import enhanced data directly
import sys
sys.path.insert(0, '/home/valalav/_projects/sirena-kbr/edge_lab/sirena/data'))
from enhanced_loader import load_enhanced_data


def test_model(model_name, ModelClass, df, start_date="2019-01-01", target_col="mom"):
    """Test a model."""
    print(f"\nTesting {model_name}...")
    model = ModelClass()

    # Prepare data (different for baseline vs enhanced)
    if "Enhanced" in model_name:
        # Use OPR features
        opr_cols = [c for c in df.columns if c.startswith("opr_")]
        df_prep = df.copy()
        for col in opr_cols:
            df_prep[f"{col}_L1"] = df_prep[col].shift(1)
    else:
        # Baseline - keep only base columns
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
        df_prep = df[[c for c in base_cols if c in df.columns]].copy()

    # Run backtest
    results = model.backtest(
        df_prep, start_date=start_date, target_col=target_col, horizon=1
    )

    if results.empty:
        return {"MAE": np.nan, "n_samples": 0}

    mae = results["error"].abs().mean()
    return {
        "MAE": mae,
        "n_samples": len(results),
        "n_features": len(getattr(model, "_available_features", [])),
    }


def main():
    """Main function."""
    print("Task 117: Simple Model Comparison Test")
    print("=" * 60)

    # Load enhanced data
    df = load_enhanced_data(top_opr_features=5)
    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")

    # Test baseline RidgeMacro
    baseline_ridge = test_model("Baseline RidgeMacro", RidgeMacroForecaster, df)
    print(
        f"MAE: {baseline_ridge['MAE']:.4f}, n_features: {baseline_ridge['n_features']}"
    )

    # Test enhanced RidgeMacro
    enhanced_ridge = test_model("Enhanced RidgeMacro", RidgeMacroForecaster, df)
    print(
        f"MAE: {enhanced_ridge['MAE']:.4f}, n_features: {enhanced_ridge['n_features']}"
    )

    # Test baseline Huber
    baseline_huber = test_model("Baseline Huber", HuberForecaster, df)
    print(
        f"MAE: {baseline_huber['MAE']:.4f}, n_features: {baseline_huber['n_features']}"
    )

    # Test enhanced Huber
    enhanced_huber = test_model("Enhanced Huber", HuberForecaster, df)
    print(
        f"MAE: {enhanced_huber['MAE']:.4f}, n_features: {enhanced_huber['n_features']}"
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    target_mae = 0.236
    print(f"\nTarget MAE: {target_mae:.3f}")
    print(f"Baseline RidgeMacro MAE: {baseline_ridge['MAE']:.4f}")
    print(f"Enhanced RidgeMacro MAE: {enhanced_ridge['MAE']:.4f}")

    ridge_improvement = ((target_mae - enhanced_ridge["MAE"]) / target_mae) * 100
    print(f"RidgeMacro Improvement: {ridge_improvement:+.2f}%")

    print(f"Baseline Huber MAE: {baseline_huber['MAE']:.4f}")
    print(f"Enhanced Huber MAE: {enhanced_huber['MAE']:.4f}")

    huber_improvement = ((target_mae - enhanced_huber["MAE"]) / target_mae) * 100
    print(f"Huber Improvement: {huber_improvement:+.2f}%")

    # Check if OPR features actually used
    opr_count = sum(
        1
        for c in df.columns
        if c.startswith("opr_") and enhanced_ridge["n_features"] > 0
    )
    print(f"\nOPR Features Used: {opr_count} (out of {enhanced_ridge['n_features']})")

    print(
        f"\nGoal (MAE < 0.22): {'ACHIEVED' if enhanced_ridge['MAE'] < 0.22 else 'NOT ACHIEVED'}"
    )


if __name__ == "__main__":
    main()
