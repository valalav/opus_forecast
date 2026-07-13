#!/usr/bin/env python3
"""
Metrics Aggregation Script
==========================
Aggregates backtest results from all horizons (h=1, h=2, h=12).
Calculates weighted score: 50% h1 + 30% h2 + 20% h12.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_metrics():
    """Load all horizon metrics files."""
    results_dir = Path(__file__).parent.parent / "archive" / "results"

    horizons = [1, 2, 12]
    all_metrics = []

    for horizon in horizons:
        metrics_file = results_dir / f"backtest_h{horizon}_metrics.csv"

        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            all_metrics.append(df)
            print(f"Loaded {metrics_file}: {len(df)} models")
        else:
            print(f"Warning: {metrics_file} not found")

    if not all_metrics:
        raise FileNotFoundError("No metrics files found")

    return pd.concat(all_metrics, ignore_index=True)


def calculate_weighted_score(df):
    """Calculate weighted score from all horizons."""

    pivot_df = df.pivot(index="model", columns="horizon", values="MAE")

    weights = {1: 0.5, 2: 0.3, 12: 0.2}

    weighted_scores = []
    for model in pivot_df.index:
        score = 0
        for h, weight in weights.items():
            if h in pivot_df.columns:
                if pd.notna(pivot_df.loc[model, h]):
                    score += pivot_df.loc[model, h] * weight

        weighted_scores.append(score)

    pivot_df["weighted_score"] = weighted_scores
    pivot_df = pivot_df.reset_index()

    return pivot_df


def format_consolidated_metrics(pivot_df):
    """Format consolidated metrics for output."""

    result = pivot_df[["model", "weighted_score"]].copy()

    if 1 in pivot_df.columns:
        result["MAE_h1"] = pivot_df[1]
    if 2 in pivot_df.columns:
        result["MAE_h2"] = pivot_df[2]
    if 12 in pivot_df.columns:
        result["MAE_h12"] = pivot_df[12]

    result = result.sort_values("weighted_score")
    result = result.reset_index(drop=True)

    result = result.rename(
        columns={
            "model": "Model",
            "weighted_score": "Weighted_Score",
            "MAE_h1": "MAE_h1",
            "MAE_h2": "MAE_h2",
            "MAE_h12": "MAE_h12",
        }
    )

    result["Weighted_Score"] = result["Weighted_Score"].round(4)

    for col in ["MAE_h1", "MAE_h2", "MAE_h12"]:
        if col in result.columns:
            result[col] = result[col].round(4)

    return result


def main():
    """Main aggregation function."""
    print("=" * 50)
    print("Metrics Aggregation")
    print("=" * 50)

    df = load_metrics()
    print(f"\nTotal records: {len(df)}")

    pivot_df = calculate_weighted_score(df)

    result = format_consolidated_metrics(pivot_df)

    output_path = Path(__file__).parent.parent / "data" / "consolidated_metrics.csv"
    result.to_csv(output_path, index=False)

    print(f"\nSaved consolidated metrics to {output_path}")
    print("\nConsolidated Metrics:")
    print(result.to_string(index=False))

    print(f"\n{'=' * 50}")
    print(f"Models included: {len(result)}")
    print(f"Weighted Score formula: 50%*h1 + 30%*h2 + 20%*h12")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
