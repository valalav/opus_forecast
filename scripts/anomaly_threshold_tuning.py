"""
Anomaly Detection Threshold Tuning
====================================

Tests different sigma thresholds for volatility anomaly detection to find optimal balance
between precision (few false alarms) and recall (catching true anomalies).

Methodology:
1. Load weekly price data (2016-2026)
2. Create ground truth: Top 5% of absolute WoW growth labeled as anomalies
3. Test individual thresholds: 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 sigma
4. Calculate precision/recall/F1 for each threshold
5. Find optimal threshold

Output:
- data/anomaly_threshold_results.csv: Threshold, Precision, Recall, F1
- Optimal threshold identified
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import argparse

warnings.filterwarnings("ignore")

# Add parent directory to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.data.weekly_loader import load_weekly_prices, HIGH_QUALITY_PRODUCTS


def calculate_ground_truth(
    weekly_df: pd.DataFrame,
    anomaly_percentile: float = 5.0,
) -> pd.DataFrame:
    """
    Create ground truth by labeling extreme price movements as anomalies.

    Args:
        weekly_df: Weekly prices DataFrame
        anomaly_percentile: Top percentile to label as anomalies

    Returns:
        DataFrame with is_anomaly column
    """
    df = weekly_df.copy()

    # Calculate absolute WoW growth
    df["abs_wow"] = df["wow_growth"].abs()

    # Calculate percentile threshold per product
    percentiles = (
        df.groupby("product_code")["abs_wow"]
        .quantile(anomaly_percentile / 100.0)
        .reset_index()
    )
    percentiles.columns = ["product_code", "threshold"]

    df = df.merge(percentiles, on="product_code", how="left")

    # Label anomalies
    df["is_anomaly"] = (df["abs_wow"] >= df["threshold"]).astype(int)

    return df


def calculate_rolling_z_scores(
    df: pd.DataFrame,
    lookback_weeks: int = 52,
) -> pd.DataFrame:
    """
    Calculate rolling Z-scores for each product.

    Args:
        df: DataFrame with wow_growth
        lookback_weeks: Historical window for calculating mean/std

    Returns:
        DataFrame with z_score column
    """
    df = df.sort_values(["product_code", "date"])

    # Calculate rolling stats per product
    df["rolling_mean"] = df.groupby("product_code")["wow_growth"].transform(
        lambda x: x.shift(1).rolling(lookback_weeks, min_periods=12).mean()
    )
    df["rolling_std"] = df.groupby("product_code")["wow_growth"].transform(
        lambda x: x.shift(1).rolling(lookback_weeks, min_periods=12).std()
    )

    # Calculate Z-score
    df["z_score"] = (
        (df["wow_growth"] - df["rolling_mean"]) / df["rolling_std"]
    ).fillna(0)

    return df


def evaluate_threshold(
    df: pd.DataFrame,
    threshold: float,
) -> Dict[str, float]:
    """
    Evaluate a single threshold value.

    Args:
        df: DataFrame with ground truth (is_anomaly) and z_score
        threshold: Z-score threshold for anomaly detection

    Returns:
        Dict with precision, recall, f1 metrics
    """
    # Predict anomalies based on threshold
    df_temp = df.copy()
    df_temp["predicted"] = (df_temp["z_score"].abs() >= threshold).astype(int)

    # Calculate confusion matrix
    tp = ((df_temp["predicted"] == 1) & (df_temp["is_anomaly"] == 1)).sum()
    fp = ((df_temp["predicted"] == 1) & (df_temp["is_anomaly"] == 0)).sum()
    fn = ((df_temp["predicted"] == 0) & (df_temp["is_anomaly"] == 1)).sum()

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def find_optimal_threshold(results_df: pd.DataFrame) -> Dict[str, float]:
    """
    Find optimal threshold based on F1 score.

    Args:
        results_df: DataFrame with columns: threshold, precision, recall, f1

    Returns:
        Dict with optimal threshold and metrics
    """
    best_idx = results_df["f1"].idxmax()
    best_row = results_df.loc[best_idx]

    return {
        "threshold": best_row["threshold"],
        "precision": best_row["precision"],
        "recall": best_row["recall"],
        "f1": best_row["f1"],
    }


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Tune anomaly detection thresholds for weekly price volatility monitoring"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2016-01-01",
        help="Start date for training data (default: 2016-01-01)",
    )
    parser.add_argument(
        "--threshold-start",
        type=float,
        default=1.5,
        help="Start of threshold grid (default: 1.5)",
    )
    parser.add_argument(
        "--threshold-end",
        type=float,
        default=4.0,
        help="End of threshold grid (default: 4.0)",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.5,
        help="Step size for threshold grid (default: 0.5)",
    )
    parser.add_argument(
        "--anomaly-percentile",
        type=float,
        default=5.0,
        help="Percentile for ground truth anomaly labeling (default: 5.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory for results (default: data)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Anomaly Detection Threshold Tuning")
    print("=" * 70)

    # Load data
    print("\n[1/4] Loading weekly price data...")
    weekly_df = load_weekly_prices(start_date=args.start_date)
    print(f"Loaded {len(weekly_df)} weekly observations")
    print(f"Products: {weekly_df['product_code'].nunique()}")

    # Use high-quality products only
    hq_codes = list(HIGH_QUALITY_PRODUCTS.keys())
    weekly_df = weekly_df[weekly_df["product_code"].isin(hq_codes)]
    print(f"High-quality products: {len(hq_codes)}")

    # Create ground truth
    print("\n[2/4] Creating ground truth (top 5% anomalies)...")
    df = calculate_ground_truth(weekly_df, anomaly_percentile=args.anomaly_percentile)
    n_anomalies = df["is_anomaly"].sum()
    n_total = len(df)
    print(f"Labeled {n_anomalies} anomalies ({100 * n_anomalies / n_total:.2f}%)")

    # Calculate Z-scores
    print("\n[3/4] Calculating rolling Z-scores...")
    df = calculate_rolling_z_scores(df, lookback_weeks=52)
    print(f"Z-score range: [{df['z_score'].min():.2f}, {df['z_score'].max():.2f}]")
    print(f"Z-score std: {df['z_score'].std():.2f}")

    # Remove NaN rows (start of rolling window)
    df_valid = df.dropna(subset=["z_score", "rolling_mean", "rolling_std"])
    print(f"Valid observations: {len(df_valid)}")

    # Test thresholds
    print("\n[4/4] Testing individual thresholds...")
    thresholds = np.arange(
        args.threshold_start,
        args.threshold_end + args.threshold_step,
        args.threshold_step,
    ).tolist()
    print(f"Testing {len(thresholds)} threshold values")

    results = []
    for thresh in thresholds:
        metrics = evaluate_threshold(df_valid, thresh)
        results.append(metrics)

    results_df = pd.DataFrame(results)

    # Find optimal
    optimal = find_optimal_threshold(results_df)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "anomaly_threshold_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nAll Thresholds:")
    print(results_df.to_string(index=False))

    print("\n" + "-" * 70)
    print("OPTIMAL THRESHOLD")
    print("-" * 70)

    print(f"\n🎯 Best F1 Score:")
    print(f"  Threshold: {optimal['threshold']:.1f}σ")
    print(f"  Precision: {optimal['precision']:.2%}")
    print(f"  Recall: {optimal['recall']:.2%}")
    print(f"  F1 Score: {optimal['f1']:.4f}")

    # Compare with current default (2.0 sigma)
    current_idx = results_df[results_df["threshold"] == 2.0].index
    if len(current_idx) > 0:
        current = results_df.loc[current_idx[0]]
        print(f"\nCurrent default (2.0σ):")
        print(f"  Precision: {current['precision']:.2%}")
        print(f"  Recall: {current['recall']:.2%}")
        print(f"  F1 Score: {current['f1']:.4f}")

        if optimal["f1"] > current["f1"]:
            improvement = 100 * (optimal["f1"] - current["f1"]) / current["f1"]
            print(f"\n  ✅ Optimal threshold improves F1 by +{improvement:.2f}%")
        else:
            degradation = 100 * (current["f1"] - optimal["f1"]) / current["f1"]
            print(
                f"\n  ⚠️  Current default is optimal (beats best by +{degradation:.2f}%)"
            )

    # Recommendation for VolatilityMonitor
    print("\n" + "-" * 70)
    print("RECOMMENDATION FOR VolatilityMonitor")
    print("-" * 70)
    print(f"\nRecommended defaults:")
    print(f"  warning_threshold = {optimal['threshold']:.1f}")
    print(f"  critical_threshold = {optimal['threshold'] + 1.0:.1f}  # warning + 1σ")
    print(f"  (Based on optimal F1 score of {optimal['f1']:.4f})")

    print("\n" + "=" * 70)

    return results_df, optimal


if __name__ == "__main__":
    results, optimal = main()
