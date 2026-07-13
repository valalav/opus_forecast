"""
Optimize MIDAS model parameters to improve MAE performance.

This script tests various configurations to find better hyperparameters.
"""

import pandas as pd
import numpy as np
from itertools import product
import sys
from pathlib import Path

# Add parent directory to path (from edge_lab to opus_forecast)
sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.models import MIDASForecaster


def load_data():
    """Load inflation data and merge with brent prices."""
    # Load inflation data
    df = pd.read_csv(
        "/home/valalav/_projects/sirena-kbr/data/inflation_data.csv",
        sep=";",
        decimal=",",
        index_col=0,
        dayfirst=True,
        parse_dates=True,
    )

    # Normalize dates to month-start for joining
    df.index = df.index.to_period("M").to_timestamp()

    # Load brent prices (already month-start)
    brent = pd.read_csv(
        "/home/valalav/_projects/sirena-kbr/data/brent_prices.csv",
        index_col=0,
        parse_dates=True,
    )

    # Merge
    df = df.join(brent[["brent"]], how="left")

    # Select columns (use Ki_i directly)
    df = df[["mom", "brent", "usd_nom_i"]].copy()
    df = df.rename(columns={"mom": "Все товары и услуги"})

    # Drop NA only for target and usd
    df = df.dropna(subset=["Все товары и услуги", "usd_nom_i"])
    return df


def evaluate_config(config, df):
    """Evaluate a single configuration."""
    try:
        model = MIDASForecaster(**config)
        # Use shorter backtest period for faster testing
        results = model.backtest(df, start_date="2024-06-01")

        if len(results) == 0:
            return None

        mae = (results["error"].abs()).mean()
        return {"config": config, "mae": mae, "n_predictions": len(results)}
    except Exception as e:
        print(f"  Error: {e}")
        return None

        mae = (results["error"].abs()).mean()
        return {"config": config, "mae": mae, "n_predictions": len(results)}
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    print("=" * 80)
    print("MIDAS Model Hyperparameter Optimization")
    print("=" * 80)

    df = load_data()
    print(f"\nData: {len(df)} months ({df.index[0].date()} to {df.index[-1].date()})")

    # Baseline
    print("\n" + "=" * 80)
    print("BASELINE: Current MIDAS configuration")
    print("=" * 80)
    baseline_config = {
        "weight_type": "almon",
        "poly_order": 2,
        "alpha": 0.1,
        "hf_features": ["brent", "usd", "ki"],
    }
    baseline_result = evaluate_config(baseline_config, df)
    if baseline_result:
        print(
            f"MAE: {baseline_result['mae']:.4f} (n={baseline_result['n_predictions']})"
        )
        baseline_mae = baseline_result["mae"]
    else:
        print("No baseline results")
        return

    print("\n" + "=" * 80)
    print("OPTIMIZATION GRID")
    print("=" * 80)

    # Define parameter grid (simplified for faster testing)
    weight_types = ["almon", "exp"]
    poly_orders = [1, 2]  # Only relevant for almon
    alphas = [0.01, 0.1, 1.0]
    hf_feature_sets = [
        ["brent"],  # Only Brent
        ["brent", "usd"],  # Brent + USD
        ["usd"],  # Only USD
    ]

    results = []
    total_configs = 0

    for weight_type in weight_types:
        # For almon, try different poly_orders; for others, poly_order doesn't matter
        if weight_type == "almon":
            current_poly_orders = poly_orders
        else:
            current_poly_orders = [2]  # Dummy value, not used for non-almon

        for poly_order in current_poly_orders:
            for alpha in alphas:
                for hf_features in hf_feature_sets:
                    total_configs += 1

    print(f"Testing {total_configs} configurations...\n")

    config_count = 0
    best_result = baseline_result
    best_config = baseline_config

    for weight_type in weight_types:
        if weight_type == "almon":
            current_poly_orders = poly_orders
        else:
            current_poly_orders = [2]

        for poly_order in current_poly_orders:
            for alpha in alphas:
                for hf_features in hf_feature_sets:
                    config_count += 1
                    print(
                        f"[{config_count}/{total_configs}] weight_type={weight_type}, poly_order={poly_order}, alpha={alpha}, hf={hf_features}"
                    )

                    config = {
                        "weight_type": weight_type,
                        "poly_order": poly_order,
                        "alpha": alpha,
                        "hf_features": hf_features,
                    }

                    result = evaluate_config(config, df)

                    if result:
                        results.append(result)

                        # Update best
                        if result["mae"] < best_result["mae"]:
                            best_result = result
                            best_config = config
                            print(f"  → NEW BEST! MAE: {result['mae']:.4f}")
                        else:
                            print(f"  MAE: {result['mae']:.4f}")

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nBaseline MAE:  {baseline_mae:.4f}")
    print(f"Best MAE:      {best_result['mae']:.4f}")
    print(
        f"Improvement:   {(baseline_mae - best_result['mae']) / baseline_mae * 100:+.2f}%"
    )

    print(f"\nBest config:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")

    # Top 5 configs
    print("\nTop 5 configurations:")
    results_sorted = sorted(results, key=lambda x: x["mae"])[:5]
    for i, r in enumerate(results_sorted, 1):
        cfg = r["config"]
        print(f"\n{i}. MAE: {r['mae']:.4f}")
        print(f"   {cfg}")

    # Comparison with Ridge
    print("\n" + "=" * 80)
    print("COMPARISON WITH RIDGE BASELINE")
    print("=" * 80)
    ridge_mae = 0.321  # Known from documentation
    print(f"Ridge MAE:    {ridge_mae:.4f}")
    print(f"MIDAS Best:   {best_result['mae']:.4f}")
    print(
        f"Difference:   {best_result['mae'] - ridge_mae:+.4f} ({(best_result['mae'] - ridge_mae) / ridge_mae * 100:+.1f}%)"
    )

    if best_result["mae"] < ridge_mae:
        print("\n✅ SUCCESS: MIDAS beats Ridge!")
    else:
        print("\n❌ FAILURE: MIDAS still worse than Ridge")
        print("\nPossible reasons:")
        print("1. No actual high-frequency data (monthly data treated as HF)")
        print("2. MIDAS approach not suitable for this dataset")
        print("3. Need more sophisticated feature engineering")

    # Save results
    results_df = pd.DataFrame(
        [
            {**r["config"], "mae": r["mae"], "n_predictions": r["n_predictions"]}
            for r in results
        ]
    )
    results_df = results_df.sort_values("mae")
    results_df.to_csv(
        "/home/valalav/_projects/sirena-kbr/archive/results/midas_optimization_results.csv",
        index=False,
    )
    print(
        f"\nResults saved to: /home/valalav/_projects/sirena-kbr/archive/results/midas_optimization_results.csv"
    )


if __name__ == "__main__":
    main()
