#!/usr/bin/env python3
"""
Backtest Runner for Sirena Models
=================================
Runs backtests for all registered models across multiple horizons.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.models.registry import ModelRegistry


def load_data():
    """Load inflation data."""
    data_path = Path(__file__).parent.parent / "data" / "enhanced_inflation_data.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    return df


def calculate_metrics(df):
    """Calculate evaluation metrics."""
    mae = np.mean(np.abs(df["error"]))
    rmse = np.sqrt(np.mean(df["error"] ** 2))
    mape = np.mean(np.abs(df["error"] / (df["actual"] + 100))) * 100
    bias = np.mean(df["error"])

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 2),
        "Bias": round(bias, 4),
        "N": len(df),
    }


def run_backtest(model, df, horizon, start_date="2019-01-01"):
    """Run backtest for a single model."""
    try:
        result = model.backtest(df=df, start_date=start_date, horizon=horizon)

        if result.empty:
            return None

        metrics = calculate_metrics(result)
        metrics["model"] = model.name
        metrics["horizon"] = horizon

        return metrics, result
    except Exception as e:
        print(f"Error backtesting {model.name} (h={horizon}): {e}")
        return None


def main():
    """Main backtest runner."""
    print("=" * 50)
    print("Sirena Backtest Runner")
    print("=" * 50)

    df = load_data()
    print(f"Loaded data: {len(df)} observations ({df.index.min()} to {df.index.max()})")

    models = ModelRegistry.list_models()
    print(f"Registered models: {models}")

    horizons = [1, 2, 12]
    all_metrics = []
    all_predictions = {}

    results_dir = Path(__file__).parent.parent / "archive" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for horizon in horizons:
        print(f"\n--- Running backtests for h={horizon} ---")
        horizon_metrics = []

        for model_name in models:
            try:
                model = ModelRegistry.get(model_name)
                result = run_backtest(model, df, horizon)

                if result:
                    metrics, predictions = result
                    horizon_metrics.append(metrics)
                    all_predictions[(model_name, horizon)] = predictions
                    print(f"  {model_name}: MAE={metrics['MAE']:.4f}")
            except Exception as e:
                print(f"  {model_name}: ERROR - {e}")

        if horizon_metrics:
            metrics_df = pd.DataFrame(horizon_metrics)
            all_metrics.extend(horizon_metrics)

            metrics_file = results_dir / f"backtest_h{horizon}_metrics.csv"
            metrics_df.to_csv(metrics_file, index=False)
            print(f"  Saved metrics to {metrics_file}")

    if all_metrics:
        all_metrics_df = pd.DataFrame(all_metrics)
        all_metrics_file = results_dir / "backtest_all_metrics.csv"
        all_metrics_df.to_csv(all_metrics_file, index=False)
        print(f"\nSaved all metrics to {all_metrics_file}")

    print("\nBacktest completed!")


if __name__ == "__main__":
    main()
