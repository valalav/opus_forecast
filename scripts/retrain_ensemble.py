#!/usr/bin/env python3
"""
Auto-Retraining Pipeline for SIRENA-KBR v5.0

Retrains all production models on new data and generates updated metrics.

Usage:
    python3 scripts/retrain_ensemble.py --dry-run      # Test without saving
    python3 scripts/retrain_ensemble.py                # Full retraining
    python3 scripts/retrain_ensemble.py --data my.csv  # Custom data file

Author: Claude Code
Date: 2026-01-24
"""

import sys
import os
import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

# 9 Production Models (v5.0 Ensemble)
PRODUCTION_MODELS = {
    "SubcomponentForecaster": {
        "import": "from sirena.models.subcomponent import SubcomponentForecaster",
        "weight": 0.18,
        "mae_target": 0.309,
    },
    "RidgeMacroForecaster": {
        "import": "from sirena.models.ridge_macro import RidgeMacroForecaster",
        "weight": 0.17,
        "mae_target": 0.319,
    },
    "RidgeShockDummiesForecaster": {
        "import": "from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster",
        "weight": 0.17,
        "mae_target": 0.319,
    },
    "RidgeForecaster": {
        "import": "from sirena.models.ridge import RidgeForecaster",
        "weight": 0.15,
        "mae_target": 0.321,
    },
    "HuberForecaster": {
        "import": "from sirena.models.huber import HuberForecaster",
        "weight": 0.12,
        "mae_target": 0.324,
    },
    "NGBoostForecaster": {
        "import": "from sirena.models.ngboost_model import NGBoostForecaster",
        "weight": 0.10,
        "mae_target": 0.290,
    },
    "ProphetForecaster": {
        "import": "from sirena.models.prophet import ProphetForecaster",
        "weight": 0.06,
        "mae_target": 0.310,
    },
    "EBMForecaster": {
        "import": "from sirena.models.ebm import EBMForecaster",
        "weight": 0.05,
        "mae_target": 0.336,
    },
    "ElasticNetForecaster": {
        "import": "from sirena.models.elasticnet import ElasticNetForecaster",
        "weight": 0.0,  # Not in v5.0 ensemble based on CLAUDE.md
        "mae_target": 0.320,
    },
}

DEFAULT_DATA_FILE = str(project_root / "data" / "infl_kbr.csv")
OUTPUT_DIR = project_root / "archive" / "results"
MODEL_ARTIFACTS_DIR = project_root / "archive" / "weights"
RETRAIN_LOG = MODEL_ARTIFACTS_DIR / "retrain_log.json"

# =============================================================================
# Data Loader
# =============================================================================


def load_data(data_file: str = DEFAULT_DATA_FILE) -> pd.DataFrame:
    """
    Load inflation data from CSV file.

    Args:
        data_file: Path to data file

    Returns:
        DataFrame with inflation data
    """
    print(f"\n{'=' * 70}")
    print(f"Loading data from: {data_file}")
    print(f"{'=' * 70}\n")

    try:
        # Try infl_kbr.csv format (Rosstat format)
        df = pd.read_csv(data_file, sep=";", decimal=",", encoding="utf-8")

        # Convert MoM to numeric (handle string concatenation issues)
        df["MoM"] = pd.to_numeric(df["MoM"], errors="coerce")

        # Handle both ISO format (2010-01-01) and Russian format (01.01.2010)
        try:
            df["Day"] = pd.to_datetime(df["Day"], format="%Y-%m-%d")
        except:
            df["Day"] = pd.to_datetime(df["Day"], format="%d.%m.%Y")
        df = df.set_index("Day")

        # Pivot to have categories as columns
        df = df.pivot(columns="Товар", values="MoM")

        print(f"✓ Loaded {len(df)} rows from {df.shape[1]} categories")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Categories: {list(df.columns)[:5]}...")

        return df

    except Exception as e:
        print(f"✗ Failed to load infl_kbr.csv format: {e}")
        raise


# =============================================================================
# Model Retrainer
# =============================================================================


class ModelRetrainer:
    """
    Handles retraining of all production models.
    """

    def __init__(self, dry_run: bool = False, data_file: str = DEFAULT_DATA_FILE):
        self.dry_run = dry_run
        self.data_file = data_file
        self.df = None
        self.results = {}
        self.start_time = datetime.now()

        # Create output directories
        MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        """Load training data."""
        self.df = load_data(self.data_file)

    def retrain_model(self, model_name: str, model_config: Dict) -> Dict:
        """
        Retrain a single model.

        Args:
            model_name: Name of the model class
            model_config: Model configuration dict

        Returns:
            Dict with retraining results
        """
        print(f"\n{'-' * 70}")
        print(f"Retraining: {model_name}")
        print(f"{'-' * 70}")

        result = {
            "model": model_name,
            "status": "pending",
            "mae": None,
            "mae_change": None,
            "error": None,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Dynamic import
            exec(model_config["import"], globals())

            # Get model class
            model_class = eval(model_name)

            # Fit model
            print(f"  Fitting model...")
            model = model_class()
            model.fit(self.df, target_col="Все товары и услуги")

            # Generate forecast for validation
            print(f"  Generating forecast...")
            forecast = model.forecast(horizon=12)

            result["forecast_sample"] = (
                forecast[:3].tolist() if len(forecast) > 3 else forecast.tolist()
            )
            result["status"] = "success"

            # Calculate MAE (simplified - use last 12 months for validation)
            if len(self.df) >= 24:
                train_size = len(self.df) - 12
                actual = self.df["Все товары и услуги"].iloc[-12:].values

                # Simple validation: forecast next 12 months vs actual
                # Note: This is a simplified check, real backtest should use BacktestRunner
                if len(forecast) >= len(actual):
                    errors = np.abs(forecast[: len(actual)] - actual)
                    result["mae"] = float(np.mean(errors))
                    result["mae_change"] = result["mae"] - model_config["mae_target"]

                    print(
                        f"  ✓ MAE: {result['mae']:.3f} (target: {model_config['mae_target']:.3f})"
                    )
                    print(f"  ✓ Change: {result['mae_change']:+.3f}")

            # Save model artifacts if not dry run
            if not self.dry_run:
                artifact_file = MODEL_ARTIFACTS_DIR / f"{model_name}_artifact.pkl"
                import joblib

                joblib.dump(model, artifact_file)
                print(f"  ✓ Saved: {artifact_file}")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"  ✗ Error: {e}")

        return result

    def retrain_all(self) -> Dict:
        """
        Retrain all production models.

        Returns:
            Dict with all retraining results
        """
        print(f"\n{'=' * 70}")
        print(f"RETRAINING PIPELINE (dry_run={self.dry_run})")
        print(f"{'=' * 70}")

        self.load_data()

        # Retrain each model
        for model_name, model_config in PRODUCTION_MODELS.items():
            if model_config["weight"] > 0 or model_name == "ElasticNetForecaster":
                # Only train models with weight > 0 or ElasticNet (for testing)
                result = self.retrain_model(model_name, model_config)
                self.results[model_name] = result

        return self.results

    def generate_backtest_metrics(self):
        """
        Generate new backtest_h1_metrics.csv by running backtest framework.

        This is more accurate than the simple validation above.
        """
        print(f"\n{'=' * 70}")
        print(f"GENERATING BACKTEST METRICS (h=1)")
        print(f"{'=' * 70}\n")

        if self.dry_run:
            print("  [DRY RUN] Skipping backtest generation")
            return

        try:
            from backtest_framework import BacktestRunner

            # Create runner
            runner = BacktestRunner(
                horizon=1, test_months=12, output_dir=str(OUTPUT_DIR)
            )

            # Run backtest
            results = runner.run()

            # Calculate metrics
            metrics = runner.calculate_metrics(results)

            # Save results
            runner.save_results(results, metrics)

            print(f"\n  ✓ Backtest h=1 complete")
            print(f"  ✓ Results saved to:")
            print(f"    - {OUTPUT_DIR / 'backtest_h1_predictions.csv'}")
            print(f"    - {OUTPUT_DIR / 'backtest_h1_metrics.csv'}")

        except Exception as e:
            print(f"  ✗ Backtest failed: {e}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Auto-Retraining Pipeline for SIRENA-KBR v5.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run              # Test without saving artifacts
  %(prog)s                       # Full retraining with backtest
  %(prog)s --data my_data.csv     # Use custom data file
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (no artifacts saved, no backtest)",
    )

    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_DATA_FILE,
        help=f"Path to data file (default: {DEFAULT_DATA_FILE})",
    )

    args = parser.parse_args()

    # Run retraining
    retrainer = ModelRetrainer(dry_run=args.dry_run, data_file=args.data)
    results = retrainer.retrain_all()

    # Generate backtest metrics if not dry run
    if not args.dry_run:
        retrainer.generate_backtest_metrics()

        # Save retrain log
        log = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": args.dry_run,
            "data_file": args.data,
            "models_trained": len(results),
            "results": results,
        }

        with open(RETRAIN_LOG, "w") as f:
            json.dump(log, f, indent=2, default=str)

        print(f"\n{'=' * 70}")
        print(f"RETRAIN LOG SAVED: {RETRAIN_LOG}")
        print(f"{'=' * 70}")

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"RETRAINING SUMMARY")
    print(f"{'=' * 70}")

    success_count = sum(1 for r in results.values() if r["status"] == "success")
    error_count = sum(1 for r in results.values() if r["status"] == "error")

    print(f"\nModels trained: {success_count}/{len(results)}")
    print(f"Errors: {error_count}")

    if success_count > 0:
        print(f"\n✓ Success:")
        for name, result in results.items():
            if result["status"] == "success":
                mae = result.get("mae", "N/A")
                change = result.get("mae_change", "N/A")
                print(f"  - {name}: MAE={mae}, Δ={change}")

    if error_count > 0:
        print(f"\n✗ Errors:")
        for name, result in results.items():
            if result["status"] == "error":
                print(f"  - {name}: {result['error']}")

    duration = (datetime.now() - retrainer.start_time).total_seconds()
    print(f"\nTotal time: {duration:.1f}s")

    if args.dry_run:
        print(f"\n[DRY RUN] No artifacts or metrics saved")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
