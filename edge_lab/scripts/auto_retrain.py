#!/usr/bin/env python3
"""
Monthly Auto-Retrain Pipeline for Sirena Production Models
==========================================================

Cron-ready script that:
1. Loads latest data from data/infl_kbr.csv
2. Retrains all production models
3. Saves updated weights to archive/weights/
4. Logs performance delta vs previous run

Usage:
    python3 scripts/auto_retrain.py --dry-run    # Validate setup without retraining
    python3 scripts/auto_retrain.py              # Run full retrain pipeline
    python3 scripts/auto_retrain.py --models ridge,huber  # Retrain specific models

Cron entry (monthly on 1st at 2am):
    0 2 1 * * cd /home/valalav/_projects/sirena-kbr/edge_lab && python3 scripts/auto_retrain.py >> logs/retrain.log 2>&1
"""

import pandas as pd
import numpy as np
import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import warnings
import traceback

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.models.registry import ModelRegistry
from sirena.models.base import BaseForecaster

try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


class RetrainPipeline:
    """
    Automated model retraining pipeline.

    Retrains all production models with latest data and maintains
    historical performance logs for monitoring degradation.
    """

    def __init__(
        self,
        data_path: Optional[Path] = None,
        weights_dir: Optional[Path] = None,
        log_path: Optional[Path] = None,
        target_col: str = "mom",
    ):
        """
        Initialize the retrain pipeline.

        Args:
            data_path: Path to inflation data file
            weights_dir: Directory to save model weights
            log_path: Path to retrain log file
            target_col: Target column name for training
        """
        base_dir = Path(__file__).parent.parent

        self.data_path = data_path or base_dir / "data" / "infl_kbr.csv"
        self.weights_dir = weights_dir or base_dir / "archive" / "weights"
        self.log_path = log_path or self.weights_dir / "retrain_log.json"
        self.target_col = target_col

        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.df = None
        self.previous_log = None
        self.results = {}

    def _find_data_file(self) -> Path:
        """
        Find the inflation data file.

        Looks for infl_kbr.csv first, falls back to enhanced_inflation_data.csv.
        Creates a symlink if needed for backward compatibility.
        """
        candidates = [
            Path(__file__).parent.parent / "data" / "infl_kbr.csv",
            Path(__file__).parent.parent / "data" / "enhanced_inflation_data.csv",
        ]

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(f"No inflation data found. Tried: {candidates}")

    def _ensure_data_link(self) -> Path:
        """
        Ensure infl_kbr.csv exists (creates symlink if needed).
        """
        target = Path(__file__).parent.parent / "data" / "infl_kbr.csv"
        source = Path(__file__).parent.parent / "data" / "enhanced_inflation_data.csv"

        if not target.exists() and source.exists():
            target.symlink_to(source)
            print(f"Created symlink: {target} -> {source}")

        return self._find_data_file()

    def load_data(self) -> pd.DataFrame:
        """
        Load inflation data.

        Returns:
            DataFrame with datetime index and target column
        """
        data_path = self._ensure_data_link()

        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()

        print(f"Loaded {len(df)} observations ({df.index.min()} to {df.index.max()})")

        self.df = df
        return df

    def load_previous_log(self) -> Optional[Dict]:
        """
        Load previous retrain log for performance comparison.

        Returns:
            Previous log dict or None if no log exists
        """
        if not self.log_path.exists():
            return None

        with open(self.log_path, "r") as f:
            self.previous_log = json.load(f)

        print(
            f"Loaded previous log from: {self.previous_log.get('timestamp', 'unknown')}"
        )
        return self.previous_log

    def calculate_model_mae(self, model: BaseForecaster, df: pd.DataFrame) -> float:
        """
        Calculate MAE for a model on the last 12 months of data.

        Args:
            model: Trained model
            df: Full dataset

        Returns:
            MAE value
        """
        test_size = min(12, len(df) - 24)
        if test_size < 1:
            return 0.0

        train_df = df.iloc[:-test_size]
        test_df = df.iloc[-test_size:]

        try:
            model.fit(train_df, target_col=self.target_col)
            predictions = model.forecast(horizon=test_size)

            actual = test_df[self.target_col].values - 100
            errors = np.abs(actual - predictions)
            mae = np.mean(errors)

            return float(mae)
        except Exception as e:
            print(f"  Error calculating MAE for {model.name}: {e}")
            return 0.0

    def save_model_weights(
        self, model: BaseForecaster, model_name: str, timestamp: str
    ) -> Path:
        """
        Save model weights to disk.

        Args:
            model: Trained model instance
            model_name: Name of the model
            timestamp: Training timestamp

        Returns:
            Path to saved weights file
        """
        if not JOBLIB_AVAILABLE:
            print(f"  Warning: joblib not available, skipping save for {model_name}")
            return None

        filename = f"{model_name}_{timestamp}.pkl"
        weights_path = self.weights_dir / filename

        try:
            joblib.dump(model, weights_path)
            print(f"  Saved weights to: {weights_path}")

            latest_link = self.weights_dir / f"{model_name}_latest.pkl"
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(filename)

        except Exception as e:
            print(f"  Warning: Failed to save weights for {model_name}: {e}")
            return None

        return weights_path

    def retrain_model(self, model_name: str, dry_run: bool = False) -> Dict[str, Any]:
        """
        Retrain a single model.

        Args:
            model_name: Name of model to retrain
            dry_run: If True, skip actual training

        Returns:
            Result dict with metrics and status
        """
        result = {
            "model": model_name,
            "status": "pending",
            "mae": None,
            "mae_delta": None,
            "weights_path": None,
            "error": None,
        }

        try:
            model = ModelRegistry.get(model_name)
            print(f"\nRetraining: {model_name}")

            if dry_run:
                result["status"] = "dry_run"
                print(f"  Dry run: would train {model_name}")
                return result

            model.fit(self.df, target_col=self.target_col)
            print(f"  Training completed")

            mae = self.calculate_model_mae(model, self.df)
            result["mae"] = round(mae, 4)
            print(f"  MAE: {mae:.4f}")

            if self.previous_log and "models" in self.previous_log:
                prev_mae = self.previous_log["models"].get(model_name, {}).get("mae")
                if prev_mae is not None:
                    delta = round(mae - prev_mae, 4)
                    result["mae_delta"] = delta
                    print(f"  MAE delta vs previous: {delta:+.4f}")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            weights_path = self.save_model_weights(model, model_name, timestamp)
            if weights_path:
                result["weights_path"] = str(weights_path)

            result["status"] = "success"

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"  Error: {e}")
            traceback.print_exc()

        return result

    def run(
        self,
        models: Optional[List[str]] = None,
        dry_run: bool = False,
        min_train_size: int = 24,
    ) -> Dict[str, Any]:
        """
        Run the full retrain pipeline.

        Args:
            models: List of model names to retrain (None = all)
            dry_run: If True, validate setup without training
            min_train_size: Minimum observations required

        Returns:
            Pipeline run results
        """
        print("=" * 60)
        print("Sirena Monthly Auto-Retrain Pipeline")
        print("=" * 60)

        timestamp = datetime.now().isoformat()

        results = {
            "timestamp": timestamp,
            "dry_run": dry_run,
            "status": "running",
            "models": {},
            "summary": {},
        }

        try:
            self.load_data()

            if len(self.df) < min_train_size:
                raise ValueError(
                    f"Insufficient data: {len(self.df)} < {min_train_size}"
                )

            self.load_previous_log()

            registered_models = ModelRegistry.list_models()

            if models:
                models_to_train = [m for m in models if m in registered_models]
                if not models_to_train:
                    raise ValueError(
                        f"No valid models found. Requested: {models}, "
                        f"Available: {registered_models}"
                    )
            else:
                models_to_train = registered_models

            print(f"\nModels to train: {models_to_train}")

            success_count = 0
            error_count = 0
            total_mae = 0.0
            mae_count = 0

            for model_name in models_to_train:
                model_result = self.retrain_model(model_name, dry_run)
                results["models"][model_name] = model_result

                if model_result["status"] == "success":
                    success_count += 1
                    if model_result["mae"] is not None:
                        total_mae += model_result["mae"]
                        mae_count += 1
                elif model_result["status"] == "error":
                    error_count += 1

            results["summary"] = {
                "total_models": len(models_to_train),
                "successful": success_count,
                "errors": error_count,
                "avg_mae": round(total_mae / mae_count, 4) if mae_count > 0 else None,
            }

            results["status"] = "completed" if error_count == 0 else "partial"

            if not dry_run:
                with open(self.log_path, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"\nLog saved to: {self.log_path}")

        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)
            print(f"\nPipeline failed: {e}")
            traceback.print_exc()
        else:
            if results["status"] == "running":
                results["status"] = "completed" if error_count == 0 else "partial"

        print("\n" + "=" * 60)
        print("Pipeline Summary:")
        print(f"  Status: {results['status']}")
        print(f"  Models: {results['summary'].get('total_models', 'N/A')}")
        print(f"  Successful: {results['summary'].get('successful', 'N/A')}")
        print(f"  Errors: {results['summary'].get('errors', 'N/A')}")
        if results["summary"].get("avg_mae") is not None:
            print(f"  Avg MAE: {results['summary']['avg_mae']:.4f}")
        print("=" * 60)

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Sirena Monthly Auto-Retrain Pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate setup without retraining",
    )
    parser.add_argument(
        "--models",
        type=str,
        help="Comma-separated list of models to retrain (default: all)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Path to inflation data file",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        help="Directory to save model weights",
    )

    args = parser.parse_args()

    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",")]

    pipeline = RetrainPipeline(
        data_path=args.data_path,
        weights_dir=args.weights_dir,
    )

    results = pipeline.run(
        models=models,
        dry_run=args.dry_run,
    )

    exit_code = 0 if results["status"] in ["completed", "dry_run", "pending"] else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
