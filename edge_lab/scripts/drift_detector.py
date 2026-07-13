#!/usr/bin/env python3
"""
Forecast Drift Detection Script
================================
Early warning system for model degradation.

Features:
- Tracks rolling MAE over last 6 months (simulated via backtest window)
- Alerts if current MAE > 1.5x historical average
- Generates drift_report.json with affected models

Usage:
    python3 scripts/drift_detector.py
    python3 scripts/drift_detector.py --threshold 1.5
    python3 scripts/drift_detector.py --historical-months 6
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

warnings.filterwarnings("ignore")


class DriftDetector:
    """
    Detects model performance drift by comparing current MAE
    against historical baseline.
    """

    def __init__(
        self,
        results_dir: Optional[Path] = None,
        weights_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        threshold: float = 1.5,
        historical_months: int = 6,
    ):
        """
        Initialize drift detector.

        Args:
            results_dir: Directory containing backtest metrics
            weights_dir: Directory containing retrain logs
            output_dir: Directory for output reports
            threshold: Multiplier for drift alert (current > threshold * baseline)
            historical_months: Number of months for baseline calculation
        """
        base_dir = Path(__file__).parent.parent

        self.results_dir = results_dir or base_dir / "archive" / "results"
        self.weights_dir = weights_dir or base_dir / "archive" / "weights"
        self.output_dir = output_dir or base_dir / "data"

        self.threshold = threshold
        self.historical_months = historical_months

        self.historical_mae: Dict[str, List[float]] = {}
        self.current_mae: Dict[str, float] = {}
        self.drifted_models: List[str] = []

    def load_backtest_metrics(self) -> pd.DataFrame:
        """
        Load backtest metrics from all horizons.

        Returns:
            DataFrame with model, horizon, MAE columns
        """
        horizons = [1, 2, 12]
        all_metrics = []

        for horizon in horizons:
            metrics_file = self.results_dir / f"backtest_h{horizon}_metrics.csv"

            if metrics_file.exists():
                df = pd.read_csv(metrics_file)
                if not df.empty:
                    all_metrics.append(df)

        if not all_metrics:
            raise FileNotFoundError(f"No backtest metrics found in {self.results_dir}")

        return pd.concat(all_metrics, ignore_index=True)

    def load_retrain_log(self) -> Optional[Dict]:
        """
        Load latest retrain log.

        Returns:
            Log dict or None if no log exists
        """
        log_path = self.weights_dir / "retrain_log.json"

        if not log_path.exists():
            return None

        with open(log_path, "r") as f:
            log = json.load(f)

        return log

    def calculate_historical_baseline(
        self, metrics_df: pd.DataFrame
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate historical MAE baseline for each model.

        Args:
            metrics_df: DataFrame with backtest metrics

        Returns:
            Dict mapping model to {historical_avg, historical_std}
        """
        baseline = {}

        for model_name in metrics_df["model"].unique():
            model_metrics = metrics_df[metrics_df["model"] == model_name]

            # Calculate average MAE across horizons (weighted towards h=1)
            weights = {1: 0.5, 2: 0.3, 12: 0.2}
            weighted_mae = 0.0
            total_weight = 0.0

            for _, row in model_metrics.iterrows():
                horizon = row["horizon"]
                mae = row["MAE"]
                weight = weights.get(horizon, 0.1)
                weighted_mae += mae * weight
                total_weight += weight

            historical_avg = (
                weighted_mae / total_weight
                if total_weight > 0
                else model_metrics["MAE"].mean()
            )

            # Store all MAE values for each model
            self.historical_mae[model_name] = model_metrics["MAE"].tolist()

            historical_std = model_metrics["MAE"].std()

            baseline[model_name] = {
                "historical_avg": float(historical_avg),
                "historical_std": float(historical_std),
                "n_observations": int(len(model_metrics)),
            }

        return baseline

    def get_current_performance(self) -> Dict[str, float]:
        """
        Get current model performance from retrain log.

        Returns:
            Dict mapping model name to current MAE
        """
        current = {}
        log = self.load_retrain_log()

        if log and "models" in log:
            for model_name, model_data in log["models"].items():
                if model_data.get("status") == "success":
                    mae = model_data.get("mae")
                    if mae is not None:
                        current[model_name] = float(mae)

        return current

    def detect_drift(self) -> Dict[str, Any]:
        """
        Detect model drift by comparing current vs historical MAE.

        Returns:
            Dict with drift detection results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "threshold": self.threshold,
            "historical_months": self.historical_months,
            "drift_detected": False,
            "affected_models": [],
            "model_analysis": {},
        }

        # Load historical baseline
        metrics_df = self.load_backtest_metrics()
        baseline = self.calculate_historical_baseline(metrics_df)

        # Get current performance
        self.current_mae = self.get_current_performance()

        # Analyze each model
        for model_name in baseline.keys():
            model_result = {
                "model": model_name,
                "historical_mae_avg": baseline[model_name]["historical_avg"],
                "historical_mae_std": baseline[model_name]["historical_std"],
                "historical_observations": baseline[model_name]["n_observations"],
                "current_mae": self.current_mae.get(model_name, None),
                "drift_status": "unknown",
                "degradation_pct": None,
                "alert_triggered": False,
            }

            if model_name in self.current_mae:
                current_mae = self.current_mae[model_name]
                historical_avg = baseline[model_name]["historical_avg"]

                # Calculate drift
                drift_threshold = self.threshold * historical_avg

                if current_mae > drift_threshold:
                    model_result["drift_status"] = "drift_detected"
                    model_result["alert_triggered"] = True
                    self.drifted_models.append(model_name)
                    results["drift_detected"] = True
                    results["affected_models"].append(model_name)
                else:
                    model_result["drift_status"] = "no_drift"
                    model_result["alert_triggered"] = False

                # Calculate degradation percentage
                degradation_pct = (
                    (current_mae - historical_avg) / historical_avg
                ) * 100
                model_result["degradation_pct"] = round(degradation_pct, 2)
            else:
                model_result["drift_status"] = "no_current_data"
                model_result["alert_triggered"] = False

            results["model_analysis"][model_name] = model_result

        # Summary
        results["summary"] = {
            "total_models_analyzed": len(baseline),
            "models_with_current_data": len(self.current_mae),
            "models_with_drift": len(self.drifted_models),
            "drift_threshold_multiplier": self.threshold,
        }

        return results

    def save_report(self, results: Dict[str, Any]) -> Path:
        """
        Save drift detection report to JSON.

        Args:
            results: Drift detection results

        Returns:
            Path to saved report
        """
        output_path = self.output_dir / "drift_report.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        return output_path

    def print_summary(self, results: Dict[str, Any]):
        """
        Print drift detection summary.

        Args:
            results: Drift detection results
        """
        print("=" * 60)
        print("Forecast Drift Detection Report")
        print("=" * 60)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Threshold: {results['threshold']}x historical average")
        print(
            f"Status: {'DRIFT DETECTED' if results['drift_detected'] else 'NO DRIFT DETECTED'}"
        )
        print()

        if results["drift_detected"]:
            print("⚠️  AFFECTED MODELS:")
            for model in results["affected_models"]:
                model_data = results["model_analysis"][model]
                print(f"  - {model}")
                print(f"    Historical MAE: {model_data['historical_mae_avg']:.4f}")
                print(f"    Current MAE: {model_data['current_mae']:.4f}")
                print(f"    Degradation: {model_data['degradation_pct']:+.2f}%")
            print()

        print("SUMMARY:")
        print(f"  Total models analyzed: {results['summary']['total_models_analyzed']}")
        print(
            f"  Models with current data: {results['summary']['models_with_current_data']}"
        )
        print(f"  Models with drift: {results['summary']['models_with_drift']}")
        print()

        print("MODEL DETAILS:")
        for model_name, model_data in results["model_analysis"].items():
            status_icon = "⚠️" if model_data["alert_triggered"] else "✓"
            print(f"  {status_icon} {model_name}: {model_data['drift_status']}")
            if model_data["current_mae"] is not None:
                print(
                    f"      Historical: {model_data['historical_mae_avg']:.4f} "
                    f"(±{model_data['historical_mae_std']:.4f})"
                )
                print(f"      Current: {model_data['current_mae']:.4f}")
                if model_data["degradation_pct"] is not None:
                    print(f"      Delta: {model_data['degradation_pct']:+.2f}%")

        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Forecast Drift Detection - Early Warning System"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Drift threshold multiplier (default: 1.5)",
    )
    parser.add_argument(
        "--historical-months",
        type=int,
        default=6,
        help="Number of months for historical baseline (default: 6)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory containing backtest metrics",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        help="Directory containing retrain logs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for output reports",
    )

    args = parser.parse_args()

    detector = DriftDetector(
        results_dir=args.results_dir,
        weights_dir=args.weights_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        historical_months=args.historical_months,
    )

    try:
        results = detector.detect_drift()
        output_path = detector.save_report(results)

        detector.print_summary(results)

        print(f"\nReport saved to: {output_path}")

        if results["drift_detected"]:
            print("\n⚠️  WARNING: Model drift detected!")
            sys.exit(1)
        else:
            print("\n✓ No model drift detected")
            sys.exit(0)

    except Exception as e:
        print(f"\nError during drift detection: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
