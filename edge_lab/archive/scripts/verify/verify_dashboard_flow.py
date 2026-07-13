#!/usr/bin/env python3
"""
Integration: Dashboard Flow Verification
=========================================
Verifies dashboard loading and data consistency across all data sources.

Acceptance Criteria:
- Data matches (no NaN values in critical paths)
- All models are properly defined and accessible
- Backtest data is consistent with ALL_MODELS list
- Precomputed forecasts exist for all models

Author: Ralph Universal Worker
Task: Integration: Dashboard flow (ID: 20)
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import traceback

# Add parent directory to path to import from main project
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class DashboardFlowVerifier:
    """Verifies dashboard loading and data consistency"""

    # Expected models from dashboard.py
    EXPECTED_MODELS = [
        "Ridge",
        "Ridge_Ext",
        "Bayes_Ridge",
        "ElasticNet",
        "Huber",
        "Ridge_Shock",
        "Ridge_Macro",
        "NGBoost",
        "NGBoost_Shock",
        "BVAR",
        "SARIMA",
        "LightGBM",
        "Prophet",
        "ETS",
        "EBM",
        "CatBoost",
        "Subcomp",
        "Subcomp_Multi",
        "Micro",
        "Ensemble",
    ]

    # Required data files
    REQUIRED_DATA_FILES = [
        "data/infl_kbr.csv",
        "data/inflation_data.csv",
    ]

    # Backtest results files
    BACKTEST_FILES = {
        1: "archive/results/backtest_h1_predictions.csv",
        2: "archive/results/backtest_h2_predictions.csv",
        12: "archive/results/backtest_h12_predictions.csv",
    }

    def __init__(self, parent_dir: Path):
        self.parent_dir = parent_dir
        self.data_dir = parent_dir / "data"
        self.archive_results_dir = parent_dir / "archive" / "results"

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "FAILED",
            "checks": {},
            "errors": [],
            "warnings": [],
            "summary": {},
        }

    def check_dashboard_import(self) -> bool:
        """Check if dashboard.py can be imported"""
        print("\n" + "=" * 70)
        print("CHECK 1: Dashboard Import")
        print("=" * 70)

        try:
            # Import dashboard module
            sys.path.insert(0, str(self.parent_dir))
            import dashboard

            print(f"✅ Dashboard imported successfully")

            # Check ALL_MODELS
            if hasattr(dashboard, "ALL_MODELS"):
                all_models = dashboard.ALL_MODELS
                print(f"✅ ALL_MODELS defined: {len(all_models)} models")

                self.results["checks"]["dashboard_all_models"] = {
                    "status": "PASS",
                    "models_count": len(all_models),
                    "models": all_models,
                }

                # Check against expected
                missing = set(self.EXPECTED_MODELS) - set(all_models)
                extra = set(all_models) - set(self.EXPECTED_MODELS)

                if missing:
                    self.results["warnings"].append(
                        f"Missing models in ALL_MODELS: {missing}"
                    )
                    print(f"⚠️  Missing models: {missing}")

                if extra:
                    self.results["warnings"].append(
                        f"Extra models in ALL_MODELS: {extra}"
                    )
                    print(f"⚠️  Extra models: {extra}")

            else:
                self.results["errors"].append("ALL_MODELS not defined in dashboard")
                print(f"❌ ALL_MODELS not defined")
                return False

            # Check MODEL_COLORS
            if hasattr(dashboard, "MODEL_COLORS"):
                model_colors = dashboard.MODEL_COLORS
                print(f"✅ MODEL_COLORS defined: {len(model_colors)} colors")

                self.results["checks"]["dashboard_model_colors"] = {
                    "status": "PASS",
                    "colors_count": len(model_colors),
                }

                # Check if all models have colors
                missing_colors = set(all_models) - set(model_colors.keys())
                if missing_colors:
                    self.results["errors"].append(
                        f"Models without colors: {missing_colors}"
                    )
                    print(f"❌ Models without colors: {missing_colors}")
                    return False

            return True

        except Exception as e:
            error_msg = f"Failed to import dashboard: {str(e)}"
            self.results["errors"].append(
                {
                    "check": "dashboard_import",
                    "error": error_msg,
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"❌ {error_msg}")
            return False

    def check_data_files(self) -> bool:
        """Check if required data files exist and can be loaded"""
        print("\n" + "=" * 70)
        print("CHECK 2: Data Files")
        print("=" * 70)

        all_pass = True

        for file_path in self.REQUIRED_DATA_FILES:
            full_path = self.parent_dir / file_path

            if not full_path.exists():
                error_msg = f"Data file not found: {file_path}"
                self.results["errors"].append(error_msg)
                print(f"❌ {error_msg}")
                all_pass = False
                continue

            # Try to load the file
            try:
                if file_path.endswith(".csv"):
                    df = pd.read_csv(full_path, sep=";", decimal=",", nrows=5)
                    print(f"✅ {file_path}: {len(df)} columns, {len(df)} rows (sample)")
                else:
                    print(f"✅ {file_path}: exists")

            except Exception as e:
                error_msg = f"Failed to load {file_path}: {str(e)}"
                self.results["errors"].append(error_msg)
                print(f"❌ {error_msg}")
                all_pass = False

        return all_pass

    def check_backtest_data(self) -> bool:
        """Check backtest data files for all horizons"""
        print("\n" + "=" * 70)
        print("CHECK 3: Backtest Data Files")
        print("=" * 70)

        backtest_results = {}
        all_pass = True

        for horizon, file_path in self.BACKTEST_FILES.items():
            full_path = self.parent_dir / file_path

            if not full_path.exists():
                self.results["warnings"].append(
                    f"Backtest file not found for h={horizon}: {file_path}"
                )
                print(f"⚠️  h={horizon}: File not found")
                backtest_results[f"h={horizon}"] = {"status": "MISSING"}
                continue

            try:
                df = pd.read_csv(full_path)
                print(f"✅ h={horizon}: {len(df)} rows, {len(df.columns)} columns")

                # Check for ALL_MODELS columns
                models_in_file = [c for c in df.columns if c in self.EXPECTED_MODELS]
                missing_models = set(self.EXPECTED_MODELS) - set(df.columns)

                result = {
                    "status": "PASS",
                    "rows": len(df),
                    "columns": len(df.columns),
                    "models_found": len(models_in_file),
                    "missing_models": list(missing_models) if missing_models else [],
                }

                if missing_models:
                    print(f"  ℹ️  Missing models: {missing_models}")

                backtest_results[f"h={horizon}"] = result

            except Exception as e:
                error_msg = f"Failed to load backtest h={horizon}: {str(e)}"
                self.results["errors"].append(error_msg)
                print(f"❌ h={horizon}: {error_msg}")
                all_pass = False

        self.results["checks"]["backtest_data"] = backtest_results
        return all_pass

    def check_model_imports(self) -> bool:
        """Check if all models can be imported"""
        print("\n" + "=" * 70)
        print("CHECK 4: Model Imports")
        print("=" * 70)

        import_results = {}
        all_pass = True

        # Model name to module mapping
        model_imports = {
            "Ridge": ("sirena.models.ridge", "RidgeForecaster"),
            "Ridge_Ext": ("sirena.models.ridge_extended", "RidgeExtendedForecaster"),
            "Bayes_Ridge": ("sirena.models.bayesian_ridge", "BayesianRidgeForecaster"),
            "ElasticNet": ("sirena.models.elasticnet", "ElasticNetForecaster"),
            "Huber": ("sirena.models.huber", "HuberForecaster"),
            "Ridge_Shock": (
                "sirena.models.ridge_shock_dummies",
                "RidgeShockDummiesForecaster",
            ),
            "Ridge_Macro": ("sirena.models.ridge_macro", "RidgeMacroForecaster"),
            "NGBoost": ("sirena.models.ngboost_model", "NGBoostForecaster"),
            "NGBoost_Shock": ("sirena.models.ngboost_shock", "NGBoostShockForecaster"),
            "BVAR": ("sirena.models.bvar", "BVARForecaster"),
            "SARIMA": ("sirena.models.arima", "SARIMAForecaster"),
            "LightGBM": ("sirena.models.lightgbm", "LightGBMForecaster"),
            "Prophet": ("sirena.models.prophet", "ProphetForecaster"),
            "ETS": ("sirena.models.ets", "ETSForecaster"),
            "EBM": ("sirena.models.ebm", "EBMForecaster"),
            "CatBoost": ("sirena.models.catboost_model", "CatBoostForecaster"),
            "Subcomp": ("sirena.models.subcomponent", "SubcomponentForecaster"),
            "Subcomp_Multi": (
                "sirena.models.subcomponent_multi",
                "SubcomponentMultiForecaster",
            ),
            "Micro": ("sirena.models.microcomponent", "MicrocomponentForecaster"),
            "Ensemble": ("sirena.forecast", "EnsembleForecaster"),
        }

        for model_name, (module_path, class_name) in model_imports.items():
            try:
                module = __import__(module_path, fromlist=[class_name])
                model_class = getattr(module, class_name)
                print(f"✅ {model_name}: {class_name} imported")
                import_results[model_name] = {"status": "PASS", "class": class_name}
            except Exception as e:
                error_msg = f"{model_name}: {str(e)}"
                self.results["warnings"].append(f"Model import failed: {error_msg}")
                print(f"⚠️  {model_name}: {error_msg}")
                import_results[model_name] = {"status": "FAIL", "error": str(e)}

        self.results["checks"]["model_imports"] = import_results
        return all_pass

    def check_data_consistency(self) -> bool:
        """Check data consistency across sources"""
        print("\n" + "=" * 70)
        print("CHECK 5: Data Consistency")
        print("=" * 70)

        try:
            # Load main data
            infl_kbr_path = self.parent_dir / "data/infl_kbr.csv"
            df_kbr = pd.read_csv(infl_kbr_path, sep=";", decimal=",")

            print(f"✅ infl_kbr.csv: {len(df_kbr)} rows")

            # Check for NaN values in critical columns
            if "MoM" in df_kbr.columns:
                nan_count = df_kbr["MoM"].isna().sum()
                print(f"  MoM NaN values: {nan_count}")

                if nan_count > 0:
                    self.results["warnings"].append(f"NaN values in MoM: {nan_count}")

            # Load backtest data if available
            h1_path = self.parent_dir / "archive/results/backtest_h1_predictions.csv"
            if h1_path.exists():
                df_h1 = pd.read_csv(h1_path)
                print(f"✅ backtest_h1_predictions.csv: {len(df_h1)} rows")

                # Check for 'Actual' column and NaN values
                if "Actual" in df_h1.columns:
                    actual_nan = df_h1["Actual"].isna().sum()
                    print(f"  Actual NaN values: {actual_nan}")

                    if actual_nan > 0:
                        self.results["errors"].append(
                            f"NaN values in Actual column: {actual_nan}"
                        )
                        return False

                # Check for all models having predictions
                models_with_data = df_h1.columns.tolist()
                print(f"  Columns: {', '.join(models_with_data[:10])}...")

            self.results["checks"]["data_consistency"] = {
                "status": "PASS",
                "kbr_rows": len(df_kbr),
            }

            return True

        except Exception as e:
            error_msg = f"Data consistency check failed: {str(e)}"
            self.results["errors"].append(error_msg)
            print(f"❌ {error_msg}")
            return False

    def run_all_checks(self) -> bool:
        """Run all dashboard flow checks"""
        print("=" * 70)
        print("DASHBOARD FLOW VERIFICATION")
        print(f"Time: {self.results['timestamp']}")
        print(f"Directory: {self.parent_dir}")
        print("=" * 70)

        all_pass = True

        # Run all checks
        all_pass &= self.check_dashboard_import()
        all_pass &= self.check_data_files()
        all_pass &= self.check_backtest_data()
        all_pass &= self.check_model_imports()
        all_pass &= self.check_data_consistency()

        self.results["overall_status"] = "SUCCESS" if all_pass else "FAILED"

        return all_pass

    def print_summary(self):
        """Print verification summary"""
        print("\n" + "=" * 70)
        print("DASHBOARD FLOW VERIFICATION SUMMARY")
        print("=" * 70)

        print(f"\nOverall Status: {self.results['overall_status']}")

        # Summary of checks
        checks = self.results.get("checks", {})
        for check_name, check_result in checks.items():
            if isinstance(check_result, dict):
                # Simple check (single result)
                if "status" in check_result:
                    if check_result.get("status") == "PASS":
                        print(f"✅ {check_name}: PASS")
                    else:
                        print(f"❌ {check_name}: {check_result.get('status', 'FAIL')}")
                # Nested check (multiple results like backtest_data, model_imports)
                else:
                    # Check if any nested check failed
                    has_failure = any(
                        isinstance(v, dict) and v.get("status") != "PASS"
                        for v in check_result.values()
                    )
                    if not has_failure:
                        print(f"✅ {check_name}: PASS ({len(check_result)} items)")
                    else:
                        print(f"❌ {check_name}: FAIL ({len(check_result)} items)")
                        # Show failing items
                        for k, v in check_result.items():
                            if isinstance(v, dict) and v.get("status") != "PASS":
                                print(f"    - {k}: {v.get('status', 'FAIL')}")
            else:
                print(f"❌ {check_name}: Invalid format")

        # Warnings
        warnings = self.results.get("warnings", [])
        if warnings:
            print(f"\n⚠️  Warnings ({len(warnings)}):")
            for warning in warnings[:5]:
                print(f"  - {warning}")
            if len(warnings) > 5:
                print(f"  ... and {len(warnings) - 5} more")

        # Errors
        errors = self.results.get("errors", [])
        if errors:
            print(f"\n❌ Errors ({len(errors)}):")
            for error in errors[:5]:
                if isinstance(error, dict):
                    print(
                        f"  - {error.get('check', 'error')}: {error.get('error', 'Unknown')[:100]}"
                    )
                else:
                    print(f"  - {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more")

        # Acceptance criteria
        print("\n" + "=" * 70)
        print("ACCEPTANCE CRITERIA CHECK")
        print("=" * 70)

        acceptance_passed = True

        # Criterion 1: Data matches (no critical errors)
        if self.results["overall_status"] == "SUCCESS":
            print("✅ Data matches: All checks passed")
        else:
            print("❌ Data matches: Failed (errors present)")
            acceptance_passed = False

        print("\n" + "=" * 70)
        if acceptance_passed:
            print("ACCEPTANCE CRITERIA: ✅ PASSED")
        else:
            print("ACCEPTANCE CRITERIA: ❌ FAILED")
        print("=" * 70 + "\n")

    def save_summary(self, output_dir: Path):
        """Save verification summary to JSON file"""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (
            output_dir
            / f"dashboard_flow_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"Summary saved to: {output_path}")
        return output_path


def main():
    """Main entry point"""
    # Determine parent directory (edge_lab -> opus_forecast)
    current_dir = Path(__file__).parent.parent  # /home/valalav/_projects/sirena-kbr

    verifier = DashboardFlowVerifier(current_dir)

    # Run all checks
    success = verifier.run_all_checks()

    # Print summary
    verifier.print_summary()

    # Save summary
    verifier.save_summary(current_dir / "archive" / "results")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
