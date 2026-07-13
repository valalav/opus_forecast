#!/usr/bin/env python3
"""
Data Validation Script for Opus Edge Lab
Checks for null values, outliers, and data quality issues in CSV/JSON data files.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime


class DataValidator:
    def __init__(self, data_dir: str = None):
        self.data_dir = (
            Path(data_dir) if data_dir else Path(__file__).parent.parent / "data"
        )
        self.results = {}
        self.has_errors = False

    def find_data_files(self) -> List[Path]:
        """Find all CSV and JSON files in the data directory."""
        files = []
        for ext in ["*.csv", "*.json"]:
            files.extend(self.data_dir.glob(ext))
            files.extend(self.data_dir.rglob(ext))
        return sorted(files)

    def check_nulls(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Check for null values in dataframe."""
        null_counts = df.isnull().sum()
        total_nulls = null_counts.sum()
        total_cells = df.shape[0] * df.shape[1]
        null_percentage = (total_nulls / total_cells) * 100 if total_cells > 0 else 0

        return {
            "filename": filename,
            "total_rows": df.shape[0],
            "total_columns": df.shape[1],
            "total_nulls": int(total_nulls),
            "null_percentage": round(null_percentage, 2),
            "nulls_by_column": null_counts[null_counts > 0].to_dict(),
        }

    def detect_outliers_iqr(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Detect outliers using IQR method for numeric columns."""
        outliers_info = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                continue

            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:
                continue

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(col_data)) * 100

            if outlier_count > 0:
                outliers_info[col] = {
                    "outlier_count": outlier_count,
                    "outlier_percentage": round(outlier_percentage, 2),
                    "lower_bound": round(lower_bound, 4),
                    "upper_bound": round(upper_bound, 4),
                    "min_value": float(col_data.min()),
                    "max_value": float(col_data.max()),
                    "outlier_values": outliers.tolist()[:10],
                }

        return {
            "filename": filename,
            "columns_with_outliers": list(outliers_info.keys()),
            "total_columns_with_outliers": len(outliers_info),
            "outlier_details": outliers_info,
        }

    def detect_outliers_zscore(
        self, df: pd.DataFrame, filename: str, threshold: float = 3.0
    ) -> Dict[str, Any]:
        """Detect outliers using Z-score method for numeric columns."""
        outliers_info = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) < 3:
                continue

            mean = col_data.mean()
            std = col_data.std()

            if std == 0:
                continue

            z_scores = np.abs((col_data - mean) / std)
            outliers = col_data[z_scores > threshold]
            outlier_count = len(outliers)
            outlier_percentage = (outlier_count / len(col_data)) * 100

            if outlier_count > 0:
                outliers_info[col] = {
                    "outlier_count": outlier_count,
                    "outlier_percentage": round(outlier_percentage, 2),
                    "threshold": threshold,
                    "mean": round(mean, 4),
                    "std": round(std, 4),
                    "max_zscore": float(z_scores.max()),
                }

        return {
            "filename": filename,
            "method": "zscore",
            "columns_with_outliers": list(outliers_info.keys()),
            "outlier_details": outliers_info,
        }

    def check_data_quality(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Check overall data quality metrics."""
        issues = []

        if df.empty:
            issues.append("DataFrame is empty")

        if df.duplicated().sum() > 0:
            dup_count = df.duplicated().sum()
            issues.append(f"Found {dup_count} duplicate rows")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (df[col] == 0).sum() > 0:
                zero_count = (df[col] == 0).sum()
                zero_pct = (zero_count / len(df)) * 100
                if zero_pct > 50:
                    issues.append(
                        f"Column '{col}': {zero_pct:.1f}% zeros (potential data issue)"
                    )

        if df.isnull().all().any():
            null_cols = df.columns[df.isnull().all()].tolist()
            issues.append(f"Columns completely null: {null_cols}")

        return {
            "filename": filename,
            "quality_issues": issues,
            "issue_count": len(issues),
            "has_quality_issues": len(issues) > 0,
        }

    def validate_csv_file(self, filepath: Path) -> Dict[str, Any]:
        """Validate a single CSV file."""
        try:
            df = pd.read_csv(filepath)

            null_report = self.check_nulls(df, filepath.name)
            outliers_iqr = self.detect_outliers_iqr(df, filepath.name)
            outliers_zscore = self.detect_outliers_zscore(df, filepath.name)
            quality_report = self.check_data_quality(df, filepath.name)

            return {
                "file": str(filepath),
                "status": "success",
                "null_analysis": null_report,
                "outlier_analysis_iqr": outliers_iqr,
                "outlier_analysis_zscore": outliers_zscore,
                "quality_check": quality_report,
            }
        except Exception as e:
            return {"file": str(filepath), "status": "error", "error": str(e)}

    def validate_json_file(self, filepath: Path) -> Dict[str, Any]:
        """Validate a single JSON file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            df = None
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                if "data" in data and isinstance(data["data"], list):
                    df = pd.DataFrame(data["data"])
                elif all(isinstance(v, (list, dict)) for v in data.values() if v):
                    try:
                        df = pd.DataFrame.from_dict(data, orient="index").T
                    except:
                        pass

            if df is None or df.empty or df.shape[0] < 1 or df.shape[1] < 1:
                return {
                    "file": str(filepath),
                    "status": "skip",
                    "reason": "JSON does not contain tabular data",
                }

            null_report = self.check_nulls(df, filepath.name)
            quality_report = self.check_data_quality(df, filepath.name)

            return {
                "file": str(filepath),
                "status": "success",
                "null_analysis": null_report,
                "quality_check": quality_report,
            }
        except Exception as e:
            return {"file": str(filepath), "status": "error", "error": str(e)}

    def run_validation(self) -> Dict[str, Any]:
        """Run validation on all data files."""
        files = self.find_data_files()

        if not files:
            print("No data files found!")
            return {"status": "no_files", "message": "No CSV or JSON files found"}

        print(f"Found {len(files)} data files to validate")
        print("=" * 80)

        validation_results = []

        for filepath in files:
            print(f"\nValidating: {filepath.relative_to(self.data_dir.parent)}")

            if filepath.suffix == ".csv":
                result = self.validate_csv_file(filepath)
            elif filepath.suffix == ".json":
                result = self.validate_json_file(filepath)
            else:
                continue

            validation_results.append(result)

            if result["status"] == "error":
                print(f"  ❌ ERROR: {result['error']}")
                self.has_errors = True
            elif result["status"] == "success":
                null_pct = result["null_analysis"]["null_percentage"]
                if "outlier_analysis_iqr" in result:
                    outliers_iqr = result["outlier_analysis_iqr"][
                        "total_columns_with_outliers"
                    ]
                    print(
                        f"  ✅ Valid | Nulls: {null_pct}% | Outlier columns: {outliers_iqr}"
                    )
                else:
                    print(
                        f"  ✅ Valid | Nulls: {null_pct}% | (JSON - outlier analysis skipped)"
                    )
            else:
                print(f"  ⏭️  Skipped: {result.get('reason', 'Unknown reason')}")

        return {
            "timestamp": datetime.now().isoformat(),
            "total_files": len(files),
            "successful": len(
                [r for r in validation_results if r["status"] == "success"]
            ),
            "failed": len([r for r in validation_results if r["status"] == "error"]),
            "skipped": len([r for r in validation_results if r["status"] == "skip"]),
            "results": validation_results,
        }

    def print_summary(self, results: Dict[str, Any]):
        """Print validation summary."""
        print("\n" + "=" * 80)
        print("DATA VALIDATION SUMMARY")
        print("=" * 80)

        if results.get("status") == "no_files":
            print(results["message"])
            return

        print(f"Total files processed: {results['total_files']}")
        print(f"Successful: {results['successful']} ✅")
        print(f"Failed: {results['failed']} ❌")
        print(f"Skipped: {results['skipped']} ⏭️")
        print(f"\nTimestamp: {results['timestamp']}")

        print("\n" + "-" * 80)
        print("DETAILED NULL COUNTS")
        print("-" * 80)

        for result in results["results"]:
            if result["status"] == "success":
                null_info = result["null_analysis"]
                filename = Path(result["file"]).name
                print(f"\n📄 {filename}")
                print(
                    f"   Total nulls: {null_info['total_nulls']} ({null_info['null_percentage']}%)"
                )

                if null_info["nulls_by_column"]:
                    for col, count in null_info["nulls_by_column"].items():
                        print(f"      • {col}: {count} nulls")
                else:
                    print("   ✅ No null values found")

        print("\n" + "-" * 80)
        print("OUTLIER DETECTION (IQR Method)")
        print("-" * 80)

        for result in results["results"]:
            if result["status"] == "success":
                if "outlier_analysis_iqr" in result:
                    outlier_info = result["outlier_analysis_iqr"]
                    filename = Path(result["file"]).name

                    if outlier_info["total_columns_with_outliers"] > 0:
                        print(f"\n📄 {filename}")
                        print(
                            f"   Columns with outliers: {outlier_info['total_columns_with_outliers']}"
                        )

                        for col, details in outlier_info["outlier_details"].items():
                            print(
                                f"      • {col}: {details['outlier_count']} outliers ({details['outlier_percentage']}%)"
                            )
                            print(
                                f"        Range: [{details['lower_bound']}, {details['upper_bound']}]"
                            )
                            print(
                                f"        Min/Max: [{details['min_value']}, {details['max_value']}]"
                            )

        print("\n" + "-" * 80)
        print("DATA QUALITY ISSUES")
        print("-" * 80)

        quality_issues_found = False
        for result in results["results"]:
            if (
                result["status"] == "success"
                and result["quality_check"]["has_quality_issues"]
            ):
                filename = Path(result["file"]).name
                issues = result["quality_check"]["quality_issues"]
                print(f"\n📄 {filename}")
                for issue in issues:
                    print(f"   ⚠️  {issue}")
                quality_issues_found = True

        if not quality_issues_found:
            print("\n✅ No data quality issues detected")

        print("\n" + "=" * 80)
        if self.has_errors or results["failed"] > 0:
            print("VALIDATION COMPLETED WITH ISSUES")
            return 1
        else:
            print("VALIDATION COMPLETED SUCCESSFULLY")
            return 0


def main():
    validator = DataValidator()
    results = validator.run_validation()
    exit_code = validator.print_summary(results)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
