#!/usr/bin/env python3
"""
Docstring Standardizer Checker

Scans sirena/models/ directory to check for Google-style docstrings
with Args and Returns sections.

Usage:
    python3 scripts/check_docstrings.py

Output:
    - Console report with summary
    - JSON report saved to data/docstring_report.json
"""

import ast
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


class DocstringChecker:
    """Check for Google-style docstrings in Python files."""

    def __init__(self, models_dir: str = "sirena/models"):
        self.models_dir = Path(models_dir)
        self.results = defaultdict(dict)

    def check_file(self, filepath: Path) -> Dict:
        """Check a single Python file for docstring compliance.

        Args:
            filepath: Path to Python file

        Returns:
            Dictionary with file status and issues
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content, filename=str(filepath))

            # Check for Args and Returns in docstrings
            has_args = "Args:" in content
            has_returns = "Returns:" in content

            # Count functions with docstrings
            functions_with_docstrings = []
            functions_without_docstrings = []

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    docstring = ast.get_docstring(node)
                    if node.name.startswith("_") and node.name != "__init__":
                        continue

                    if docstring:
                        functions_with_docstrings.append(node.name)
                        # Check if docstring has Args/Returns for main methods
                        if node.name in ["fit", "predict", "forecast", "backtest"]:
                            if "Args:" not in docstring:
                                has_args = False
                            if "Returns:" not in docstring:
                                has_returns = False
                    else:
                        if not node.name.startswith("_"):
                            functions_without_docstrings.append(node.name)

            # Determine status
            if has_args and has_returns and not functions_without_docstrings:
                status = "OK"
            elif has_args or has_returns:
                status = "PARTIAL"
            else:
                status = "MISSING"

            return {
                "status": status,
                "has_args": has_args,
                "has_returns": has_returns,
                "functions_with_docstrings": len(functions_with_docstrings),
                "functions_without_docstrings": len(functions_without_docstrings),
                "missing_docstring_functions": functions_without_docstrings,
            }

        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    def scan_directory(self) -> Dict:
        """Scan all Python files in models directory.

        Returns:
            Dictionary with scan results for all files
        """
        results = {
            "total_files": 0,
            "ok": 0,
            "partial": 0,
            "missing": 0,
            "error": 0,
            "files": {},
        }

        for py_file in sorted(self.models_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue

            results["total_files"] += 1
            file_result = self.check_file(py_file)
            results["files"][py_file.name] = file_result

            status = file_result.get("status", "error").lower()
            if status in results:
                results[status] += 1

        return results

    def print_report(self, results: Dict):
        """Print formatted report to console.

        Args:
            results: Dictionary with scan results
        """
        print("=" * 70)
        print("DOCSTRING STANDARDIZATION REPORT")
        print("=" * 70)

        print(f"\nTotal files scanned: {results['total_files']}")
        print(
            f"  OK (complete Google-style):      {results['ok']} ({results['ok'] / results['total_files'] * 100:.1f}%)"
        )
        print(
            f"  PARTIAL (Args or Returns):       {results['partial']} ({results['partial'] / results['total_files'] * 100:.1f}%)"
        )
        print(
            f"  MISSING (no Google-style docs):  {results['missing']} ({results['missing'] / results['total_files'] * 100:.1f}%)"
        )
        print(f"  ERROR (parse errors):             {results['error']}")

        # Report files with issues
        if results["missing"] > 0:
            print("\n" + "=" * 70)
            print("FILES MISSING GOOGLE-STYLE DOCSTRINGS")
            print("=" * 70)
            for filename, info in results["files"].items():
                if info.get("status") == "MISSING":
                    print(f"\n❌ {filename}")
                    print(
                        f"   Functions without docstrings: {len(info.get('missing_docstring_functions', []))}"
                    )
                    if info.get("missing_docstring_functions"):
                        for func in info["missing_docstring_functions"]:
                            print(f"     - {func}")

        if results["partial"] > 0:
            print("\n" + "=" * 70)
            print("FILES WITH PARTIAL DOCSTRINGS")
            print("=" * 70)
            for filename, info in results["files"].items():
                if info.get("status") == "PARTIAL":
                    print(f"\n⚠️  {filename}")
                    if not info.get("has_args"):
                        print(f"   Missing: Args section")
                    if not info.get("has_returns"):
                        print(f"   Missing: Returns section")

        # Report OK files (summary only)
        if results["ok"] > 0:
            print("\n" + "=" * 70)
            print(
                f"FILES WITH COMPLETE GOOGLE-STYLE DOCSTRINGS ({results['ok']} files)"
            )
            print("=" * 70)
            ok_files = [
                f for f, i in results["files"].items() if i.get("status") == "OK"
            ]
            for i, f in enumerate(ok_files, 1):
                print(f"{i}. {f}")

        print("\n" + "=" * 70)

    def save_report(
        self, results: Dict, output_path: str = "data/docstring_report.json"
    ):
        """Save report to JSON file.

        Args:
            results: Dictionary with scan results
            output_path: Path to save JSON report
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Report saved to: {output_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check for Google-style docstrings in sirena/models/"
    )
    parser.add_argument(
        "--models-dir",
        default="sirena/models",
        help="Path to models directory (default: sirena/models)",
    )
    parser.add_argument(
        "--output",
        default="data/docstring_report.json",
        help="Path to save JSON report (default: data/docstring_report.json)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary, no detailed file lists",
    )

    args = parser.parse_args()

    checker = DocstringChecker(args.models_dir)
    results = checker.scan_directory()

    if not args.quiet:
        checker.print_report(results)
    else:
        print(
            f"Total: {results['total_files']}, OK: {results['ok']}, PARTIAL: {results['partial']}, MISSING: {results['missing']}"
        )

    checker.save_report(results, args.output)


if __name__ == "__main__":
    main()
