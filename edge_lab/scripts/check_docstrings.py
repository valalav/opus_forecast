#!/usr/bin/env python3
"""
Docstring Standardizer - Checks and reports on docstring compliance.

Usage:
    python scripts/check_docstrings.py [--help] [--path PATH] [--format FORMAT]
"""

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def check_docstring(node: ast.AST) -> Tuple[bool, str]:
    """Check if an AST node has a docstring."""
    docstring = ast.get_docstring(node)
    if docstring:
        return True, docstring[:50] + "..." if len(docstring) > 50 else docstring
    return False, ""


def analyze_file(filepath: str) -> Dict:
    """Analyze a Python file for docstring compliance."""
    results = {
        "filepath": filepath,
        "module_docstring": False,
        "classes": [],
        "functions": [],
        "total": 0,
        "documented": 0,
    }

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        tree = ast.parse(source)

        # Check module docstring
        has_doc, doc = check_docstring(tree)
        results["module_docstring"] = has_doc
        results["total"] += 1
        if has_doc:
            results["documented"] += 1

        # Check classes and functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                has_doc, doc = check_docstring(node)
                results["classes"].append({
                    "name": node.name,
                    "line": node.lineno,
                    "has_docstring": has_doc,
                })
                results["total"] += 1
                if has_doc:
                    results["documented"] += 1

            elif isinstance(node, ast.FunctionDef):
                has_doc, doc = check_docstring(node)
                results["functions"].append({
                    "name": node.name,
                    "line": node.lineno,
                    "has_docstring": has_doc,
                })
                results["total"] += 1
                if has_doc:
                    results["documented"] += 1

    except SyntaxError as e:
        results["error"] = f"Syntax error: {e}"
    except Exception as e:
        results["error"] = f"Error: {e}"

    return results


def report_results(results: List[Dict], output_format: str = "text") -> str:
    """Generate a report from analysis results."""
    if output_format == "json":
        import json
        return json.dumps(results, indent=2)

    lines = []
    total_items = 0
    total_documented = 0

    for r in results:
        if "error" in r:
            lines.append(f"ERROR: {r['filepath']}: {r['error']}")
            continue

        total_items += r["total"]
        total_documented += r["documented"]

        missing = []
        if not r["module_docstring"]:
            missing.append("module")
        for c in r["classes"]:
            if not c["has_docstring"]:
                missing.append(f"class {c['name']} (line {c['line']})")
        for f in r["functions"]:
            if not f["has_docstring"] and not f["name"].startswith("_"):
                missing.append(f"function {f['name']} (line {f['line']})")

        if missing:
            lines.append(f"\n{r['filepath']}:")
            for m in missing:
                lines.append(f"  - Missing: {m}")

    coverage = (total_documented / total_items * 100) if total_items > 0 else 0

    lines.insert(0, "=" * 60)
    lines.insert(1, "Docstring Analysis Report")
    lines.insert(2, "=" * 60)
    lines.insert(3, f"Total items: {total_items}")
    lines.insert(4, f"Documented: {total_documented}")
    lines.insert(5, f"Coverage: {coverage:.1f}%")
    lines.insert(6, "-" * 60)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Check Python files for docstring compliance"
    )
    parser.add_argument(
        "--path", "-p",
        default=".",
        help="Path to check (file or directory)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively check directories"
    )

    args = parser.parse_args()

    path = Path(args.path)
    files = []

    if path.is_file():
        files = [str(path)]
    elif path.is_dir():
        pattern = "**/*.py" if args.recursive else "*.py"
        files = [str(f) for f in path.glob(pattern)]
    else:
        print(f"Error: Path not found: {path}", file=sys.stderr)
        sys.exit(1)

    if not files:
        print("No Python files found", file=sys.stderr)
        sys.exit(1)

    results = [analyze_file(f) for f in files]
    report = report_results(results, args.format)
    print(report)

    # Exit with error if coverage < 50%
    total = sum(r.get("total", 0) for r in results)
    documented = sum(r.get("documented", 0) for r in results)
    if total > 0 and documented / total < 0.5:
        sys.exit(1)


if __name__ == "__main__":
    main()
