#!/usr/bin/env python3
"""
Dependency Audit Script for SIRENA-KBR Forecasting Project.

Identifies installed packages that are not imported in the codebase.
Uses `pip freeze` to get installed packages and `grep` to search for imports.

Usage:
    python3 scripts/audit_deps.py                  # Run audit
    python3 scripts/audit_deps.py --verbose        # Detailed output
    python3 scripts/audit_deps.py --output FILE     # Custom output path
"""

import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Configuration
PROJECT_ROOT = Path("/home/valalav/_projects/sirena-kbr")
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "dependency_audit.json"

# Directories to scan for imports (excluding cache, __pycache__, etc.)
SCAN_DIRS = [
    "sirena",
    "api",
    "agents",
    "scripts",
    "tests",
    "pages",
    "edge_lab",
]

# Packages that are always used (CLI tools, indirect deps, etc.)
WHITELIST = {
    "pip",
    "setuptools",
    "wheel",
    "pkg-resources",
    "importlib-resources",
    "typing-extensions",
    "certifi",
    "charset-normalizer",
    "idna",
    "urllib3",
    "numpydoc",
    "pandocfilters",
    "soupsieve",
    "smmap",
    "gitdb",
    "pathspec",
    "wcwidth",
    "execnet",
    "iniconfig",
    "pluggy",
    "tomli",
    "zipp",
    "importlib-metadata",
    "mypy-extensions",
    "traitlets",
    "pure-eval",
    "stack-data",
    "parso",
    "jedi",
    "matplotlib-inline",
    "pickleshare",
    "appnope",
    "asttokens",
    "decorator",
    "ipython-genutils",
    "pexpect",
    "ptyprocess",
    "backcall",
    "prompt-toolkit",
}

# Common import patterns
IMPORT_PATTERNS = [
    r"^import\s+(\S+)",
    r"^from\s+(\S+)\s+import",
]


def get_installed_packages() -> Dict[str, str]:
    """
    Get all installed packages using `pip freeze`.

    Returns:
        Dict mapping package name to version (e.g., {'pandas': '2.0.0'})
    """
    result = subprocess.run(
        ["pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )

    packages = {}
    for line in result.stdout.strip().split("\n"):
        if "==" in line:
            name, version = line.split("==", 1)
            # Normalize package name (lowercase, replace - with _)
            normalized = name.lower().replace("-", "_")
            packages[normalized] = version

    return packages


def scan_python_files(directories: List[Path]) -> List[Path]:
    """
    Recursively find all Python files in specified directories.

    Args:
        directories: List of directory paths to scan

    Returns:
        List of Path objects for Python files
    """
    python_files = []

    for directory in directories:
        if not directory.exists():
            continue

        for root, dirs, files in os.walk(directory):
            # Skip cache directories
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith("_")
                and d
                not in [
                    "__pycache__",
                    ".pytest_cache",
                    ".ruff_cache",
                    "node_modules",
                    ".git",
                ]
            ]

            for file in files:
                if file.endswith(".py"):
                    python_files.append(Path(root) / file)

    return python_files


def extract_imports_from_file(file_path: Path) -> Set[str]:
    """
    Extract all imported module names from a Python file.

    Args:
        file_path: Path to the Python file

    Returns:
        Set of imported module names (normalized)
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return set()

    imports = set()
    lines = content.split("\n")

    for line in lines:
        # Skip comments and docstrings
        stripped = line.strip()
        if (
            stripped.startswith("#")
            or stripped.startswith('"""')
            or stripped.startswith("'''")
        ):
            continue

        for pattern in IMPORT_PATTERNS:
            match = re.match(pattern, stripped)
            if match:
                module = match.group(1).split(".")[0]  # Get top-level module
                imports.add(module.lower().replace("-", "_"))
                break

    return imports


def get_imported_modules(files: List[Path]) -> Dict[str, int]:
    """
    Get all imported modules from Python files with usage counts.

    Args:
        files: List of Python files to scan

    Returns:
        Dict mapping module name to import count
    """
    module_counts = defaultdict(int)

    for file_path in files:
        imports = extract_imports_from_file(file_path)
        for module in imports:
            module_counts[module] += 1

    return dict(module_counts)


def compare_dependencies(
    installed: Dict[str, str], imported: Dict[str, int], whitelist: Set[str]
) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    """
    Compare installed packages with imported modules.

    Args:
        installed: Dict of installed packages {name: version}
        imported: Dict of imported modules {name: count}
        whitelist: Set of package names to ignore

    Returns:
        Tuple of (unused_packages, used_packages, imported_summary)
    """
    unused = []
    used = []

    for name, version in installed.items():
        if name in whitelist:
            continue

        # Check if package is imported
        if name in imported:
            used.append(
                {
                    "package": name,
                    "version": version,
                    "import_count": imported[name],
                    "status": "used",
                }
            )
        else:
            unused.append(
                {
                    "package": name,
                    "version": version,
                    "import_count": 0,
                    "status": "unused",
                }
            )

    # Sort unused by package name
    unused.sort(key=lambda x: x["package"])
    used.sort(key=lambda x: x["import_count"], reverse=True)

    return unused, used, imported


def save_audit_report(
    output_path: Path,
    unused: List[Dict],
    used: List[Dict],
    imported_summary: Dict[str, int],
    scan_stats: Dict,
) -> None:
    """
    Save audit report to JSON file.

    Args:
        output_path: Path to output JSON file
        unused: List of unused packages
        used: List of used packages
        imported_summary: Dict of imported modules with counts
        scan_stats: Statistics about the scan
    """
    report = {
        "timestamp": scan_stats["timestamp"],
        "summary": {
            "total_installed": scan_stats["total_installed"],
            "total_scanned_files": scan_stats["total_files"],
            "unique_imports": len(imported_summary),
            "total_packages": scan_stats["total_packages"],
            "unused_count": len(unused),
            "used_count": len(used),
        },
        "unused_packages": unused,
        "used_packages": used,
        "imported_modules": [
            {"module": k, "import_count": v}
            for k, v in sorted(imported_summary.items(), key=lambda x: -x[1])
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✓ Audit report saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Audit project dependencies for unused packages"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output file path (default: {DEFAULT_OUTPUT})",
    )

    args = parser.parse_args()

    # Get scan directories (relative to PROJECT_ROOT)
    scan_paths = [PROJECT_ROOT / d for d in SCAN_DIRS]

    # Step 1: Get installed packages
    print("🔍 Step 1: Getting installed packages...")
    installed = get_installed_packages()
    total_installed = len(installed)
    print(f"   Found {total_installed} installed packages")

    # Step 2: Scan Python files
    print("\n📂 Step 2: Scanning Python files...")
    python_files = scan_python_files(scan_paths)
    total_files = len(python_files)
    print(f"   Found {total_files} Python files")

    # Step 3: Extract imports
    print("\n📦 Step 3: Extracting import statements...")
    imported = get_imported_modules(python_files)
    unique_imports = len(imported)
    print(f"   Found {unique_imports} unique imported modules")

    # Step 4: Compare
    print("\n🔎 Step 4: Comparing dependencies...")
    unused, used, imported_summary = compare_dependencies(
        installed, imported, WHITELIST
    )

    # Step 5: Save report
    print("\n💾 Step 5: Saving audit report...")
    from datetime import datetime

    scan_stats = {
        "timestamp": datetime.now().isoformat(),
        "total_installed": total_installed,
        "total_files": total_files,
        "total_packages": total_installed,
    }

    save_audit_report(args.output, unused, used, imported_summary, scan_stats)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 AUDIT SUMMARY")
    print("=" * 60)
    print(f"Total installed packages: {total_installed}")
    print(f"Total scanned files: {total_files}")
    print(f"Unique imports: {unique_imports}")
    print(f"Used packages: {len(used)}")
    print(f"Unused packages: {len(unused)}")

    if unused and args.verbose:
        print("\n" + "=" * 60)
        print("🗑️  UNUSED PACKAGES (potential candidates for removal)")
        print("=" * 60)
        for pkg in unused[:20]:  # Show first 20
            print(f"  • {pkg['package']}=={pkg['version']}")

        if len(unused) > 20:
            print(f"  ... and {len(unused) - 20} more")

    print(f"\n✅ Audit complete. Full report: {args.output}")


if __name__ == "__main__":
    main()
