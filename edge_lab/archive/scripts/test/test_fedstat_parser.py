#!/usr/bin/env python3
"""
Test script for Fedstat Parser - verifies acceptance criteria for Task 113
"""

import csv
from pathlib import Path


def test_fedstat_parser():
    """Test acceptance criteria for Fedstat Parser task."""

    print("=" * 60)
    print("Testing Fedstat Parser - Task 113")
    print("=" * 60)

    all_passed = True

    # Criterion 1: Catalog csv created with >5000 name-link pairs
    catalog_path = Path("data/fedstat_catalog.csv")
    if catalog_path.exists():
        with open(catalog_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            num_pairs = len(rows) - 1  # Exclude header

        if num_pairs >= 5000:
            print(
                f"✓ Criterion 1: Catalog csv created with {num_pairs} name-link pairs (>5000)"
            )
        else:
            print(
                f"✗ Criterion 1: Catalog has only {num_pairs} entries (<5000 requirement)"
            )
            all_passed = False
    else:
        print(f"✗ Criterion 1: Catalog CSV not found at {catalog_path}")
        all_passed = False

    # Criterion 2: Filtering script identifies Top-50 relevant series
    feed_path = Path("data/prioritized_fedstat_feed.csv")
    if feed_path.exists():
        with open(feed_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            num_top = len(rows)

        if num_top == 50:
            print(f"✓ Criterion 2: Top-{num_top} relevant series identified")
        else:
            print(f"✗ Criterion 2: Expected 50, got {num_top} entries")
            all_passed = False
    else:
        print(f"✗ Criterion 2: Prioritized feed CSV not found at {feed_path}")
        all_passed = False

    # Criterion 3: Zero API errors (safe mode)
    if feed_path.exists():
        with open(feed_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        api_errors = sum(1 for row in rows if row.get("error") and row["error"].strip())

        if api_errors == 0:
            print(f"✓ Criterion 3: Zero API errors ({api_errors} errors found)")
        else:
            print(f"✗ Criterion 3: {api_errors} API errors found")
            all_passed = False

    # Additional verification: Metadata was fetched
    if feed_path.exists():
        with open(feed_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Check if metadata columns exist
        required_cols = [
            "data_available",
            "accessible",
            "has_data",
            "download_available",
            "status_code",
        ]
        header_with_meta = (
            next(open(feed_path, "r", encoding="utf-8")).strip().split(",")
        )

        if all(col in header_with_meta for col in required_cols):
            print(f"✓ Metadata columns present: {', '.join(required_cols)}")
        else:
            print(f"✗ Missing metadata columns")
            all_passed = False

        # Check if metadata was fetched (not just placeholder)
        data_available_count = sum(
            1 for row in rows if row.get("data_available") == "True"
        )
        status_code_count = sum(1 for row in rows if row.get("status_code"))

        if data_available_count > 0 and status_code_count == len(rows):
            print(
                f"✓ Metadata fetched for {len(rows)} candidates ({data_available_count} with data)"
            )
        else:
            print(f"✗ Metadata not properly fetched")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0
    else:
        print("SOME TESTS FAILED ✗")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit(test_fedstat_parser())
