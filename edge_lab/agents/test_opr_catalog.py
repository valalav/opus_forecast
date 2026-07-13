#!/usr/bin/env python3

import csv
import sys
from pathlib import Path

CATALOG_FILE = Path("data/opr_data_catalog.csv")
DEST_DIR = Path("data/raw/opr_stat")


def test_catalog_exists():
    assert CATALOG_FILE.exists(), f"Catalog file not found: {CATALOG_FILE}"
    print("✓ Catalog file exists")


def test_catalog_entries():
    with open(CATALOG_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        entries = list(reader)

    assert len(entries) > 0, "Catalog is empty"
    print(f"✓ Catalog has {len(entries)} entries")

    filenames = [e["filename"].lower() for e in entries]

    key_terms = [
        ("консолидированные бюджеты", "Консолидированные бюджеты"),
        ("hh", "hh.ru"),
        ("домклик", "DomClick"),
    ]

    for term, display_name in key_terms:
        found = any(term in fname for fname in filenames)
        assert found, f"Missing {display_name} files in catalog"
        print(f"✓ Catalog contains entries for {display_name} files")


def test_files_accessible():
    assert DEST_DIR.exists(), f"Destination directory not found: {DEST_DIR}"

    files = list(DEST_DIR.glob("*"))
    assert len(files) > 0, f"No files found in {DEST_DIR}"
    print(f"✓ Files accessible in {DEST_DIR}: {len(files)} files")


def test_catalog_metadata():
    with open(CATALOG_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        entries = list(reader)

    assert "size_bytes" in entries[0], "Missing size_bytes column"
    assert "size_mb" in entries[0], "Missing size_mb column"
    assert "modified" in entries[0], "Missing modified column"
    print("✓ Catalog contains file size and modification date")


def main():
    try:
        test_catalog_exists()
        test_catalog_entries()
        test_files_accessible()
        test_catalog_metadata()
        print("\n✅ All tests passed!")
        return 0
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
