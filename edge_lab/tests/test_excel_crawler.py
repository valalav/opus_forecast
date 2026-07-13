#!/usr/bin/env python3

import csv
import os
from pathlib import Path


def verify_crawler_output():
    audit_file = Path("data/audit_file_index.csv")

    print("=" * 60)
    print("VERIFICATION: Excel Crawler Output")
    print("=" * 60)

    # Criterion 1: Script exists
    script_path = Path("agents/excel_crawler.py")
    criterion_1 = script_path.exists() and script_path.is_file()
    print(f"\n[1] Script 'agents/excel_crawler.py' exists: {criterion_1}")
    if criterion_1:
        print(f"    Size: {script_path.stat().st_size} bytes")

    if not audit_file.exists():
        print(f"\n❌ Audit file not found: {audit_file}")
        return False

    # Read audit file
    with open(audit_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total_sheets = len(rows)
    criterion_2 = total_sheets > 200
    print(f"\n[2] Output contains > 200 rows (sheets): {criterion_2}")
    print(f"    Total sheets: {total_sheets}")

    # Find the large file
    large_file_sheets = [
        r for r in rows if r["filename"] == "Основная статистика ЮГУ.xlsx"
    ]
    large_file_count = len(large_file_sheets)
    criterion_3 = large_file_count >= 100
    print(f"\n[3] Successfully lists all sheets in the 150MB file: {criterion_3}")
    print(f"    Sheets in 'Основная статистика ЮГУ.xlsx': {large_file_count}")

    # Show some sample sheets from large file
    if large_file_sheets:
        print(f"\n    Sample sheets (first 10):")
        for sheet in large_file_sheets[:10]:
            print(f"      - {sheet['sheet_name']} (index: {sheet['sheet_index']})")

    # Check file size is correct
    large_file_sheet = large_file_sheets[0] if large_file_sheets else None
    if large_file_sheet:
        expected_size = 137340439  # ~137 MB
        actual_size = int(large_file_sheet["file_size_bytes"])
        size_ok = abs(actual_size - expected_size) < 1000000  # Allow 1MB tolerance
        print(f"\n[3b] File size verification: {size_ok}")
        print(f"    Expected: ~137 MB, Actual: {large_file_sheet['file_size_human']}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = criterion_1 and criterion_2 and criterion_3
    print(f"\nAll criteria met: {all_passed}")
    print(f"  [1] Script exists: {criterion_1}")
    print(f"  [2] > 200 sheets: {criterion_2} ({total_sheets})")
    print(f"  [3] Large file sheets: {criterion_3} ({large_file_count})")

    if not all_passed:
        print("\n❌ VERIFICATION FAILED")
        return False
    else:
        print("\n✅ VERIFICATION PASSED")
        return True


if __name__ == "__main__":
    import sys

    passed = verify_crawler_output()
    sys.exit(0 if passed else 1)
