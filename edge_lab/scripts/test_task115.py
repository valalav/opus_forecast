#!/usr/bin/env python3

import csv
import subprocess
from pathlib import Path

OUTPUT_FILE = "data/raw_yugu_dump.csv"
AUDIT_FILE = "data/audit_file_index.csv"


def test_file_exists():
    return Path(OUTPUT_FILE).exists()


def test_file_size():
    size_mb = Path(OUTPUT_FILE).stat().st_size / (1024 * 1024)
    return size_mb > 1.0, size_mb


def test_data_coverage():
    with open(AUDIT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        yugu_sheets = [
            row for row in reader if "Основная статистика ЮГУ.xlsx" in row["filename"]
        ]

    total_sheets = len(yugu_sheets)

    extracted_sheets = set()
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            extracted_sheets.add(row["Sheet"])

    extracted_count = len(extracted_sheets)
    coverage_pct = (extracted_count / total_sheets) * 100

    return coverage_pct > 65.0, coverage_pct, extracted_count, total_sheets


def test_record_count():
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = sum(1 for _ in reader)

    return count > 5000, count


def test_data_quality():
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    non_null = sum(1 for row in rows if row["Value"] and row["Value"].strip())

    return non_null == total, total, non_null, total - non_null


def main():
    results = []

    print("=== Task 115 Verification ===")
    print()

    # Test 1: File exists
    exists = test_file_exists()
    results.append(("File exists", exists, str(exists)))
    print(f"1. File exists: {exists}")

    # Test 2: File size > 1MB
    size_ok, size_mb = test_file_size()
    results.append(("File size > 1MB", size_ok, f"{size_mb:.2f} MB"))
    print(f"2. File size > 1MB: {size_ok} ({size_mb:.2f} MB)")

    # Test 3: Coverage
    coverage_ok, coverage_pct, extracted, total = test_data_coverage()
    results.append(
        ("Coverage > 65%", coverage_ok, f"{coverage_pct:.1f}% ({extracted}/{total})")
    )
    print(
        f"3. Coverage > 65%: {coverage_ok} ({coverage_pct:.1f}% = {extracted}/{total} sheets)"
    )

    # Test 4: Record count
    count_ok, count = test_record_count()
    results.append(("Record count > 5000", count_ok, f"{count} records"))
    print(f"4. Record count > 5000: {count_ok} ({count} records)")

    # Test 5: Data quality
    quality_ok, total, non_null, nulls = test_data_quality()
    results.append(
        ("Data quality (no nulls)", quality_ok, f"{non_null}/{total} non-null")
    )
    print(f"5. Data quality (no nulls): {quality_ok} ({non_null}/{total} non-null)")

    print()
    print("=== Summary ===")
    all_pass = all(r[1] for r in results)
    print(f"All tests passed: {all_pass}")

    for test, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test}: {detail}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())
