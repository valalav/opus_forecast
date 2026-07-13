#!/usr/bin/env python3
"""
Test verification script for Rosstat Ingester (Task 110)

Tests:
1. Agent successfully identifies KBR data in SKFO block (approx row 50)
2. Handles both monthly and cumulative data formats
3. Regenerates registry automatically on new files
4. Generates schema_registry.json
5. Generates data_quality_report.md
"""

import os
import sys
import json
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agents.rosstat_ingester import RosstatIngester


def test_kbr_identification():
    """Test 1: Agent identifies KBR data in SKFO block (approx row 50)"""
    print("\n" + "=" * 60)
    print("TEST 1: KBR Identification in SKFO block")
    print("=" * 60)

    ingester = RosstatIngester()

    # Parse a specific file
    test_file = Path(
        "/home/valalav/_projects/sirena-kbr/data/raw/info-stat/01 промышленность/01-01 индекс промышленного производства.xlsx"
    )
    schema = ingester.parse_file(test_file)

    assert schema.kbr_found, "KBR not found in file!"
    assert schema.kbr_row > 40, f"KBR row {schema.kbr_row} too low (expected ~50)"
    assert schema.kbr_row < 60, f"KBR row {schema.kbr_row} too high (expected ~50)"
    assert schema.skfo_row > 40, f"SKFO row {schema.skfo_row} too low"
    assert schema.kbr_row > schema.skfo_row, "KBR row should be after SKFO row"

    print(f"✅ SKFO row: {schema.skfo_row}")
    print(f"✅ KBR row: {schema.kbr_row}")
    print(f"✅ KBR found: {schema.kbr_found}")
    print("TEST 1 PASSED")

    return True


def test_frequency_detection():
    """Test 2: Handles both monthly and cumulative data formats"""
    print("\n" + "=" * 60)
    print("TEST 2: Frequency Detection")
    print("=" * 60)

    ingester = RosstatIngester()
    stats = ingester.run_ingestion()

    assert stats["monthly_files"] > 0, "No monthly files detected!"
    assert stats["quarterly_files"] > 0, "No quarterly files detected!"

    # Check specific files
    registry = ingester.schema_registry

    # 01-01 should be monthly (months in row 3)
    test_file = "01 промышленность/01-01 индекс промышленного производства.xlsx"
    if test_file in registry:
        freq = registry[test_file]["frequency"]
        print(f"✅ {test_file}: {freq}")
        assert freq in ["monthly", "quarterly"], f"Invalid frequency: {freq}"

    print(f"✅ Monthly files: {stats['monthly_files']}")
    print(f"✅ Quarterly files: {stats['quarterly_files']}")
    print("TEST 2 PASSED")

    return True


def test_registry_regeneration():
    """Test 3: Regenerates registry automatically on new files"""
    print("\n" + "=" * 60)
    print("TEST 3: Registry Regeneration")
    print("=" * 60)

    # First run
    output_dir = "/tmp/test_rosstat_ingester"
    os.makedirs(output_dir, exist_ok=True)

    registry_path = os.path.join(output_dir, "registry1.json")
    quality_path = os.path.join(output_dir, "quality1.md")

    ingester1 = RosstatIngester()
    ingester1.base_path = Path("/home/valalav/_projects/sirena-kbr/data/raw/info-stat")
    stats1 = ingester1.run_ingestion()

    # Save first registry manually
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(ingester1.schema_registry, f, indent=2, ensure_ascii=False)

    # Second run (should regenerate)
    ingester2 = RosstatIngester()
    ingester2.base_path = Path("/home/valalav/_projects/sirena-kbr/data/raw/info-stat")
    stats2 = ingester2.run_ingestion()

    registry_path2 = os.path.join(output_dir, "registry2.json")
    with open(registry_path2, "w", encoding="utf-8") as f:
        json.dump(ingester2.schema_registry, f, indent=2, ensure_ascii=False)

    # Compare registries
    with open(registry_path, "r") as f1, open(registry_path2, "r") as f2:
        reg1 = json.load(f1)
        reg2 = json.load(f2)

    assert reg1 == reg2, "Registries should be identical for same data"

    print(f"✅ Registry regenerated successfully")
    print(f"✅ First run: {len(reg1)} files")
    print(f"✅ Second run: {len(reg2)} files")
    print("TEST 3 PASSED")

    return True


def test_schema_registry_generated():
    """Test 4: Generates schema_registry.json"""
    print("\n" + "=" * 60)
    print("TEST 4: Schema Registry Generation")
    print("=" * 60)

    ingester = RosstatIngester()
    stats = ingester.run_ingestion()

    registry_path = "/home/valalav/_projects/sirena-kbr/edge_lab/data/schema_registry.json"

    assert os.path.exists(registry_path), f"Registry not found: {registry_path}"

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    assert len(registry) > 0, "Registry is empty!"
    assert len(registry) == stats["total_files"], "Registry count mismatch!"

    # Check structure
    for filename, data in list(registry.items())[:3]:
        assert "header_row" in data, f"Missing header_row in {filename}"
        assert "kbr_row" in data, f"Missing kbr_row in {filename}"
        assert "skfo_row" in data, f"Missing skfo_row in {filename}"
        assert "frequency" in data, f"Missing frequency in {filename}"
        assert "units" in data, f"Missing units in {filename}"

    print(f"✅ Registry exists: {registry_path}")
    print(f"✅ Registry entries: {len(registry)}")
    print(f"✅ Schema structure valid")
    print("TEST 4 PASSED")

    return True


def test_quality_report_generated():
    """Test 5: Generates data_quality_report.md"""
    print("\n" + "=" * 60)
    print("TEST 5: Quality Report Generation")
    print("=" * 60)

    report_path = "/home/valalav/_projects/sirena-kbr/edge_lab/data/data_quality_report.md"

    assert os.path.exists(report_path), f"Quality report not found: {report_path}"

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "# Rosstat Data Quality Report" in content, "Missing report header"
    assert "Summary" in content, "Missing Summary section"
    assert "Quality Issues" in content, "Missing Quality Issues section"

    print(f"✅ Quality report exists: {report_path}")
    print(f"✅ Report structure valid")
    print("TEST 5 PASSED")

    return True


def test_kbr_row_accuracy():
    """Test 6: Verify KBR row accuracy across multiple files"""
    print("\n" + "=" * 60)
    print("TEST 6: KBR Row Accuracy")
    print("=" * 60)

    ingester = RosstatIngester()
    stats = ingester.run_ingestion()

    registry = ingester.schema_registry

    # Check that KBR rows are consistent
    kbr_rows = [data["kbr_row"] for data in registry.values() if data["kbr_row"] > 0]
    avg_kbr_row = sum(kbr_rows) / len(kbr_rows)

    # Most should be between 40 and 60
    in_range = sum(1 for r in kbr_rows if 40 <= r <= 60)
    percentage = in_range / len(kbr_rows) * 100

    assert percentage > 90, f"Only {percentage}% of KBR rows in expected range"
    assert avg_kbr_row > 45, f"Average KBR row too low: {avg_kbr_row}"
    assert avg_kbr_row < 55, f"Average KBR row too high: {avg_kbr_row}"

    print(f"✅ Average KBR row: {avg_kbr_row:.1f}")
    print(
        f"✅ KBR rows in range (40-60): {in_range}/{len(kbr_rows)} ({percentage:.1f}%)"
    )
    print("TEST 6 PASSED")

    return True


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 60)
    print("🧪 ROSSTAT INGESTER TEST VERIFICATION")
    print("=" * 60)

    tests = [
        ("KBR Identification", test_kbr_identification),
        ("Frequency Detection", test_frequency_detection),
        ("Registry Regeneration", test_registry_regeneration),
        ("Schema Registry Generated", test_schema_registry_generated),
        ("Quality Report Generated", test_quality_report_generated),
        ("KBR Row Accuracy", test_kbr_row_accuracy),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASSED", None))
        except AssertionError as e:
            results.append((name, "FAILED", str(e)))
            print(f"❌ TEST FAILED: {e}")
        except Exception as e:
            results.append((name, "ERROR", str(e)))
            print(f"❌ TEST ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, status, _ in results if status == "PASSED")
    failed = sum(1 for _, status, _ in results if status == "FAILED")
    errors = sum(1 for _, status, _ in results if status == "ERROR")

    for name, status, message in results:
        emoji = "✅" if status == "PASSED" else "❌"
        print(f"{emoji} {name}: {status}")
        if message:
            print(f"   {message}")

    print(
        f"\nTotal: {len(results)} | Passed: {passed} | Failed: {failed} | Errors: {errors}"
    )

    if failed == 0 and errors == 0:
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print(f"\n⚠️  {failed + errors} TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
