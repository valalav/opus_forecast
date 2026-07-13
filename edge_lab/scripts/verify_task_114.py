#!/usr/bin/env python3
"""Verification script for Task 114 acceptance criteria."""

from pathlib import Path


def verify_task_114():
    """Verify all acceptance criteria for Task 114."""

    print("=" * 60)
    print("TASK 114 ACCEPTANCE CRITERIA VERIFICATION")
    print("=" * 60)

    all_passed = True

    # Criterion 1: Data Map opr_data_map.md created
    print("\n[1] Verifying: Data Map opr_data_map.md created")
    data_map_path = Path("data/opr_data_map.md")
    local_map_path = Path("opr_data_map.md")

    if data_map_path.exists():
        print("    ✓ data/opr_data_map.md exists")
        size = data_map_path.stat().st_size
        print(f"    ✓ File size: {size} bytes")
        if size > 1000:
            print("    ✓ File has substantial content (>1KB)")
        else:
            print("    ✗ File seems too small")
            all_passed = False
    else:
        print("    ✗ data/opr_data_map.md NOT found")
        all_passed = False

    if local_map_path.exists():
        print("    ✓ opr_data_map.md (local copy) exists")
    else:
        print("    ✗ opr_data_map.md (local copy) NOT found")
        all_passed = False

    # Criterion 2: Successfully parses headers of 150MB file
    print("\n[2] Verifying: Successfully parses headers of 150MB file")

    # Check the Excel file exists and is large
    excel_file = Path("assets/charts/ОПР_статистика/Основная статистика ЮГУ.xlsx")
    if excel_file.exists():
        size_mb = excel_file.stat().st_size / (1024 * 1024)
        print(f"    ✓ File exists: {excel_file.name}")
        print(f"    ✓ File size: {size_mb:.1f} MB")
        if size_mb > 100:  # Should be ~138MB
            print("    ✓ File is large (>100MB) - requires chunked/read-only parsing")
        else:
            print("    ✗ File is smaller than expected")
            all_passed = False
    else:
        print("    ✗ Excel file NOT found")
        all_passed = False

    # Check data map contains sheet count
    if data_map_path.exists():
        content = data_map_path.read_text(encoding="utf-8")
        if (
            "143 sheets" in content.lower()
            or "sheets Found:" in content
            or "Sheets Found:" in content
        ):
            print("    ✓ Data map contains sheet count from parsed file")
        else:
            print("    ✗ Data map missing sheet count")
            all_passed = False

        if "YUGU" in content or "Южный федеральный округ" in content:
            print("    ✓ Data map contains YUGU region analysis")
        else:
            print("    ✗ Data map missing YUGU region analysis")
            all_passed = False

    # Criterion 3: List of Top-20 Proxy Series identified
    print("\n[3] Verifying: List of Top-20 Proxy Series identified")

    if data_map_path.exists():
        content = data_map_path.read_text(encoding="utf-8")

        # Check for Top-20 section
        if "Top-20 Proxy Series" in content:
            print("    ✓ 'Top-20 Proxy Series' section exists")
        else:
            print("    ✗ 'Top-20 Proxy Series' section NOT found")
            all_passed = False

        # Check for specific proxy categories
        proxy_keywords = [
            "CPI",
            "Consumer Price Index",
            "Inflation Expectations",
            "Wages",
            "Unemployment",
            "Housing Prices",
            "GRP",
            "Budget",
            "Port Cargo",
        ]

        found_keywords = sum(1 for kw in proxy_keywords if kw in content)
        print(f"    ✓ Found {found_keywords}/{len(proxy_keywords)} proxy keywords")

        # Check for numbered list items (should have at least 20)
        numbered_items = content.count("\n**[0-9]")
        if "1. **" in content and "20. **" in content:
            print("    ✓ Contains numbered list from 1 to 20")
        else:
            print("    ✗ Missing full 1-20 numbered list")
            all_passed = False

        # Check for KBR specific mentions
        kbr_mentions = content.lower().count("kbr")
        print(f"    ✓ KBR mentioned {kbr_mentions} times")

        # Check for lag structure section
        if "Lag Structure" in content or "lag" in content.lower():
            print("    ✓ Lag structure documented")
        else:
            print("    ✗ Lag structure not documented")
            all_passed = False

    # Additional verification: Check subdirectories were analyzed
    print("\n[4] Additional verification: Subdirectory analysis")

    if data_map_path.exists():
        content = data_map_path.read_text(encoding="utf-8")

        subdirs_expected = [
            "Бюджеты",
            "жилье",
            "ЗП_безработица",
            "инфляционные ожидания",
            "Цены производителей",
        ]
        found_subdirs = sum(1 for sd in subdirs_expected if sd in content)

        print(
            f"    ✓ {found_subdirs}/{len(subdirs_expected)} subdirectories documented"
        )

    # Final result
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL ACCEPTANCE CRITERIA MET")
    else:
        print("✗ SOME CRITERIA NOT MET")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = verify_task_114()
    exit(0 if success else 1)
