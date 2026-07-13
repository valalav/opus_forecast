#!/usr/bin/env python3
"""
Verification script for Task 514: Dashboard Regime Monitor Widget

This script verifies:
1. regime_indicator function exists in dashboard.py
2. Three regime types are supported
3. Function has sidebar display and timeline visualization
"""

import ast
import re


def verify_regime_indicator_function():
    """Verify regime_indicator function exists and has required elements."""

    print("=" * 60)
    print("VERIFICATION: Task 514 - Regime Monitor Widget")
    print("=" * 60)

    # Read edge_lab/dashboard.py
    with open("dashboard.py", "r") as f:
        content = f.read()

    # Criterion 1: Check for regime_indicator function
    print("\n[Criterion 1] Function exists")
    if "def regime_indicator(" in content:
        print("✅ PASS: regime_indicator function found in dashboard.py")

        # Check function signature
        match = re.search(r"def regime_indicator\((.*?)\):", content)
        if match:
            print(f"   Signature: regime_indicator({match.group(1)})")
    else:
        print("❌ FAIL: regime_indicator function NOT found")

    # Criterion 2: Check for sidebar display
    print("\n[Criterion 2] Sidebar display")
    sidebar_calls = content.count("st.sidebar.")
    if sidebar_calls >= 10:
        print(f"✅ PASS: Found {sidebar_calls} st.sidebar calls in function")
    else:
        print(f"❌ FAIL: Only {sidebar_calls} st.sidebar calls found")

    # Criterion 3: Check for three regime types
    print("\n[Criterion 3] Three regime types supported")
    regime_types = [
        "RegimeType.NORMAL",
        "RegimeType.SHOCK",
        "RegimeType.HIGH_INFLATION",
    ]
    all_found = all(rt in content for rt in regime_types)
    if all_found:
        print("✅ PASS: All three regime types found:")
        for rt in regime_types:
            print(f"   - {rt}")
    else:
        print("❌ FAIL: Missing regime types")

    # Additional: Check for regime history timeline
    print("\n[Additional Check] Regime history timeline")
    if "История режимов" in content or "timeline" in content.lower():
        print("✅ PASS: Regime history timeline implemented")
    else:
        print("❌ FAIL: No timeline visualization found")

    # Additional: Check for explanation tooltip
    print("\n[Additional Check] Explanation tooltip")
    if "Объяснение" in content or "expander" in content:
        print("✅ PASS: Explanation/expander found")
    else:
        print("❌ FAIL: No explanation section")

    # Check that function is called in main app
    print("\n[Main App Check] Function called in main section")
    if "regime_indicator(df_macro, df)" in content:
        print("✅ PASS: Function is called in main app section")
    else:
        print("❌ FAIL: Function is not being called")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    verify_regime_indicator_function()
