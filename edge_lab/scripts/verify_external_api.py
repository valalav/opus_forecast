#!/usr/bin/env python3
"""
API Connectivity Verification Script
Tests connectivity to CBR and MinFin external data sources
"""

import requests
import sys
from typing import Tuple


def test_cbr_api() -> Tuple[bool, str]:
    """Test CBR Data Service API connectivity"""
    try:
        # Test swagger endpoint
        response = requests.get("https://www.cbr.ru/dataservice/swagger", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return True, f"CBR API accessible. OpenAPI spec contains {len(data)} keys"
        else:
            return False, f"CBR API returned status code {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "CBR API request timed out"
    except requests.exceptions.RequestException as e:
        return False, f"CBR API connection error: {e}"
    except Exception as e:
        return False, f"CBR API parsing error: {e}"


def test_cbr_stats_page() -> Tuple[bool, str]:
    """Test CBR statistics data service page"""
    try:
        response = requests.get(
            "https://www.cbr.ru/statistics/data-service/", timeout=10
        )
        if response.status_code == 200:
            return True, "CBR Data Service page accessible (HTML)"
        else:
            return False, f"CBR page returned status code {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "CBR page request timed out"
    except requests.exceptions.RequestException as e:
        return False, f"CBR page connection error: {e}"


def test_minfin_registry() -> Tuple[bool, str]:
    """Test MinFin Open Data Registry"""
    try:
        response = requests.get("https://minfin.gov.ru/opendata/list.csv", timeout=10)
        if response.status_code == 200:
            lines = response.text.strip().split("\n")
            return True, f"MinFin registry accessible. Contains {len(lines)} datasets"
        else:
            return False, f"MinFin registry returned status code {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "MinFin registry request timed out"
    except requests.exceptions.RequestException as e:
        return False, f"MinFin registry connection error: {e}"
    except Exception as e:
        return False, f"MinFin registry parsing error: {e}"


def test_minfin_page() -> Tuple[bool, str]:
    """Test MinFin open data page"""
    try:
        response = requests.get(
            "https://minfin.gov.ru/ru/opendata/registry/", timeout=10
        )
        if response.status_code == 200:
            return True, "MinFin Open Data page accessible (HTML)"
        else:
            return False, f"MinFin page returned status code {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "MinFin page request timed out"
    except requests.exceptions.RequestException as e:
        return False, f"MinFin page connection error: {e}"


def test_github_repo(url: str, name: str) -> Tuple[bool, str]:
    """Test GitHub repo accessibility"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return True, f"{name} repository accessible"
        else:
            return False, f"{name} returned status code {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"{name} connection error: {e}"


def main():
    """Run all connectivity tests"""
    print("=" * 60)
    print("External Data API Connectivity Verification")
    print("=" * 60)
    print()

    results = []
    check = "\u2713"
    cross = "\u2717"

    # Test CBR
    print("Testing CBR (Bank of Russia)...")
    success, msg = test_cbr_api()
    results.append(("CBR API", success, msg))
    print(f"  [{check if success else cross}] {msg}")

    success, msg = test_cbr_stats_page()
    results.append(("CBR Data Service", success, msg))
    print(f"  [{check if success else cross}] {msg}")
    print()

    # Test MinFin
    print("Testing MinFin (Ministry of Finance)...")
    success, msg = test_minfin_registry()
    results.append(("MinFin Registry", success, msg))
    print(f"  [{check if success else cross}] {msg}")

    success, msg = test_minfin_page()
    results.append(("MinFin Page", success, msg))
    print(f"  [{check if success else cross}] {msg}")
    print()

    # Test GitHub repos
    print("Testing GitHub Repositories...")
    success, msg = test_github_repo(
        "https://github.com/alexisakov/seasonal_bankofrussia", "seasonal_bankofrussia"
    )
    results.append(("seasonal_bankofrussia", success, msg))
    print(f"  [{check if success else cross}] {msg}")

    success, msg = test_github_repo(
        "https://github.com/abnegantes/open-russian-data", "open-russian-data"
    )
    results.append(("open-russian-data", success, msg))
    print(f"  [{check if success else cross}] {msg}")
    print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    success_count = sum(1 for _, s, _ in results if s)
    total_count = len(results)

    for name, success, msg in results:
        status = "PASS" if success else "FAIL"
        print(f"  {status:4} | {name:25} | {msg}")

    print()
    print(f"Results: {success_count}/{total_count} tests passed")

    if success_count == total_count:
        print("\nAll connectivity tests PASSED! API integration is viable.")
        return 0
    else:
        print(f"\nWARNING: {total_count - success_count} test(s) FAILED.")
        print("Some APIs may be blocked or have changed endpoints.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
