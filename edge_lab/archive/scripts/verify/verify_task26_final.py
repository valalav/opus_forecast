#!/usr/bin/env python3
"""Verification script for Task 26: API /explain endpoint"""

import subprocess
import time
import json
import sys
import requests
from pathlib import Path


def check_file_exists():
    """Check if api/routes/explain.py exists with >30 lines"""
    filepath = Path("api/routes/explain.py")
    if not filepath.exists():
        print("❌ FAILED: api/routes/explain.py does not exist")
        return False

    line_count = sum(1 for _ in filepath.open())
    if line_count <= 30:
        print(
            f"❌ FAILED: api/routes/explain.py has only {line_count} lines (need >30)"
        )
        return False

    print(f"✅ PASSED: api/routes/explain.py exists with {line_count} lines")
    return True


def check_endpoint_returns_200():
    """Check if curl localhost:8000/explain returns 200"""
    try:
        response = requests.get("http://localhost:8000/explain", timeout=5)
        if response.status_code == 200:
            print("✅ PASSED: curl localhost:8000/explain returns 200")
            return True
        else:
            print(
                f"❌ FAILED: curl localhost:8000/explain returned {response.status_code}"
            )
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ FAILED: curl localhost:8000/explain failed: {e}")
        return False


def check_json_contains_feature_importance():
    """Check if JSON contains feature importance"""
    try:
        response = requests.post(
            "http://localhost:8000/explain/",
            json={"model_name": "test", "features": [{"f1": 0.5}, {"f2": 0.3}]},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        if "feature_importance" not in data:
            print("❌ FAILED: JSON response does not contain 'feature_importance'")
            return False

        if not isinstance(data["feature_importance"], list):
            print("❌ FAILED: 'feature_importance' is not a list")
            return False

        if len(data["feature_importance"]) == 0:
            print("❌ FAILED: 'feature_importance' list is empty")
            return False

        for feature in data["feature_importance"]:
            if "feature_name" not in feature or "importance" not in feature:
                print("❌ FAILED: Feature item missing 'feature_name' or 'importance'")
                return False

        print(
            f"✅ PASSED: JSON contains feature importance ({len(data['feature_importance'])} features)"
        )
        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ FAILED: POST request failed: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ FAILED: JSON decode error: {e}")
        return False


def main():
    """Run all verification checks"""
    print("=" * 60)
    print("Task 26 Verification: API /explain endpoint")
    print("=" * 60)

    results = []

    results.append(check_file_exists())
    results.append(check_endpoint_returns_200())
    results.append(check_json_contains_feature_importance())

    print("=" * 60)
    if all(results):
        print("✅ ALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("❌ SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
