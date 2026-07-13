#!/usr/bin/env python3
"""
Verification script for /health endpoint.

This script starts the СИРЕНА API, tests the /health endpoint,
and verifies it returns the required fields (status, version).

Usage:
    python3 tests/verify_health.py
"""

import subprocess
import time
import json
import sys
import signal
import requests
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def start_api(port: int = 8001):
    """Start the API server in background."""
    print(f"Starting СИРЕНА API on port {port}...")

    # Start uvicorn in background
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--port",
            str(port),
            "--log-level",
            "error",  # Reduce log noise
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd="/home/valalav/_projects/sirena-kbr",
    )

    # Wait for API to start
    for i in range(30):
        try:
            response = requests.get(f"http://localhost:{port}/", timeout=1)
            if response.status_code == 404:
                # FastAPI returns 404 for root (expected), but server is up
                print(f"API started on port {port} (PID: {process.pid})")
                return process
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            time.sleep(0.5)
    else:
        raise TimeoutError("API failed to start within 30 seconds")


def stop_api(process):
    """Stop the API server."""
    print(f"Stopping API (PID: {process.pid})...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    print("API stopped")


def verify_health_endpoint(port: int = 8001):
    """Verify /health endpoint returns correct response."""
    print(f"\n{'=' * 60}")
    print("HEALTH ENDPOINT VERIFICATION")
    print("=" * 60)

    api_process = None
    try:
        # Start API
        api_process = start_api(port)

        # Wait a bit more for full initialization
        time.sleep(2)

        # Test /health endpoint
        print(f"\nTesting GET /health...")
        response = requests.get(f"http://localhost:{port}/health", timeout=5)

        print(f"Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type')}")

        if response.status_code != 200:
            print(f"\n❌ FAILED: Expected status 200, got {response.status_code}")
            return False

        data = response.json()
        print(f"\nResponse JSON:\n{json.dumps(data, indent=2)}")

        # Verify required fields
        required_fields = ["status", "version"]
        missing_fields = [f for f in required_fields if f not in data]

        if missing_fields:
            print(f"\n❌ FAILED: Missing required fields: {missing_fields}")
            return False

        # Verify field values
        print(f"\n{'=' * 60}")
        print("VERIFICATION RESULTS")
        print("=" * 60)

        checks = []

        # Check 1: status field exists
        has_status = "status" in data
        checks.append(("status field exists", has_status))
        print(f"✓ status field exists: {has_status}")

        # Check 2: version field exists
        has_version = "version" in data
        checks.append(("version field exists", has_version))
        print(f"✓ version field exists: {has_version}")

        # Check 3: status value
        status_value = data.get("status")
        status_ok = status_value in ["ok", "error"]
        checks.append(("status value is valid", status_ok))
        print(
            f"✓ status value is valid ('ok' or 'error'): {status_ok} (value: '{status_value}')"
        )

        # Check 4: version is non-empty string
        version_value = data.get("version")
        version_ok = isinstance(version_value, str) and len(version_value) > 0
        checks.append(("version is non-empty string", version_ok))
        print(f"✓ version is non-empty string: {version_ok} (value: '{version_value}')")

        # Check 5: Additional optional fields
        optional_fields = ["models_available", "data_loaded", "uptime_seconds"]
        for field in optional_fields:
            if field in data:
                print(f"✓ Optional field '{field}': {data[field]}")

        # Summary
        all_passed = all(result for _, result in checks)

        print(f"\n{'=' * 60}")
        if all_passed:
            print("✅ ALL CHECKS PASSED")
            print("=" * 60)
            print("\n📝 Note: Port 8000 is occupied by Whisper service.")
            print(
                "   The /health endpoint is verified on port {port}.".format(port=port)
            )
            print("   When СИРЕНА API runs on port 8000, it will work identically.")
            return True
        else:
            print("❌ SOME CHECKS FAILED")
            print("=" * 60)
            return False

    except Exception as e:
        print(f"\n❌ ERROR during verification: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        if api_process:
            stop_api(api_process)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify /health endpoint")
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to use for testing (default: 8001, since 8000 is occupied)",
    )
    args = parser.parse_args()

    try:
        success = verify_health_endpoint(args.port)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
