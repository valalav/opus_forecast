#!/usr/bin/env python3
"""
Batch API Endpoint Verification Script

Task 430 / Task 568: Verify batch API endpoint works correctly.

Usage:
    python3 tests/verify_batch.py

This script:
1. Starts the edge_lab API server (if not running)
2. Checks if batch endpoint is accessible
3. Sends POST request with real data
4. Verifies response code is 200
5. Validates response JSON structure and types
"""

import requests
import json
import sys
import subprocess
import time
import signal
import os


# Edge lab API runs on port 8001 to avoid conflict with main API
EDGE_LAB_PORT = 8001
BASE_URL = f"http://localhost:{EDGE_LAB_PORT}"
BATCH_ENDPOINT = f"{BASE_URL}/forecast/batch"

# Global process reference for cleanup
api_process = None


def start_edge_lab_api():
    """Start the edge_lab API server."""
    global api_process

    print("Starting Edge Lab API server on port 8001...")

    try:
        api_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "api.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(EDGE_LAB_PORT),
            ],
            cwd="/home/valalav/_projects/sirena-kbr/edge_lab",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

        # Wait for server to start
        for i in range(10):
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=1)
                if response.status_code == 200:
                    print_success(f"Edge Lab API started on port {EDGE_LAB_PORT}")
                    return True
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                time.sleep(0.5)

        # If we get here, server didn't start
        print_error("Failed to start Edge Lab API within timeout")
        return False

    except Exception as e:
        print_error(f"Failed to start Edge Lab API: {e}")
        return False


def stop_edge_lab_api():
    """Stop the edge_lab API server."""
    global api_process
    if api_process:
        print("\nStopping Edge Lab API server...")
        try:
            os.killpg(os.getpgid(api_process.pid), signal.SIGTERM)
            api_process.wait(timeout=5)
        except:
            pass
        print_success("Edge Lab API stopped")


def print_success(msg: str):
    """Print success message."""
    print(f"✓ {msg}")


def print_error(msg: str):
    """Print error message and exit."""
    print(f"✗ {msg}")
    sys.exit(1)


def check_endpoint_health():
    """Check if batch endpoint is accessible."""
    try:
        response = requests.get(f"{BATCH_ENDPOINT}/limits", timeout=5)
        if response.status_code != 200:
            print_error(f"Endpoint health check failed: {response.status_code}")
        data = response.json()
        if "max_scenarios_per_request" not in data:
            print_error("Response missing max_scenarios_per_request field")
        print_success("Batch endpoint is accessible")
        return True
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to API. Is the server running on port 8000?")
    except requests.exceptions.Timeout:
        print_error("Request timed out")
    except json.JSONDecodeError:
        print_error("Invalid JSON response from health check")
    return False


def send_batch_request():
    """Send POST request with real data."""
    scenarios = [
        {"ki": 7.5, "usd": 92.0, "brent": 80.0},
        {"ki": 16.0, "usd": 105.0, "brent": 65.0},
    ]

    try:
        response = requests.post(
            BATCH_ENDPOINT,
            json=scenarios,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        return response
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to API. Is the server running on port 8000?")
    except requests.exceptions.Timeout:
        print_error("Request timed out")
    return None


def verify_response_status(response):
    """Verify response code is 200."""
    if response is None:
        return False
    if response.status_code != 200:
        print_error(f"Expected status code 200, got {response.status_code}")
    print_success("Response status code is 200")
    return True


def verify_json_structure(data):
    """Verify response JSON structure and types."""
    required_fields = ["success", "total_scenarios", "forecasts"]

    for field in required_fields:
        if field not in data:
            print_error(f"Response JSON missing required field: {field}")

    if not isinstance(data["success"], bool):
        print_error(f"'success' field must be boolean, got {type(data['success'])}")
    if data["success"] is not True:
        print_error(f"'success' field must be True, got {data['success']}")

    if not isinstance(data["total_scenarios"], int):
        print_error(
            f"'total_scenarios' must be int, got {type(data['total_scenarios'])}"
        )
    if data["total_scenarios"] != 2:
        print_error(f"Expected 2 scenarios, got {data['total_scenarios']}")

    if not isinstance(data["forecasts"], list):
        print_error(f"'forecasts' must be list, got {type(data['forecasts'])}")
    if len(data["forecasts"]) != 2:
        print_error(f"Expected 2 forecasts, got {len(data['forecasts'])}")

    print_success("Response JSON structure is valid")
    return True


def verify_forecast_content(forecasts):
    """Verify forecast content structure."""
    for idx, forecast in enumerate(forecasts):
        required_forecast_fields = [
            "scenario_index",
            "input_scenario",
            "forecast",
            "horizon",
            "forecast_dates",
        ]

        for field in required_forecast_fields:
            if field not in forecast:
                print_error(f"Forecast {idx} missing field: {field}")

        if forecast["scenario_index"] != idx:
            print_error(
                f"Forecast {idx} has wrong scenario_index: {forecast['scenario_index']}"
            )

        input_scen = forecast["input_scenario"]
        if not isinstance(input_scen, dict):
            print_error(f"Forecast {idx} input_scenario must be dict")

        if "forecast" not in forecast or not isinstance(forecast["forecast"], list):
            print_error(f"Forecast {idx} forecast field must be list")

        if len(forecast["forecast"]) != 12:
            print_error(
                f"Forecast {idx} must have 12 months, got {len(forecast['forecast'])}"
            )

        if forecast["horizon"] != 12:
            print_error(f"Forecast {idx} horizon must be 12, got {forecast['horizon']}")

        if len(forecast["forecast_dates"]) != 12:
            print_error(
                f"Forecast {idx} must have 12 dates, got {len(forecast['forecast_dates'])}"
            )

    print_success("Forecast content structure is valid")
    return True


def main():
    """Main verification routine."""
    print("=" * 60)
    print("Batch API Endpoint Verification")
    print("=" * 60)
    print()

    # Start the edge_lab API
    if not start_edge_lab_api():
        print_error("Failed to start Edge Lab API")

    try:
        print("Step 1: Checking endpoint health...")
        if not check_endpoint_health():
            print_error("Endpoint health check failed")

        print("\nStep 2: Sending batch request with 2 scenarios...")
        response = send_batch_request()
        if response is None:
            print_error("Failed to send batch request")

        print("\nStep 3: Verifying response status code...")
        if not verify_response_status(response):
            print_error("Response status verification failed")

        print("\nStep 4: Parsing response JSON...")
        try:
            data = response.json()
            print_success("Response JSON parsed successfully")
        except json.JSONDecodeError as e:
            print_error(f"Failed to parse response JSON: {e}")

        print("\nStep 5: Verifying response JSON structure...")
        if not verify_json_structure(data):
            print_error("Response JSON structure verification failed")

        print("\nStep 6: Verifying forecast content...")
        if not verify_forecast_content(data["forecasts"]):
            print_error("Forecast content verification failed")

        print()
        print("=" * 60)
        print("PASS: All verifications passed!")
        print("=" * 60)
        print()
        print("PASS")
        sys.exit(0)

    finally:
        # Always stop the API server
        stop_edge_lab_api()


if __name__ == "__main__":
    main()
