"""
Test batch forecast endpoint verification.

Task 430: API: Batch Scenario (Part 2 - Verification)
"""

import pytest
import requests
import json


BASE_URL = "http://localhost:8000"
BATCH_ENDPOINT = f"{BASE_URL}/forecast/batch"


def test_batch_endpoint_health():
    """Test that the batch endpoint is accessible."""
    response = requests.get(f"{BATCH_ENDPOINT}/limits")
    assert response.status_code == 200
    data = response.json()
    assert "max_scenarios_per_request" in data


def test_batch_forecast_two_scenarios():
    """
    Verify batch endpoint with 2 scenarios.

    Acceptance criteria:
    - @functional: curl -X POST localhost:8000/forecast/batch returns 200
    - @metric: Response contains list of 2 items
    """
    # Define 2 scenarios with different macro parameters
    scenarios = [
        {"ki": 7.5, "usd": 92.0, "brent": 80.0},
        {"ki": 16.0, "usd": 105.0, "brent": 65.0},
    ]

    # Send POST request
    response = requests.post(
        BATCH_ENDPOINT, json=scenarios, headers={"Content-Type": "application/json"}
    )

    # Criterion 1: Returns 200
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    # Parse response
    data = response.json()

    # Verify response structure
    assert "success" in data
    assert "total_scenarios" in data
    assert "forecasts" in data
    assert data["success"] is True

    # Criterion 2: Response contains list of 2 items
    assert "forecasts" in data
    forecasts = data["forecasts"]
    assert isinstance(forecasts, list), "forecasts should be a list"
    assert len(forecasts) == 2, f"Expected 2 forecasts, got {len(forecasts)}"

    # Verify forecast structure
    for idx, forecast in enumerate(forecasts):
        assert forecast["scenario_index"] == idx
        assert "input_scenario" in forecast
        assert "forecast" in forecast
        assert "horizon" in forecast
        assert "forecast_dates" in forecast

        # Verify input scenario matches what we sent
        assert forecast["input_scenario"]["ki"] == scenarios[idx]["ki"]
        assert forecast["input_scenario"]["usd"] == scenarios[idx]["usd"]
        assert forecast["input_scenario"]["brent"] == scenarios[idx]["brent"]

        # Verify forecast is a list of 12 values (default horizon)
        assert isinstance(forecast["forecast"], list)
        assert len(forecast["forecast"]) == 12
        assert forecast["horizon"] == 12
        assert len(forecast["forecast_dates"]) == 12


def test_batch_forecast_single_scenario():
    """Test batch endpoint with 1 scenario (edge case)."""
    scenarios = [{"ki": 10.0, "usd": 95.0, "brent": 75.0}]

    response = requests.post(
        BATCH_ENDPOINT, json=scenarios, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["forecasts"]) == 1


def test_batch_forecast_optional_fields():
    """Test batch endpoint with optional USD/Brent fields."""
    scenarios = [{"ki": 8.0}, {"ki": 12.0, "usd": 100.0}]

    response = requests.post(
        BATCH_ENDPOINT, json=scenarios, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["forecasts"]) == 2


def test_batch_forecast_max_scenarios():
    """Test batch endpoint with maximum allowed scenarios (10)."""
    scenarios = [{"ki": 10.0, "usd": 95.0, "brent": 75.0}] * 10  # Repeat 10 times

    response = requests.post(
        BATCH_ENDPOINT, json=scenarios, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_scenarios"] == 10
    assert len(data["forecasts"]) == 10


def test_batch_forecast_too_many_scenarios():
    """Test batch endpoint with >10 scenarios (should fail)."""
    scenarios = [{"ki": 10.0}] * 11  # 11 scenarios (exceeds limit)

    response = requests.post(
        BATCH_ENDPOINT, json=scenarios, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "Maximum 10" in data["detail"]


def test_batch_forecast_empty_scenarios():
    """Test batch endpoint with empty scenarios array (should fail)."""
    response = requests.post(
        BATCH_ENDPOINT, json=[], headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_batch_forecast_invalid_json():
    """Test batch endpoint with invalid JSON (should fail)."""
    response = requests.post(
        BATCH_ENDPOINT,
        data="not valid json",
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 400


def test_batch_forecast_via_curl_simulation():
    """
    Simulate curl command (as per acceptance criteria).

    This tests that the equivalent of:
        curl -X POST localhost:8000/forecast/batch -d '[{"ki": 7.5}, {"ki": 16.0}]'
    returns 200.
    """
    scenarios = [{"ki": 7.5}, {"ki": 16.0}]

    response = requests.post(
        BATCH_ENDPOINT,
        data=json.dumps(scenarios),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["forecasts"]) == 2
    assert data["success"] is True


if __name__ == "__main__":
    # Run main verification test manually
    print("Running verification test...")
    test_batch_endpoint_health()
    print("✓ Batch endpoint is accessible")

    test_batch_forecast_two_scenarios()
    print("✓ Batch forecast with 2 scenarios works")

    print("\nAll tests passed!")
