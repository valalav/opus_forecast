"""
Test batch forecast endpoint
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_batch_endpoint_returns_200():
    """Verify batch endpoint returns 200 with 2 scenarios."""
    request_data = {
        "scenarios": [
            {"horizon": 3, "models": ["ridge"], "weights": {"ridge": 1.0}},
            {"horizon": 6, "models": ["ridge"], "weights": {"ridge": 1.0}},
        ]
    }

    response = client.post("/forecast/batch", json=request_data)

    # Criterion 1: curl returns 200
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    data = response.json()

    # Criterion 2: Response contains list of 2 items
    assert "results" in data, "Response missing 'results' key"
    assert isinstance(data["results"], list), "Results is not a list"
    assert len(data["results"]) == 2, f"Expected 2 results, got {len(data['results'])}"
    assert data.get("count") == 2, f"Expected count=2, got {data.get('count')}"

    # Verify each result has required fields
    for result in data["results"]:
        assert "ensemble" in result, "Result missing 'ensemble'"
        assert "models" in result, "Result missing 'models'"
        assert "data_date" in result, "Result missing 'data_date'"


def test_batch_endpoint_with_no_models():
    """Test batch endpoint with empty models list (should use defaults)."""
    request_data = {"scenarios": [{"horizon": 3}, {"horizon": 6}]}

    response = client.post("/forecast/batch", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2


def test_batch_endpoint_empty_scenarios():
    """Test batch endpoint with empty scenarios list."""
    request_data = {"scenarios": []}

    response = client.post("/forecast/batch", json=request_data)

    # Should return error for empty scenarios (422 is FastAPI validation error)
    assert response.status_code in [400, 422, 500], (
        f"Expected error for empty scenarios, got {response.status_code}"
    )


if __name__ == "__main__":
    # Run directly for quick verification
    import sys

    test_batch_endpoint_returns_200()
    print("✓ Test 1 passed: Returns 200 with 2 scenarios")

    test_batch_endpoint_with_no_models()
    print("✓ Test 2 passed: Works with default models")

    test_batch_endpoint_empty_scenarios()
    print("✓ Test 3 passed: Handles empty scenarios")

    print("\n✅ All tests passed!")
