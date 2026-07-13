from fastapi import APIRouter, HTTPException, Body, Request
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import json

router = APIRouter(prefix="/forecast/batch", tags=["Batch Forecast"])


MAX_SCENARIOS_PER_REQUEST = 10


class ScenarioInput(BaseModel):
    """Input scenario for forecasting."""

    ki: float = Field(..., description="Key Rate (%)", ge=0, le=100)
    usd: Optional[float] = Field(None, description="USD Exchange Rate (RUB)", ge=0)
    brent: Optional[float] = Field(None, description="Brent Oil Price (USD/bbl)", ge=0)


class SingleForecast(BaseModel):
    """Single forecast result."""

    scenario_index: int
    input_scenario: Dict[str, Optional[float]]
    forecast: List[float]
    horizon: int
    forecast_dates: List[str]


class BatchForecastResponse(BaseModel):
    """Batch forecast response."""

    success: bool
    total_scenarios: int
    forecasts: List[SingleForecast]
    message: Optional[str] = None


_demo_model_params = None


def _get_demo_model_params():
    """Get or initialize demo model parameters."""
    global _demo_model_params
    if _demo_model_params is None:
        _demo_model_params = {
            "base_inflation": 0.4,
            "ki_coeff": 0.015,
            "usd_coeff": 0.002,
            "brent_coeff": 0.001,
            "seasonality": {
                1: 0.8,
                2: 0.6,
                3: 0.4,
                4: 0.3,
                5: 0.2,
                6: 0.1,
                7: 0.15,
                8: 0.25,
                9: 0.35,
                10: 0.5,
                11: 0.7,
                12: 1.0,
            },
        }
    return _demo_model_params


def _generate_scenario_forecast(
    scenario: ScenarioInput, horizon: int
) -> Dict[str, Any]:
    """
    Generate forecast for a single scenario.

    Uses a simplified model based on macro variables:
    - Higher Key Rate -> Higher inflation expectations
    - Higher USD -> Imported inflation
    - Higher Brent -> Energy price pressure
    """
    params = _get_demo_model_params()

    base_rate = params["base_inflation"]

    ki_effect = scenario.ki * params["ki_coeff"]
    usd_effect = (scenario.usd or 90) * params["usd_coeff"]
    brent_effect = (scenario.brent or 75) * params["brent_coeff"]

    monthly_inflation = base_rate + ki_effect + usd_effect + brent_effect

    forecast = []
    current_date = pd.Timestamp.now()

    for h in range(1, horizon + 1):
        forecast_date = current_date + pd.DateOffset(months=h)
        seasonality = params["seasonality"].get(forecast_date.month, 1.0)

        monthly_inflation *= 0.95

        adj_inflation = monthly_inflation * seasonality + np.random.normal(0, 0.05)
        adj_inflation = max(-2.0, min(5.0, adj_inflation))

        forecast.append(round(adj_inflation, 4))

    forecast_dates = [
        (current_date + pd.DateOffset(months=h)).strftime("%Y-%m-%d")
        for h in range(1, horizon + 1)
    ]

    return {
        "scenario_index": 0,
        "input_scenario": {
            "ki": scenario.ki,
            "usd": scenario.usd,
            "brent": scenario.brent,
        },
        "forecast": forecast,
        "horizon": horizon,
        "forecast_dates": forecast_dates,
    }


@router.get("")
@router.get("/")
async def batch_forecast_root():
    """Batch forecast endpoint info."""
    return {
        "message": "Batch Forecast API",
        "endpoint": "/forecast/batch",
        "max_scenarios_per_request": MAX_SCENARIOS_PER_REQUEST,
        "input_format": [
            {
                "ki": "Key Rate (%)",
                "usd": "USD Exchange Rate (optional)",
                "brent": "Brent Oil Price (optional)",
            }
        ],
    }


@router.post("", response_model=BatchForecastResponse)
async def batch_forecast_no_slash(request: Request):
    """Generate forecasts for multiple scenarios (without trailing slash)."""
    return await batch_forecast(request)


@router.post("/", response_model=BatchForecastResponse)
async def batch_forecast(request: Request):
    """
    Generate forecasts for multiple scenarios.

    Each scenario specifies macro variables (Ki, USD, Brent).
    The API returns a forecast for each scenario independently.

    Rate limit: Maximum 10 scenarios per request.
    Accepts JSON data with or without Content-Type header.
    """
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        scenarios_data = await request.json()
    elif "application/x-www-form-urlencoded" in content_type or content_type == "":
        try:
            body = await request.body()
            scenarios_data = json.loads(body.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise HTTPException(
                status_code=400, detail="Invalid JSON format in request body"
            )
    else:
        raise HTTPException(
            status_code=415, detail=f"Unsupported media type: {content_type}"
        )

    if not isinstance(scenarios_data, list):
        raise HTTPException(
            status_code=400, detail="Request body must be a JSON array of scenarios"
        )

    scenarios = [ScenarioInput(**s) for s in scenarios_data]

    horizon = 12

    try:
        if not scenarios:
            raise HTTPException(status_code=400, detail="No scenarios provided")

        if len(scenarios) > MAX_SCENARIOS_PER_REQUEST:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {MAX_SCENARIOS_PER_REQUEST} scenarios allowed per request",
            )

        forecasts = []

        for idx, scenario in enumerate(scenarios):
            try:
                result = _generate_scenario_forecast(scenario, horizon)
                result["scenario_index"] = idx
                forecasts.append(SingleForecast(**result))
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing scenario {idx}: {str(e)}",
                )

        response = BatchForecastResponse(
            success=True,
            total_scenarios=len(forecasts),
            forecasts=forecasts,
            message=f"Generated {len(forecasts)} forecasts successfully",
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing batch forecast: {str(e)}"
        )


@router.get("/limits")
async def get_limits():
    """Get API limits for batch forecasting."""
    return {
        "max_scenarios_per_request": MAX_SCENARIOS_PER_REQUEST,
        "max_horizon_months": 36,
        "rate_limit": "10 requests/minute",
    }
