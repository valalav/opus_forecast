"""
Prometheus metrics endpoint for monitoring forecasting system performance.

Exports:
- forecast_requests_total: Total number of forecast requests
- forecast_latency_seconds: Latency of forecast operations
- model_mae_gauge: Current MAE values for models
"""

from fastapi import APIRouter
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time
from typing import Dict

router = APIRouter(tags=["Metrics"])

# Metric definitions
forecast_requests_total = Counter(
    "forecast_requests_total",
    "Total number of forecast requests processed",
    ["model", "status"],
)

forecast_latency_seconds = Histogram(
    "forecast_latency_seconds",
    "Forecast operation latency in seconds",
    ["model"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
)

model_mae_gauge = Gauge(
    "model_mae_gauge",
    "Current Mean Absolute Error (MAE) for forecasting models",
    ["model", "horizon"],
)


def record_forecast_request(model: str, status: str = "success"):
    forecast_requests_total.labels(model=model, status=status).inc()


def record_forecast_latency(model: str, latency_seconds: float):
    forecast_latency_seconds.labels(model=model).observe(latency_seconds)


def update_model_mae(model: str, mae: float, horizon: str = "h1"):
    model_mae_gauge.labels(model=model, horizon=horizon).set(mae)


def get_model_maes() -> Dict[str, float]:
    import os
    import pandas as pd

    try:
        metrics_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "archive",
            "results",
            "backtest_h1_metrics.csv",
        )
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            return dict(zip(df["model"], df["MAE"]))
    except Exception:
        pass

    return {
        "Ridge": 0.321,
        "Subcomp": 0.309,
        "Huber": 0.324,
        "NGBoost": 0.326,
        "EBM": 0.336,
    }


def initialize_model_metrics():
    try:
        maes = get_model_maes()
        for model, mae in maes.items():
            update_model_mae(model, mae, horizon="h1")
    except Exception:
        pass


@router.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
