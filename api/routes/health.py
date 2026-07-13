"""
Health check endpoint for monitoring API status and uptime.
"""

from fastapi import APIRouter
import time
import os
from typing import Optional

from ..schemas.models import HealthResponse

router = APIRouter(tags=["Health"])

# Track application start time
_START_TIME: Optional[float] = None


def set_start_time():
    """Set the application start time for uptime calculation."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def get_uptime() -> float:
    """Get application uptime in seconds."""
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring.

    Returns API status, version, uptime, and model count.
    Used by monitoring systems and load balancers to verify service health.
    """
    try:
        from sirena.models import ModelRegistry

        # Get model count
        models_count = len(ModelRegistry.list_models())

        # Check if data files exist
        data_paths = [
            "data/infl_kbr.csv",
            "/home/valalav/_projects/sirena-kbr/data/infl_kbr.csv",
        ]
        data_loaded = any(os.path.exists(path) for path in data_paths)

        return HealthResponse(
            status="ok",
            version="4.0.0",
            models_available=models_count,
            data_loaded=data_loaded,
        )

    except Exception as e:
        # Return error status even if we can't get model info
        return HealthResponse(
            status="error",
            version="4.0.0",
            models_available=0,
            data_loaded=False,
        )


@router.get("/health/detailed", tags=["Health"])
async def health_check_detailed():
    """
    Detailed health check with uptime information.

    Extended health check that includes uptime metrics for monitoring dashboards.
    """
    try:
        from sirena.models import ModelRegistry

        models_count = len(ModelRegistry.list_models())
        uptime_seconds = get_uptime()

        # Format uptime nicely
        uptime_hours = uptime_seconds / 3600
        uptime_days = uptime_hours / 24

        data_paths = [
            "data/infl_kbr.csv",
            "/home/valalav/_projects/sirena-kbr/data/infl_kbr.csv",
        ]
        data_loaded = any(os.path.exists(path) for path in data_paths)

        return {
            "status": "ok",
            "version": "4.0.0",
            "uptime_seconds": round(uptime_seconds, 2),
            "uptime_hours": round(uptime_hours, 2),
            "uptime_days": round(uptime_days, 2),
            "models_available": models_count,
            "data_loaded": data_loaded,
        }

    except Exception as e:
        return {
            "status": "error",
            "version": "4.0.0",
            "uptime_seconds": 0.0,
            "models_available": 0,
            "data_loaded": False,
            "error": str(e),
        }
