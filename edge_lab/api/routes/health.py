"""
Health check endpoint for monitoring API status and uptime.
"""

from fastapi import APIRouter
import time
from typing import Optional

router = APIRouter(tags=["Health"])

# Track application start time
_START_TIME: Optional[float] = None


def set_start_time():
    """Set application start time for uptime calculation."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()


def get_uptime() -> float:
    """Get application uptime in seconds."""
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


@router.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.

    Returns API status, version, uptime, and model count.
    Used by monitoring systems and load balancers to verify service health.
    """
    try:
        # Try to import model registry to get model count
        import sys
        import os

        # Add parent directory to path for imports
        sys.path.insert(
            0,
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
        )

        try:
            from sirena.models import ModelRegistry

            model_count = len(ModelRegistry.list_models())
        except ImportError:
            # Fallback: use edge_lab models if available
            model_count = 0
    except Exception:
        model_count = 0

    return {
        "status": "healthy",
        "version": "1.0.0",
        "uptime": get_uptime(),
        "model_count": model_count,
    }
