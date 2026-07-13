"""
API Routes
"""

from .forecast import router as forecast_router
from .models import router as models_router
from .backtest import router as backtest_router
from .weekly import router as weekly_router
from .batch import router as batch_router
from .health import router as health_router
from .metrics import router as metrics_router

__all__ = [
    "forecast_router",
    "models_router",
    "backtest_router",
    "weekly_router",
    "batch_router",
    "health_router",
    "metrics_router",
]
