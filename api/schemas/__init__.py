"""
Pydantic схемы для API
"""

from .forecast import ForecastRequest, ForecastResponse, ModelForecast, EnsembleForecast
from .backtest import BacktestRequest, BacktestResponse, BacktestResult
from .models import ModelInfo, ModelsListResponse
from .batch import BatchForecastRequest, BatchForecastResponse

__all__ = [
    "ForecastRequest",
    "ForecastResponse",
    "ModelForecast",
    "EnsembleForecast",
    "BacktestRequest",
    "BacktestResponse",
    "BacktestResult",
    "ModelInfo",
    "ModelsListResponse",
    "BatchForecastRequest",
    "BatchForecastResponse",
]
