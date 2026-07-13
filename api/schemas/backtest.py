"""
Схемы для бэктестирования
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import date


class BacktestRequest(BaseModel):
    """Запрос на бэктест."""

    model: str = Field(description="Название модели")
    start_date: str = Field(
        default="2019-01-01", description="Начало периода (YYYY-MM-DD)"
    )
    end_date: Optional[str] = Field(default=None, description="Конец периода")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "ridge",
                "start_date": "2019-01-01",
                "end_date": "2024-12-01",
            }
        }


class BacktestResult(BaseModel):
    """Результат одной итерации бэктеста."""

    date: str
    actual: float
    prediction: float
    error: float


class BacktestMetrics(BaseModel):
    """Метрики бэктеста."""

    MAE: float = Field(description="Mean Absolute Error")
    RMSE: float = Field(description="Root Mean Squared Error")
    KPI: float = Field(description="% прогнозов с ошибкой <= 0.5")
    count: int = Field(description="Количество наблюдений")


class BacktestResponse(BaseModel):
    """Ответ с результатами бэктеста."""

    model: str = Field(description="Название модели")
    start_date: str = Field(description="Начало периода")
    end_date: str = Field(description="Конец периода")
    metrics: BacktestMetrics = Field(description="Метрики качества")
    results: List[BacktestResult] = Field(description="Детальные результаты")

    class Config:
        json_schema_extra = {
            "example": {
                "model": "ridge",
                "start_date": "2019-01-01",
                "end_date": "2024-12-01",
                "metrics": {"MAE": 0.25, "RMSE": 0.32, "KPI": 85.0, "count": 72},
                "results": [
                    {"date": "2019-01", "actual": 0.5, "prediction": 0.4, "error": 0.1}
                ],
            }
        }


class HistoryRequest(BaseModel):
    """Запрос исторических прогнозов."""

    start_date: Optional[str] = Field(
        default=None, description="Начало периода (YYYY-MM-DD)"
    )
    end_date: Optional[str] = Field(
        default=None, description="Конец периода (YYYY-MM-DD)"
    )
    model: Optional[str] = Field(default=None, description="Фильтр по названию модели")
    horizon: int = Field(default=1, description="Горизонт прогноза (1, 2, или 12)")

    class Config:
        json_schema_extra = {
            "example": {
                "start_date": "2025-01-01",
                "end_date": "2025-12-01",
                "model": "Ridge",
                "horizon": 1,
            }
        }


class HistoryEntry(BaseModel):
    """Одна запись исторического прогноза."""

    date: str = Field(description="Дата прогноза (YYYY-MM-DD)")
    actual: float = Field(description="Фактическое значение")
    prediction: float = Field(description="Прогноз модели")
    error: float = Field(description="Ошибка (actual - prediction)")


class HistoryResponse(BaseModel):
    """Ответ с историческими прогнозами."""

    count: int = Field(description="Количество записей")
    horizon: int = Field(description="Горизонт прогноза")
    model: Optional[str] = Field(description="Фильтр модели (если был указан)")
    start_date: str = Field(description="Начало периода")
    end_date: str = Field(description="Конец периода")
    data: List[HistoryEntry] = Field(description="Список записей")
