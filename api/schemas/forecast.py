"""
Схемы для прогнозирования
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import date


class ForecastRequest(BaseModel):
    """Запрос на прогноз."""
    horizon: int = Field(default=12, ge=1, le=24, description="Горизонт прогноза в месяцах")
    models: Optional[List[str]] = Field(default=None, description="Список моделей (если None - все)")
    weights: Optional[Dict[str, float]] = Field(default=None, description="Веса моделей")
    include_intervals: bool = Field(default=False, description="Включить доверительные интервалы")

    class Config:
        json_schema_extra = {
            "example": {
                "horizon": 12,
                "models": ["ridge", "bvar", "lightgbm"],
                "weights": {"ridge": 0.5, "bvar": 0.3, "lightgbm": 0.2},
                "include_intervals": True
            }
        }


class ModelForecast(BaseModel):
    """Прогноз одной модели."""
    model: str = Field(description="Название модели")
    values: List[float] = Field(description="Прогнозные значения (MoM %)")
    dates: List[str] = Field(description="Даты прогноза")
    lower: Optional[List[float]] = Field(default=None, description="Нижняя граница 95% CI")
    upper: Optional[List[float]] = Field(default=None, description="Верхняя граница 95% CI")
    weight: float = Field(description="Вес модели в ансамбле")


class EnsembleForecast(BaseModel):
    """Ансамблевый прогноз."""
    values: List[float] = Field(description="Взвешенный прогноз")
    dates: List[str] = Field(description="Даты прогноза")
    lower: Optional[List[float]] = Field(default=None, description="Нижняя граница")
    upper: Optional[List[float]] = Field(default=None, description="Верхняя граница")


class ForecastResponse(BaseModel):
    """Ответ с прогнозом."""
    ensemble: EnsembleForecast = Field(description="Ансамблевый прогноз")
    models: Dict[str, ModelForecast] = Field(description="Прогнозы по моделям")
    data_date: str = Field(description="Дата последних данных")
    version: str = Field(default="4.0", description="Версия API")

    class Config:
        json_schema_extra = {
            "example": {
                "ensemble": {
                    "values": [0.5, 0.4, 0.6],
                    "dates": ["2025-01", "2025-02", "2025-03"]
                },
                "models": {
                    "ridge": {
                        "model": "ridge",
                        "values": [0.5, 0.4, 0.6],
                        "dates": ["2025-01", "2025-02", "2025-03"],
                        "weight": 0.4
                    }
                },
                "data_date": "2024-12",
                "version": "4.0"
            }
        }
