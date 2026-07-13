"""
Схемы для информации о моделях
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class ModelInfo(BaseModel):
    """Информация о модели."""

    name: str = Field(description="Название модели")
    weight: float = Field(description="Вес в ансамбле")
    min_train_size: int = Field(description="Минимум данных для обучения")
    description: str = Field(description="Описание модели")
    mae: Optional[float] = Field(default=None, description="MAE из backtest h=1")


class ModelsListResponse(BaseModel):
    """Список доступных моделей."""

    models: List[ModelInfo] = Field(description="Список моделей")
    total_weight: float = Field(description="Сумма весов (должна быть 1.0)")

    class Config:
        json_schema_extra = {
            "example": {
                "models": [
                    {
                        "name": "ridge",
                        "weight": 0.40,
                        "min_train_size": 36,
                        "description": "Ridge регрессия с ETS",
                    },
                    {
                        "name": "bvar",
                        "weight": 0.20,
                        "min_train_size": 24,
                        "description": "Байесовская VAR",
                    },
                ],
                "total_weight": 1.0,
            }
        }


class HealthResponse(BaseModel):
    """Статус сервера."""

    status: str = Field(default="ok")
    version: str = Field(description="Версия API")
    models_available: int = Field(description="Количество моделей")
    data_loaded: bool = Field(description="Данные загружены")
