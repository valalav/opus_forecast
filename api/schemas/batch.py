"""
Схемы для пакетного прогнозирования
"""

from pydantic import BaseModel, Field
from typing import List, Optional

from .forecast import ForecastRequest, ForecastResponse


class BatchForecastRequest(BaseModel):
    """Запрос на пакетное прогнозирование."""

    scenarios: List[ForecastRequest] = Field(
        description="Список сценариев прогнозирования", min_length=1, max_length=10
    )

    class Config:
        json_schema_extra = {
            "example": {
                "scenarios": [
                    {
                        "horizon": 12,
                        "models": ["ridge", "bvar"],
                        "weights": {"ridge": 0.5, "bvar": 0.5},
                        "include_intervals": False,
                    },
                    {
                        "horizon": 6,
                        "models": ["lightgbm", "prophet"],
                        "weights": {"lightgbm": 0.7, "prophet": 0.3},
                        "include_intervals": True,
                    },
                ]
            }
        }


class BatchForecastResponse(BaseModel):
    """Ответ с пакетными прогнозами."""

    results: List[ForecastResponse] = Field(description="Список прогнозов по сценариям")
    count: int = Field(description="Количество обработанных сценариев")
    version: str = Field(default="4.0", description="Версия API")

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "ensemble": {
                            "values": [0.5, 0.4],
                            "dates": ["2025-01", "2025-02"],
                        },
                        "models": {
                            "ridge": {
                                "model": "ridge",
                                "values": [0.5, 0.4],
                                "dates": ["2025-01", "2025-02"],
                                "weight": 0.5,
                            }
                        },
                        "data_date": "2024-12",
                        "version": "4.0",
                    }
                ],
                "count": 1,
                "version": "4.0",
            }
        }
