"""
Эндпоинты для прогнозирования
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from ..schemas.forecast import (
    ForecastRequest,
    ForecastResponse,
    ModelForecast,
    EnsembleForecast
)

router = APIRouter(prefix="/forecast", tags=["Forecast"])


def get_data():
    """Загрузка данных."""
    import os
    data_paths = [
        'data/infl_kbr.csv',
        '/home/valalav/_projects/sirena-kbr/data/infl_kbr.csv'
    ]

    for path in data_paths:
        if os.path.exists(path):
            df = pd.read_csv(path, sep=';', decimal=',')
            df['Date'] = pd.to_datetime(df['Day'], format='%d.%m.%Y')

            if 'Товар' in df.columns:
                pivot = df.pivot_table(
                    index='Date',
                    columns='Товар',
                    values='MoM',
                    aggfunc='first'
                )
            else:
                pivot = df.set_index('Date')

            return pivot.sort_index()

    raise HTTPException(status_code=500, detail="Данные не найдены")


@router.post("", response_model=ForecastResponse)
async def create_forecast(request: ForecastRequest):
    """
    Создать прогноз инфляции.

    Поддерживает:
    - Выбор моделей
    - Кастомные веса
    - Доверительные интервалы
    """
    try:
        from sirena.models import ModelRegistry

        # Загружаем данные
        df = get_data()
        last_date = df.dropna(subset=['Все товары и услуги']).index.max()

        # Определяем модели
        if request.models:
            models_to_use = [m for m in request.models if ModelRegistry.is_registered(m)]
        else:
            models_to_use = ModelRegistry.list_models()

        if not models_to_use:
            raise HTTPException(status_code=400, detail="Нет доступных моделей")

        # Веса
        if request.weights:
            weights = request.weights
        else:
            weights = {m: ModelRegistry.get_default_weight(m) for m in models_to_use}

        # Нормализация весов
        total_weight = sum(weights.get(m, 0) for m in models_to_use)
        if total_weight > 0:
            weights = {m: weights.get(m, 0) / total_weight for m in models_to_use}

        # Даты прогноза
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=request.horizon,
            freq='MS'
        )
        date_strings = [d.strftime('%Y-%m') for d in forecast_dates]

        # Прогнозы по моделям
        model_forecasts: Dict[str, ModelForecast] = {}
        ensemble_values = np.zeros(request.horizon)

        for model_name in models_to_use:
            try:
                model = ModelRegistry.get(model_name)
                model.fit(df)
                fc = model.forecast(horizon=request.horizon)

                # Конвертируем если нужно
                if hasattr(fc, 'values'):
                    fc = fc.values

                # Нормализуем формат
                fc = np.array(fc).flatten()
                if len(fc) < request.horizon:
                    fc = np.pad(fc, (0, request.horizon - len(fc)), mode='edge')
                elif len(fc) > request.horizon:
                    fc = fc[:request.horizon]

                model_forecasts[model_name] = ModelForecast(
                    model=model_name,
                    values=[round(float(v), 3) for v in fc],
                    dates=date_strings,
                    weight=weights.get(model_name, 0)
                )

                # Добавляем к ансамблю
                ensemble_values += fc * weights.get(model_name, 0)

            except Exception as e:
                # Логируем, но продолжаем
                print(f"Ошибка модели {model_name}: {e}")
                continue

        if not model_forecasts:
            raise HTTPException(status_code=500, detail="Все модели дали ошибку")

        # Ансамбль
        ensemble = EnsembleForecast(
            values=[round(float(v), 3) for v in ensemble_values],
            dates=date_strings
        )

        return ForecastResponse(
            ensemble=ensemble,
            models=model_forecasts,
            data_date=last_date.strftime('%Y-%m'),
            version="4.0"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/quick")
async def quick_forecast(horizon: int = 3):
    """
    Быстрый прогноз только Ridge моделью.

    Для максимальной скорости использует только основную модель.
    """
    try:
        from sirena.models import ModelRegistry

        df = get_data()
        last_date = df.dropna(subset=['Все товары и услуги']).index.max()

        model = ModelRegistry.get("ridge")
        model.fit(df)
        fc = model.forecast(horizon=horizon)

        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq='MS'
        )

        return {
            "values": [round(float(v), 3) for v in fc],
            "dates": [d.strftime('%Y-%m') for d in forecast_dates],
            "model": "ridge",
            "data_date": last_date.strftime('%Y-%m')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
