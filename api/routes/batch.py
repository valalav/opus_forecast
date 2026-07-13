"""
Пакетный эндпоинт для прогнозирования
"""

from fastapi import APIRouter, HTTPException
from typing import List
import pandas as pd
import numpy as np
from datetime import datetime

from ..schemas.batch import BatchForecastRequest, BatchForecastResponse
from ..schemas.forecast import ForecastResponse

router = APIRouter(prefix="/forecast", tags=["Batch"])


def get_data():
    """Загрузка данных."""
    import os

    data_paths = ["data/infl_kbr.csv", "/home/valalav/_projects/sirena-kbr/data/infl_kbr.csv"]

    for path in data_paths:
        if os.path.exists(path):
            df = pd.read_csv(path, sep=";", decimal=",")
            df["Date"] = pd.to_datetime(df["Day"], format="%d.%m.%Y")

            if "Товар" in df.columns:
                pivot = df.pivot_table(
                    index="Date", columns="Товар", values="MoM", aggfunc="first"
                )
                # Convert to numeric to fix dtype issues
                pivot = pivot.apply(pd.to_numeric, errors="coerce")
            else:
                pivot = df.set_index("Date")
                pivot = pivot.apply(pd.to_numeric, errors="coerce")

            return pivot.sort_index()

    raise HTTPException(status_code=500, detail="Данные не найдены")


def create_forecast(request, df, last_date):
    """Создать прогноз для одного сценария."""
    from sirena.models import ModelRegistry

    # Определяем модели
    if request.models:
        models_to_use = [m for m in request.models if ModelRegistry.is_registered(m)]
    else:
        models_to_use = ModelRegistry.list_models()

    if not models_to_use:
        raise ValueError("Нет доступных моделей")

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
        start=last_date + pd.DateOffset(months=1), periods=request.horizon, freq="MS"
    )
    date_strings = [d.strftime("%Y-%m") for d in forecast_dates]

    # Прогнозы по моделям
    from ..schemas.forecast import ModelForecast, EnsembleForecast

    model_forecasts = {}
    ensemble_values = np.zeros(request.horizon)

    for model_name in models_to_use:
        try:
            model = ModelRegistry.get(model_name)
            model.fit(df)
            fc = model.forecast(horizon=request.horizon)

            # Конвертируем если нужно
            if hasattr(fc, "values"):
                fc = fc.values

            # Нормализуем формат
            fc = np.array(fc).flatten()
            if len(fc) < request.horizon:
                fc = np.pad(fc, (0, request.horizon - len(fc)), mode="edge")
            elif len(fc) > request.horizon:
                fc = fc[: request.horizon]

            model_forecasts[model_name] = ModelForecast(
                model=model_name,
                values=[round(float(v), 3) for v in fc],
                dates=date_strings,
                weight=weights.get(model_name, 0),
            )

            # Добавляем к ансамблю
            ensemble_values += fc * weights.get(model_name, 0)

        except Exception as e:
            print(f"Ошибка модели {model_name}: {e}")
            continue

    if not model_forecasts:
        raise ValueError("Все модели дали ошибку")

    # Ансамбль
    ensemble = EnsembleForecast(
        values=[round(float(v), 3) for v in ensemble_values], dates=date_strings
    )

    return ForecastResponse(
        ensemble=ensemble,
        models=model_forecasts,
        data_date=last_date.strftime("%Y-%m"),
        version="4.0",
    )


@router.post("/batch", response_model=BatchForecastResponse)
async def create_batch_forecast(request: BatchForecastRequest):
    """
    Пакетное прогнозирование инфляции.

    Принимает список сценариев с разными параметрами
    и возвращает прогноз для каждого сценария.

    Пример использования:
    - Сравнить разные горизонты (h=6 vs h=12)
    - Сравнить разные модели (Ridge vs LightGBM)
    - Сравнить разные веса (равные vs оптимизированные)
    """
    try:
        # Загружаем данные один раз
        df = get_data()
        last_date = df.dropna(subset=["Все товары и услуги"]).index.max()

        # Обрабатываем каждый сценарий
        results = []
        for i, scenario in enumerate(request.scenarios):
            try:
                forecast = create_forecast(scenario, df, last_date)
                results.append(forecast)
            except Exception as e:
                print(f"Ошибка сценария {i}: {e}")
                # Продолжаем с другими сценариями
                continue

        if not results:
            raise HTTPException(
                status_code=500, detail="Все сценарии завершились с ошибкой"
            )

        return BatchForecastResponse(results=results, count=len(results), version="4.0")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
