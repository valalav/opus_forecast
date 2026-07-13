"""
Эндпоинты для прогнозирования недельных цен
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

router = APIRouter(prefix="/weekly", tags=["Weekly Forecast"])


# Schemas
class WeeklyForecastRequest(BaseModel):
    """Запрос прогноза недельных цен."""
    horizon: int = Field(default=4, ge=1, le=52, description="Горизонт в неделях")
    target: str = Field(default="aggregate", description="Цель: 'aggregate', 'fruits_veg' или название товара")
    include_products: bool = Field(default=False, description="Включить прогнозы по товарам")


class WeeklyForecastResponse(BaseModel):
    """Ответ прогноза недельных цен."""
    forecast: List[float]
    dates: List[str]
    target: str
    horizon: int
    last_data_date: str
    products: Optional[Dict[str, List[float]]] = None


class WeeklySignalResponse(BaseModel):
    """Сигнал для nowcasting."""
    last_date: str
    last_4w_avg: float
    fruits_veg_4w: Optional[float]
    forecast_4w: List[float]
    forecast_sum: float
    monthly_proxy: float


class WeeklyProductsResponse(BaseModel):
    """Список доступных товаров."""
    products: List[str]
    key_products: List[str]
    fruits_veg: List[str]
    total_weeks: int
    date_range: Dict[str, str]


def get_weekly_model():
    """Получить и обучить модель недельных цен."""
    from sirena.models import ModelRegistry

    model = ModelRegistry.get("weekly")
    model.fit()
    return model


@router.get("/products", response_model=WeeklyProductsResponse)
async def list_products():
    """
    Список доступных товаров для прогнозирования.

    Возвращает:
    - Все доступные товары
    - Ключевые товары корзины
    - Плодоовощи (высокая волатильность)
    """
    try:
        model = get_weekly_model()

        return WeeklyProductsResponse(
            products=model.available_products,
            key_products=model.KEY_PRODUCTS[:20],
            fruits_veg=model.FRUITS_VEG,
            total_weeks=len(model.df_wide) if model.df_wide is not None else 0,
            date_range={
                "start": model.df_wide.index.min().strftime('%Y-%m-%d') if model.df_wide is not None else "",
                "end": model.df_wide.index.max().strftime('%Y-%m-%d') if model.df_wide is not None else ""
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecast", response_model=WeeklyForecastResponse)
async def create_weekly_forecast(request: WeeklyForecastRequest):
    """
    Прогноз недельных цен.

    Поддерживает:
    - Агрегированный прогноз (медиана по корзине)
    - Прогноз плодоовощей
    - Прогноз отдельных товаров
    """
    try:
        model = get_weekly_model()

        # Основной прогноз
        forecast = model.forecast(horizon=request.horizon, target_col=request.target)

        # Даты прогноза
        last_date = model._last_train_date
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=request.horizon,
            freq='W-MON'
        )

        # Прогнозы по товарам (опционально)
        products_forecast = None
        if request.include_products:
            products_forecast = {
                k: [round(float(v), 3) for v in vals]
                for k, vals in model.forecast_products(horizon=request.horizon).items()
            }

        return WeeklyForecastResponse(
            forecast=[round(float(v), 3) for v in forecast],
            dates=[d.strftime('%Y-%m-%d') for d in forecast_dates],
            target=request.target,
            horizon=request.horizon,
            last_data_date=last_date.strftime('%Y-%m-%d'),
            products=products_forecast
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast/quick")
async def quick_weekly_forecast(
    horizon: int = Query(default=4, ge=1, le=12, description="Горизонт в неделях")
):
    """
    Быстрый прогноз недельных цен (агрегат).
    """
    try:
        model = get_weekly_model()
        forecast = model.forecast(horizon=horizon)

        last_date = model._last_train_date
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=horizon,
            freq='W-MON'
        )

        return {
            "forecast": [round(float(v), 3) for v in forecast],
            "dates": [d.strftime('%Y-%m-%d') for d in forecast_dates],
            "target": "aggregate",
            "last_data_date": last_date.strftime('%Y-%m-%d')
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signal", response_model=WeeklySignalResponse)
async def get_nowcast_signal():
    """
    Получить сигнал для nowcasting месячной инфляции.

    Возвращает:
    - Среднее WoW за последние 4 недели
    - Прогноз на следующие 4 недели
    - Аппроксимацию месячного прогноза
    """
    try:
        model = get_weekly_model()
        signal = model.get_current_signal()

        return WeeklySignalResponse(
            last_date=signal['last_date'].strftime('%Y-%m-%d'),
            last_4w_avg=round(signal['last_4w_avg'], 4),
            fruits_veg_4w=round(signal['fruits_veg_4w'], 4) if signal['fruits_veg_4w'] else None,
            forecast_4w=[round(v, 4) for v in signal['forecast_4w']],
            forecast_sum=round(signal['forecast_sum'], 4),
            monthly_proxy=round(signal['monthly_proxy'], 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{product}")
async def get_product_history(
    product: str,
    weeks: int = Query(default=52, ge=4, le=200, description="Количество недель истории")
):
    """
    Историческая динамика цены товара.
    """
    try:
        model = get_weekly_model()

        if model.df_wide is None:
            raise HTTPException(status_code=500, detail="Данные не загружены")

        # Ищем товар (частичное совпадение)
        matching = [p for p in model.df_wide.columns if product.lower() in p.lower()]
        if not matching:
            raise HTTPException(status_code=404, detail=f"Товар '{product}' не найден")

        product_col = matching[0]
        data = model.df_wide[product_col].dropna().tail(weeks)

        # WoW изменения
        wow = data.pct_change() * 100

        return {
            "product": product_col,
            "prices": [round(float(v), 2) for v in data.values],
            "wow_changes": [round(float(v), 3) if not np.isnan(v) else None for v in wow.values],
            "dates": [d.strftime('%Y-%m-%d') for d in data.index],
            "stats": {
                "mean_price": round(float(data.mean()), 2),
                "min_price": round(float(data.min()), 2),
                "max_price": round(float(data.max()), 2),
                "mean_wow": round(float(wow.mean()), 3),
                "volatility": round(float(wow.std()), 3)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest")
async def backtest_weekly(
    start_date: str = Query(default="2024-06-01", description="Начальная дата бэктеста")
):
    """
    Бэктест модели недельных цен.
    """
    try:
        model = get_weekly_model()
        results = model.backtest(start_date=start_date)

        if results.empty:
            return {
                "message": "Недостаточно данных для бэктеста",
                "metrics": None,
                "results": []
            }

        metrics = model.get_metrics(results)

        return {
            "model": "weekly",
            "start_date": start_date,
            "metrics": {
                "MAE": round(metrics['MAE'], 4),
                "RMSE": round(metrics['RMSE'], 4),
                "KPI": round(metrics['KPI'], 1),
                "n_observations": metrics['n_obs']
            },
            "results": [
                {
                    "date": row['date'].strftime('%Y-%m-%d'),
                    "actual": round(row['actual'], 4),
                    "prediction": round(row['prediction'], 4),
                    "error": round(row['error'], 4)
                }
                for _, row in results.iterrows()
            ]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
