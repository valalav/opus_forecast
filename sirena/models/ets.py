"""
ETS модель для прогнозирования инфляции КБР
============================================

Exponential Smoothing с сезонностью.

Вес в ансамбле: 5%
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Dict, Any, Optional
import warnings

warnings.filterwarnings('ignore')

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("ets")
class ETSForecaster(BaseForecaster):
    """
    ETS (Error-Trend-Seasonality) модель.

    Использует Holt-Winters экспоненциальное сглаживание.
    """

    name = "ets"
    MIN_TRAIN_SIZE = 24

    def __init__(
        self,
        trend: str = 'add',
        seasonal: str = 'add',
        seasonal_periods: int = 12,
        damped_trend: bool = False,
        **kwargs
    ):
        """
        Инициализация ETS.

        Args:
            trend: 'add' или 'mul'
            seasonal: 'add' или 'mul'
            seasonal_periods: период сезонности
            damped_trend: затухающий тренд
        """
        super().__init__(**kwargs)
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend

        self.model = None
        self.fit_result = None
        self.last_values = None

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'ETSForecaster':
        """Обучение ETS."""
        series = self._validate_data(df, target_col)

        # Конвертируем в MoM
        if series.mean() > 50:
            series = series - 100

        self.last_values = series.values

        # Для ETS нужно минимум 2 сезона
        if len(series) < 2 * self.seasonal_periods:
            self.model = ExponentialSmoothing(
                series,
                trend=self.trend,
                seasonal=None,
                damped_trend=self.damped_trend
            )
        else:
            self.model = ExponentialSmoothing(
                series,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                damped_trend=self.damped_trend
            )

        self.fit_result = self.model.fit(optimized=True)
        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Прогноз ETS."""
        self._check_fitted()

        forecast = self.fit_result.forecast(steps=horizon)
        return forecast.values

    def forecast_with_intervals(self, horizon: int = 12) -> Dict[str, np.ndarray]:
        """Прогноз с интервалами."""
        self._check_fitted()

        forecast = self.fit_result.forecast(steps=horizon)

        # Оценка интервалов через residuals
        residuals = self.fit_result.resid
        std_resid = np.std(residuals)

        z = 1.96
        lower = forecast - z * std_resid * np.sqrt(np.arange(1, horizon + 1))
        upper = forecast + z * std_resid * np.sqrt(np.arange(1, horizon + 1))

        return {
            'mean': forecast.values,
            'lower': lower.values,
            'upper': upper.values,
            'aic': getattr(self.fit_result, 'aic', None)
        }

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """Бэктестирование ETS."""
        if target_col in df.columns:
            series = df[target_col].dropna()
        else:
            series = df.dropna()

        if series.mean() > 50:
            series = series - 100

        test_dates = series[series.index >= start_date].index
        results = []

        for target_date in test_dates:
            cutoff = target_date - pd.DateOffset(months=1)
            train_data = series[series.index <= cutoff]

            if len(train_data) < self.MIN_TRAIN_SIZE:
                continue

            try:
                model = ETSForecaster(
                    trend=self.trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonal_periods
                )
                train_df = pd.DataFrame({target_col: train_data + 100})
                model.fit(train_df, target_col)
                fc = model.forecast(horizon=1)

                actual = series.loc[target_date]

                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': fc[0],
                    'error': actual - fc[0]
                })
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_components(self) -> Dict[str, Any]:
        """Компоненты модели."""
        if self.fit_result is None:
            return {}

        return {
            'level': self.fit_result.level,
            'trend': getattr(self.fit_result, 'trend', None),
            'season': getattr(self.fit_result, 'season', None),
            'resid': self.fit_result.resid
        }


# Алиас для обратной совместимости
SirenaETS = ETSForecaster
