"""
Prophet модель для прогнозирования инфляции КБР
================================================

Facebook Prophet с автоматической сезонностью.

Вес в ансамбле: 10%
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import warnings
import logging

warnings.filterwarnings('ignore')

from .base import BaseForecaster
from .registry import ModelRegistry

# Проверка Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    logging.getLogger('prophet').setLevel(logging.WARNING)
    logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
except ImportError:
    PROPHET_AVAILABLE = False


@ModelRegistry.register("prophet")
class ProphetForecaster(BaseForecaster):
    """
    Prophet модель прогнозирования.

    Автоматическое определение тренда и сезонности.
    """

    name = "prophet"
    MIN_TRAIN_SIZE = 24

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = False,
        daily_seasonality: bool = False,
        seasonality_mode: str = 'additive',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        outlier_years: List[int] = None,
        **kwargs
    ):
        """
        Инициализация Prophet.

        Args:
            yearly_seasonality: Годовая сезонность
            weekly_seasonality: Недельная сезонность
            daily_seasonality: Дневная сезонность
            seasonality_mode: 'additive' или 'multiplicative'
            changepoint_prior_scale: Prior для изменений тренда
            seasonality_prior_scale: Prior для сезонности
            outlier_years: Годы-выбросы
        """
        super().__init__(**kwargs)

        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not installed. Run: pip install prophet")

        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.outlier_years = outlier_years or [2022]

        self.model = None
        self.last_date = None

    def _prepare_prophet_df(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Преобразование в формат Prophet."""
        if target_col in df.columns:
            series = df[target_col].dropna()
        else:
            series = df.dropna()

        if series.mean() > 50:
            series = series - 100

        return pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'ProphetForecaster':
        """Обучение Prophet."""
        self._validate_data(df, target_col)

        prophet_df = self._prepare_prophet_df(df, target_col)

        # Исключаем выбросы
        prophet_df['year'] = prophet_df['ds'].dt.year
        prophet_df = prophet_df[~prophet_df['year'].isin(self.outlier_years)]
        prophet_df = prophet_df.drop('year', axis=1)

        self.last_date = prophet_df['ds'].max()

        # Создаём модель
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale
        )

        # Месячная сезонность
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )

        self.model.fit(prophet_df)

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Прогноз Prophet."""
        self._check_fitted()

        future = self.model.make_future_dataframe(periods=horizon, freq='MS')
        forecast = self.model.predict(future)

        forecast_future = forecast[forecast['ds'] > self.last_date]
        return forecast_future['yhat'].values

    def forecast_with_intervals(self, horizon: int = 12) -> Dict[str, Any]:
        """Прогноз с интервалами."""
        self._check_fitted()

        future = self.model.make_future_dataframe(periods=horizon, freq='MS')
        forecast = self.model.predict(future)

        forecast_future = forecast[forecast['ds'] > self.last_date]

        return {
            'mean': forecast_future['yhat'].values,
            'lower': forecast_future['yhat_lower'].values,
            'upper': forecast_future['yhat_upper'].values,
            'dates': forecast_future['ds'].values,
            'trend': forecast_future['trend'].values
        }

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """Бэктестирование Prophet."""
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
            train_series = series[series.index <= cutoff]

            if len(train_series) < self.MIN_TRAIN_SIZE:
                continue

            try:
                train_df = pd.DataFrame({
                    target_col: train_series.values + 100
                }, index=train_series.index)

                model = ProphetForecaster(
                    yearly_seasonality=self.yearly_seasonality,
                    seasonality_mode=self.seasonality_mode,
                    outlier_years=self.outlier_years
                )
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

    def get_components(self) -> pd.DataFrame:
        """Компоненты модели."""
        if self.model is None:
            return pd.DataFrame()

        future = self.model.make_future_dataframe(periods=0, freq='MS')
        forecast = self.model.predict(future)

        cols = ['ds', 'trend', 'yhat']
        if 'yearly' in forecast.columns:
            cols.append('yearly')

        return forecast[cols]
