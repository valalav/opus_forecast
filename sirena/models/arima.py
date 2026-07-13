"""
ARIMA/SARIMA модель для прогнозирования инфляции КБР
=====================================================

Сезонная авторегрессия.

Вес в ансамбле: 5%
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import Dict, Any, Tuple
import warnings

warnings.filterwarnings('ignore')

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("sarima")
class SARIMAForecaster(BaseForecaster):
    """
    SARIMA модель прогнозирования.

    Параметры:
    - order: (p, d, q) для ARIMA
    - seasonal_order: (P, D, Q, s) для сезонности
    """

    name = "sarima"
    MIN_TRAIN_SIZE = 24

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 0, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 12),
        **kwargs
    ):
        """
        Инициализация SARIMA.

        Args:
            order: (p, d, q) параметры ARIMA
            seasonal_order: (P, D, Q, s) сезонные параметры
        """
        super().__init__(**kwargs)
        self.order = order
        self.seasonal_order = seasonal_order

        self.model = None
        self.fit_result = None
        self.last_index = None

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'SARIMAForecaster':
        """Обучение SARIMA."""
        series = self._validate_data(df, target_col)

        # Конвертируем в MoM
        if series.mean() > 50:
            series = series - 100

        self.last_index = series.index[-1]

        self.model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.fit_result = self.model.fit(disp=False)

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Прогноз SARIMA."""
        self._check_fitted()

        forecast = self.fit_result.get_forecast(steps=horizon)
        return forecast.predicted_mean.values

    def forecast_with_intervals(self, horizon: int = 12, alpha: float = 0.05) -> Dict[str, np.ndarray]:
        """Прогноз с доверительными интервалами."""
        self._check_fitted()

        forecast = self.fit_result.get_forecast(steps=horizon)
        mean_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=alpha)

        return {
            'mean': mean_forecast.values,
            'lower': conf_int.iloc[:, 0].values,
            'upper': conf_int.iloc[:, 1].values,
            'aic': self.fit_result.aic
        }

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """Бэктестирование SARIMA."""
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
                model = SARIMAForecaster(
                    order=self.order,
                    seasonal_order=self.seasonal_order
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

    def get_diagnostics(self) -> Dict[str, Any]:
        """Диагностика модели."""
        if self.fit_result is None:
            return {}

        return {
            'aic': self.fit_result.aic,
            'bic': self.fit_result.bic,
            'llf': self.fit_result.llf,
            'resid_std': np.std(self.fit_result.resid)
        }


@ModelRegistry.register("ar1")
class AR1Forecaster(BaseForecaster):
    """Простая AR(1) модель."""

    name = "ar1"
    MIN_TRAIN_SIZE = 12

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.fit_result = None

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'AR1Forecaster':
        """Обучение AR(1)."""
        series = self._validate_data(df, target_col)

        if series.mean() > 50:
            series = series - 100

        self.model = ARIMA(series, order=(1, 0, 0))
        self.fit_result = self.model.fit()

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Прогноз AR(1)."""
        self._check_fitted()

        forecast = self.fit_result.get_forecast(steps=horizon)
        return forecast.predicted_mean.values

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """Бэктестирование AR(1)."""
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
                model = AR1Forecaster()
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


# Алиасы для обратной совместимости
SirenaARIMA = SARIMAForecaster
