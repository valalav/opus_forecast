"""
MSTL + Ridge — декомпозиция временного ряда + ML на остатках
============================================================

Идея:
1. Декомпозиция ряда: trend + seasonal + residual
2. Прогноз trend через drift
3. Прогноз seasonal — повторение паттерна
4. Прогноз residual — Ridge регрессия
5. Комбинирование: forecast = trend + seasonal + residual
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from typing import Dict, Optional, Any, List
import warnings

from .base import BaseForecaster
from .registry import ModelRegistry

# Проверяем statsmodels
try:
    from statsmodels.tsa.seasonal import STL
    STL_AVAILABLE = True
except ImportError:
    STL_AVAILABLE = False


@ModelRegistry.register("mstl_ridge")
class MSTLRidgeForecaster(BaseForecaster):
    """
    MSTL декомпозиция + Ridge на остатках.

    Компоненты:
    - Trend: прогноз drift методом
    - Seasonal: повторение последнего цикла
    - Residual: Ridge регрессия на лагах
    """

    name = "mstl_ridge"
    MIN_TRAIN_SIZE = 36
    OUTLIER_YEARS = [2010, 2022]
    ALPHA = 0.3

    # STL параметры
    SEASONAL_PERIOD = 12  # месячная сезонность

    RESIDUAL_FEATURES = [
        'resid_lag1', 'resid_lag2', 'resid_lag3',
        'resid_ma3',
        'month_sin', 'month_cos',
    ]

    def __init__(self, alpha: float = None, **kwargs):
        super().__init__(**kwargs)

        if not STL_AVAILABLE:
            raise ImportError("statsmodels не установлен. Выполните: pip install statsmodels")

        self.alpha = alpha or self.ALPHA
        self.ridge = None
        self.scaler = None

        # Компоненты STL
        self.trend = None
        self.seasonal = None
        self.residual = None
        self.last_trend_value = None
        self.trend_drift = None

        self._features = None

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'MSTLRidgeForecaster':
        """Обучение MSTL + Ridge."""
        self._validate_data(df, target_col)

        # Исключаем выбросные годы
        df_clean = df[~df.index.year.isin(self.OUTLIER_YEARS)].copy()
        y = df_clean[target_col].dropna()

        if len(y) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных: {len(y)} < {self.MIN_TRAIN_SIZE}")

        # STL декомпозиция
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stl = STL(y, period=self.SEASONAL_PERIOD, robust=True)
            result = stl.fit()

        self.trend = result.trend
        self.seasonal = result.seasonal
        self.residual = result.resid

        # Параметры для прогноза trend (drift метод)
        self.last_trend_value = self.trend.iloc[-1]
        # Drift = среднее изменение тренда за период
        trend_diff = self.trend.diff().dropna()
        self.trend_drift = trend_diff.mean()

        # Подготовка признаков для residual
        df_resid = pd.DataFrame(index=y.index)
        df_resid['resid'] = self.residual
        df_resid['month'] = df_resid.index.month

        df_resid['resid_lag1'] = df_resid['resid'].shift(1)
        df_resid['resid_lag2'] = df_resid['resid'].shift(2)
        df_resid['resid_lag3'] = df_resid['resid'].shift(3)
        df_resid['resid_ma3'] = df_resid['resid'].rolling(3).mean().shift(1)
        df_resid['month_sin'] = np.sin(2 * np.pi * df_resid['month'] / 12)
        df_resid['month_cos'] = np.cos(2 * np.pi * df_resid['month'] / 12)

        self._features = self.RESIDUAL_FEATURES.copy()

        # Обучаем Ridge на residual
        train_clean = df_resid.dropna(subset=self._features + ['resid'])
        X = train_clean[self._features].values
        y_resid = train_clean['resid'].values

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(X_scaled, y_resid)

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Прогноз на горизонт."""
        self._check_fitted()

        predictions = []

        for h in range(1, horizon + 1):
            # Trend: drift метод
            trend_fc = self.last_trend_value + h * self.trend_drift

            # Seasonal: повторение паттерна
            seasonal_idx = -self.SEASONAL_PERIOD + ((h - 1) % self.SEASONAL_PERIOD)
            seasonal_fc = self.seasonal.iloc[seasonal_idx]

            # Residual: среднее (для простоты)
            resid_fc = 0

            predictions.append(trend_fc + seasonal_fc + resid_fc)

        return np.array(predictions)

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Точечный прогноз на дату."""
        self._check_fitted()

        # Определяем горизонт от последней даты тренда
        h = (target_date.year - self._last_train_date.year) * 12 + \
            (target_date.month - self._last_train_date.month)
        h = max(1, h)

        # Trend: drift
        trend_fc = self.last_trend_value + h * self.trend_drift

        # Seasonal
        target_month = target_date.month
        seasonal_values = self.seasonal.groupby(self.seasonal.index.month).mean()
        seasonal_fc = seasonal_values.get(target_month, 0)

        # Residual: прогноз через Ridge
        # Берём последние residuals
        last_resids = self.residual.iloc[-3:].tolist()
        while len(last_resids) < 3:
            last_resids.insert(0, 0)

        resid_ma3 = np.mean(last_resids)

        month_sin = np.sin(2 * np.pi * target_month / 12)
        month_cos = np.cos(2 * np.pi * target_month / 12)

        X_test = np.array([[last_resids[-1], last_resids[-2], last_resids[-3],
                           resid_ma3, month_sin, month_cos]])
        X_test_scaled = self.scaler.transform(X_test)
        resid_fc = self.ridge.predict(X_test_scaled)[0]

        # Комбинируем
        prediction = trend_fc + seasonal_fc + resid_fc

        return {
            'date': target_date,
            'prediction': prediction,
            'trend': trend_fc,
            'seasonal': seasonal_fc,
            'residual': resid_fc,
            'model': self.name
        }

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """Бэктест."""
        start = pd.Timestamp(start_date)
        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []

        for target_date in test_dates:
            train_df = df[df.index < target_date].copy()

            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
                continue

            try:
                model = MSTLRidgeForecaster(alpha=self.alpha)
                model.fit(train_df, target_col)

                test_df = df[df.index <= target_date].copy()
                pred_result = model.predict(test_df, target_date)

                actual = df.loc[target_date, target_col]

                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred_result['prediction'],
                    'error': actual - pred_result['prediction'],
                    'trend': pred_result['trend'],
                    'seasonal': pred_result['seasonal'],
                    'residual': pred_result['residual']
                })
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_model_info(self) -> Dict:
        """Информация о модели."""
        return {
            'name': self.name,
            'alpha': self.alpha,
            'trend_drift': self.trend_drift,
            'last_trend': self.last_trend_value,
            'is_fitted': self._is_fitted
        }
