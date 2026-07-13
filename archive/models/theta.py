"""
Theta Method — простой и эффективный метод прогнозирования
==========================================================

Часто побеждает в M-competitions благодаря простоте.

Метод:
1. Декомпозиция ряда на theta-lines
2. Theta=0 (линейный тренд)
3. Theta=2 (усиленная кривизна)
4. Прогноз = среднее двух theta-lines

Для сезонных данных: сначала десезонализация, потом Theta.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import warnings

from .base import BaseForecaster
from .registry import ModelRegistry

# Проверяем statsmodels
try:
    from statsmodels.tsa.forecasting.theta import ThetaModel
    THETA_AVAILABLE = True
except ImportError:
    THETA_AVAILABLE = False


@ModelRegistry.register("theta")
class ThetaForecaster(BaseForecaster):
    """
    Theta Method с сезонной коррекцией.

    Простая, но эффективная модель для short-term forecasting.
    """

    name = "theta"
    MIN_TRAIN_SIZE = 36
    OUTLIER_YEARS = [2010, 2022]
    SEASONAL_PERIOD = 12

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not THETA_AVAILABLE:
            raise ImportError("statsmodels >= 0.12 не установлен")

        self.model = None
        self.seasonal_factors = None

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'ThetaForecaster':
        """Обучение Theta модели."""
        self._validate_data(df, target_col)

        # Исключаем выбросные годы
        df_clean = df[~df.index.year.isin(self.OUTLIER_YEARS)].copy()
        y = df_clean[target_col].dropna()

        if len(y) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных: {len(y)} < {self.MIN_TRAIN_SIZE}")

        # Вычисляем сезонные факторы
        monthly_means = y.groupby(y.index.month).mean()
        overall_mean = y.mean()
        self.seasonal_factors = monthly_means / overall_mean

        # Десезонализация
        y_deseas = y.copy()
        for idx in y.index:
            y_deseas.loc[idx] = y.loc[idx] / self.seasonal_factors[idx.month]

        # Обучаем Theta на десезонализированных данных
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = ThetaModel(y_deseas, period=self.SEASONAL_PERIOD)
            self.fit_result = self.model.fit()

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Прогноз на горизонт."""
        self._check_fitted()

        # Theta прогноз (десезонализированный)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc_deseas = self.fit_result.forecast(horizon)

        # Ресезонализация
        predictions = []
        last_month = self._last_train_date.month

        for i, val in enumerate(fc_deseas):
            month = ((last_month + i) % 12) + 1
            seasonal_factor = self.seasonal_factors.get(month, 1.0)
            predictions.append(val * seasonal_factor)

        return np.array(predictions)

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Точечный прогноз на дату."""
        self._check_fitted()

        # Определяем горизонт
        h = (target_date.year - self._last_train_date.year) * 12 + \
            (target_date.month - self._last_train_date.month)
        h = max(1, h)

        # Theta прогноз
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc_deseas = self.fit_result.forecast(h)

        pred_deseas = fc_deseas.iloc[-1] if hasattr(fc_deseas, 'iloc') else fc_deseas[-1]

        # Ресезонализация
        target_month = target_date.month
        seasonal_factor = self.seasonal_factors.get(target_month, 1.0)
        prediction = pred_deseas * seasonal_factor

        return {
            'date': target_date,
            'prediction': prediction,
            'pred_deseas': pred_deseas,
            'seasonal_factor': seasonal_factor,
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
                model = ThetaForecaster()
                model.fit(train_df, target_col)

                test_df = df[df.index <= target_date].copy()
                pred_result = model.predict(test_df, target_date)

                actual = df.loc[target_date, target_col]

                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred_result['prediction'],
                    'error': actual - pred_result['prediction']
                })
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_model_info(self) -> Dict:
        """Информация о модели."""
        return {
            'name': self.name,
            'seasonal_period': self.SEASONAL_PERIOD,
            'is_fitted': self._is_fitted
        }
