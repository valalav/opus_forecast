"""
Constrained Components — компонентный прогноз с ограничениями
=============================================================

Прогнозируем 3 компонента отдельно, но с ограничением:
сумма компонентов = прогноз Total

Веса компонентов (Росстат КБР):
- Продовольственные: 39.48%
- Непродовольственные: 36.53%
- Услуги: 23.42%

Корреляции с Total (высокие!):
- Food: r=0.909
- NonFood: r=0.782
- Services: r=0.210
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from typing import Dict, Optional, Any, List

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("constrained_components")
class ConstrainedComponentsForecaster(BaseForecaster):
    """
    Прогноз через компоненты с ограничением когерентности.

    Алгоритм:
    1. Прогнозируем Total отдельно (Ridge Extended)
    2. Прогнозируем каждый компонент отдельно
    3. Масштабируем компоненты так, чтобы их сумма = Total
    """

    name = "constrained_components"
    MIN_TRAIN_SIZE = 36
    ALPHA = 0.3

    OUTLIER_YEARS = [2010, 2022]

    # Веса компонентов (Росстат КБР)
    COMPONENT_WEIGHTS = {
        'Продовольственные товары': 0.3948,
        'Непродовольственные товары': 0.3653,
        'Услуги': 0.2342
    }

    # ETS веса
    ETS_WEIGHTS = {
        1: 0.9, 2: 0.0, 3: 0.5, 4: 0.3,
        5: 0.9, 6: 0.5, 7: 0.0, 8: 0.5,
        9: 0.9, 10: 0.9, 11: 0.0, 12: 0.0
    }

    BASE_FEATURES = [
        'y_lag1', 'y_lag2', 'y_lag12',
        'y_lag3', 'y_lag6',
        'y_ma3',
        'd_y_lag1',
        'y_vol3',
        'month_sin', 'month_cos',
        'quarter_sin', 'quarter_cos',
        'is_jan', 'is_dec',
        'is_tariff_month', 'is_q1', 'is_summer',
        'seasonal_norm', 'deviation_lag1',
    ]

    def __init__(self, alpha: float = None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha or self.ALPHA
        # Модели для каждого компонента + Total
        self.model_total = None
        self.model_food = None
        self.model_nonfood = None
        self.model_services = None
        self.scaler_total = None
        self.scaler_food = None
        self.scaler_nonfood = None
        self.scaler_services = None
        self.seasonal_norms = {}
        self._features = None

    def _prepare_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Подготовка признаков для конкретного компонента."""
        df = df.copy()

        df['month'] = df.index.month
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter

        y = df[target_col]

        # Лаги
        df['y_lag1'] = y.shift(1)
        df['y_lag2'] = y.shift(2)
        df['y_lag12'] = y.shift(12)
        df['y_lag3'] = y.shift(3)
        df['y_lag6'] = y.shift(6)

        # MA
        df['y_ma3'] = y.rolling(3).mean().shift(1)

        # Momentum
        df['d_y_lag1'] = (y.shift(1) - y.shift(2))

        # Volatility
        df['y_vol3'] = y.rolling(3).std().shift(1)

        # Сезонность
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

        # Календарь
        df['is_jan'] = (df['month'] == 1).astype(int)
        df['is_dec'] = (df['month'] == 12).astype(int)
        df['is_tariff_month'] = (df['month'] == 7).astype(int)
        df['is_q1'] = (df['quarter'] == 1).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)

        return df

    def _compute_seasonal_norm(self, df: pd.DataFrame, target_col: str) -> pd.Series:
        """Сезонная норма."""
        df = df.copy()
        df['month'] = df.index.month
        df['year'] = df.index.year
        clean_df = df[~df['year'].isin(self.OUTLIER_YEARS)]
        return clean_df.groupby('month')[target_col].mean()

    def _fit_component(self, df: pd.DataFrame, target_col: str):
        """Обучить модель для одного компонента."""
        df_prep = self._prepare_features(df, target_col)

        # Сезонная норма
        seasonal_norm = self._compute_seasonal_norm(df, target_col)
        self.seasonal_norms[target_col] = seasonal_norm

        df_prep['seasonal_norm'] = df_prep['month'].map(seasonal_norm)
        df_prep['deviation_lag1'] = df_prep['y_lag1'] - df_prep['month'].shift(1).map(seasonal_norm)

        # Фильтруем выбросы
        train_df = df_prep[~df_prep['year'].isin(self.OUTLIER_YEARS)]
        train_clean = train_df.dropna(subset=self.BASE_FEATURES + [target_col])

        if len(train_clean) < self.MIN_TRAIN_SIZE:
            return None, None

        X = train_clean[self.BASE_FEATURES].values
        y = train_clean[target_col].values

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        model = Ridge(alpha=self.alpha)
        model.fit(X_scaled, y)

        return model, scaler

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'ConstrainedComponentsForecaster':
        """Обучение моделей для Total и 3 компонентов."""
        self._validate_data(df, target_col)

        self._features = self.BASE_FEATURES.copy()

        # Обучаем модель для Total
        self.model_total, self.scaler_total = self._fit_component(df, 'Все товары и услуги')

        # Обучаем модели для компонентов
        if 'Продовольственные товары' in df.columns:
            self.model_food, self.scaler_food = self._fit_component(df, 'Продовольственные товары')

        if 'Непродовольственные товары' in df.columns:
            self.model_nonfood, self.scaler_nonfood = self._fit_component(df, 'Непродовольственные товары')

        if 'Услуги' in df.columns:
            self.model_services, self.scaler_services = self._fit_component(df, 'Услуги')

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Прогноз Total на горизонт."""
        self._check_fitted()

        if 'Все товары и услуги' not in self.seasonal_norms:
            return np.zeros(horizon)

        seasonal_norm = self.seasonal_norms['Все товары и услуги']
        last_month = self._last_train_date.month if self._last_train_date else 1
        predictions = []

        for i in range(horizon):
            month = ((last_month + i) % 12) + 1
            pred = seasonal_norm.get(month, 100.0)
            predictions.append(pred)

        return np.array(predictions)

    def _predict_component(self, df: pd.DataFrame, target_date: pd.Timestamp,
                           target_col: str, model, scaler) -> float:
        """Прогноз одного компонента."""
        if model is None or scaler is None:
            return None

        df_prep = self._prepare_features(df, target_col)
        seasonal_norm = self.seasonal_norms.get(target_col, {})

        df_prep['seasonal_norm'] = df_prep['month'].map(seasonal_norm)
        df_prep['deviation_lag1'] = df_prep['y_lag1'] - df_prep['month'].shift(1).map(seasonal_norm)

        test_row = df_prep.loc[[target_date]]
        X_test = scaler.transform(test_row[self.BASE_FEATURES].values)

        pred_ridge = model.predict(X_test)[0]

        target_month = target_date.month
        pred_ets = seasonal_norm.get(target_month, 100.0)

        ets_weight = self.ETS_WEIGHTS.get(target_month, 0.3)
        return (1 - ets_weight) * pred_ridge + ets_weight * pred_ets

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Прогноз с ограничением когерентности компонентов."""
        self._check_fitted()

        # Прогноз Total
        pred_total = self._predict_component(
            df, target_date, 'Все товары и услуги',
            self.model_total, self.scaler_total
        )

        # Прогнозы компонентов
        pred_food = self._predict_component(
            df, target_date, 'Продовольственные товары',
            self.model_food, self.scaler_food
        )
        pred_nonfood = self._predict_component(
            df, target_date, 'Непродовольственные товары',
            self.model_nonfood, self.scaler_nonfood
        )
        pred_services = self._predict_component(
            df, target_date, 'Услуги',
            self.model_services, self.scaler_services
        )

        # === Ограничение когерентности ===
        # Сумма компонентов (в терминах MoM индексов)
        if pred_food and pred_nonfood and pred_services:
            W = self.COMPONENT_WEIGHTS

            # Преобразуем индексы в отклонения от 100
            dev_food = pred_food - 100
            dev_nonfood = pred_nonfood - 100
            dev_services = pred_services - 100

            # Взвешенная сумма отклонений
            raw_sum = (W['Продовольственные товары'] * dev_food +
                       W['Непродовольственные товары'] * dev_nonfood +
                       W['Услуги'] * dev_services)

            # Целевое отклонение Total
            target_dev = pred_total - 100

            # Масштабируем компоненты
            if abs(raw_sum) > 0.001:
                scale = target_dev / raw_sum
                pred_food_adj = 100 + dev_food * scale
                pred_nonfood_adj = 100 + dev_nonfood * scale
                pred_services_adj = 100 + dev_services * scale
            else:
                # Если raw_sum близок к 0, используем оригинальные прогнозы
                pred_food_adj = pred_food
                pred_nonfood_adj = pred_nonfood
                pred_services_adj = pred_services

            # Альтернатива: среднее между Total и суммой компонентов
            pred_from_components = 100 + raw_sum
            pred_combined = 0.6 * pred_total + 0.4 * pred_from_components
        else:
            pred_combined = pred_total
            pred_food_adj = pred_food
            pred_nonfood_adj = pred_nonfood
            pred_services_adj = pred_services

        return {
            'date': target_date,
            'prediction': pred_combined,
            'pred_total_direct': pred_total,
            'pred_food': pred_food,
            'pred_nonfood': pred_nonfood,
            'pred_services': pred_services,
            'pred_food_adj': pred_food_adj,
            'pred_nonfood_adj': pred_nonfood_adj,
            'pred_services_adj': pred_services_adj,
            'model': self.name
        }

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """Бэктестирование."""
        start = pd.Timestamp(start_date)

        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []

        for target_date in test_dates:
            train_df = df[df.index < target_date].copy()

            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
                continue

            try:
                model = ConstrainedComponentsForecaster(alpha=self.alpha)
                model.fit(train_df, target_col)

                test_df = df[df.index <= target_date].copy()
                pred_result = model.predict(test_df, target_date)

                actual = df.loc[target_date, target_col]

                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred_result['prediction'],
                    'error': actual - pred_result['prediction'],
                    'pred_total_direct': pred_result['pred_total_direct'],
                    'pred_food': pred_result.get('pred_food'),
                    'pred_nonfood': pred_result.get('pred_nonfood'),
                    'pred_services': pred_result.get('pred_services'),
                })
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_component_importance(self) -> Dict[str, float]:
        """Вклад каждого компонента."""
        return self.COMPONENT_WEIGHTS.copy()

    def get_model_info(self) -> Dict:
        """Информация о модели."""
        return {
            'name': self.name,
            'alpha': self.alpha,
            'component_weights': self.COMPONENT_WEIGHTS,
            'has_food': self.model_food is not None,
            'has_nonfood': self.model_nonfood is not None,
            'has_services': self.model_services is not None,
            'is_fitted': self._is_fitted
        }
