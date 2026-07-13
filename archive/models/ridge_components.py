"""
Ridge с расширенными лагами компонентов
========================================

Идея: компоненты (Продовольствие, Непродовольствие, Услуги)
имеют разную инерционность. Услуги особенно инертны.

Новые признаки:
- food_lag3, food_lag6
- nonfood_lag3, nonfood_lag6
- services_lag3, services_lag6 (услуги особенно важны)
- Interaction features
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from typing import Dict, Optional, Any, List

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("ridge_components")
class RidgeComponentsForecaster(BaseForecaster):
    """
    Ridge регрессия с расширенными лагами компонентов.

    Улучшения:
    - Лаги компонентов 1, 3, 6 месяцев
    - Momentum компонентов
    - Interaction features (food × services)
    """

    name = "ridge_components"
    MIN_TRAIN_SIZE = 36
    OUTLIER_YEARS = [2010, 2022]
    ALPHA = 0.3

    # Веса компонентов (Росстат)
    COMPONENT_WEIGHTS = {
        'food': 0.3948,
        'nonfood': 0.3653,
        'services': 0.2342
    }

    # ETS веса по месяцам
    ETS_WEIGHTS = {
        1: 0.9, 2: 0.0, 3: 0.5, 4: 0.3,
        5: 0.9, 6: 0.5, 7: 0.0, 8: 0.5,
        9: 0.9, 10: 0.9, 11: 0.0, 12: 0.0
    }

    # Базовые признаки (как в Ridge Extended)
    BASE_FEATURES = [
        # Лаги целевой
        'y_lag1', 'y_lag2', 'y_lag12',
        'y_lag3', 'y_lag6',
        # MA и momentum
        'y_ma3', 'y_ma6',
        'd_y_lag1', 'd_y_lag3',
        # Volatility
        'y_vol3', 'y_vol6',
        # Сезонность
        'month_sin', 'month_cos',
        'quarter_sin', 'quarter_cos',
        # Календарь
        'is_jan', 'is_dec',
        'is_tariff_month', 'is_q1', 'is_summer',
        # ETS
        'seasonal_norm', 'deviation_lag1',
    ]

    # Расширенные признаки компонентов
    COMPONENT_FEATURES = [
        # Продовольствие (высокая волатильность)
        'food_lag1', 'food_lag3', 'food_lag6',
        'food_momentum',
        # Непродовольствие (более стабильно)
        'nonfood_lag1', 'nonfood_lag3', 'nonfood_lag6',
        'nonfood_momentum',
        # Услуги (инертные)
        'services_lag1', 'services_lag3', 'services_lag6',
        'services_momentum',
        # Interactions
        'food_services_spread',  # разница food vs services
        'weighted_components_lag1',  # взвешенное среднее компонентов
    ]

    def __init__(self, alpha: float = None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha or self.ALPHA
        self.ridge = None
        self.scaler = None
        self.seasonal_norm = None
        self._features = None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков с расширенными лагами компонентов."""
        df = df.copy()

        df['month'] = df.index.month
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter

        y = df['Все товары и услуги']

        # === Базовые признаки ===
        df['y_lag1'] = y.shift(1)
        df['y_lag2'] = y.shift(2)
        df['y_lag12'] = y.shift(12)
        df['y_lag3'] = y.shift(3)
        df['y_lag6'] = y.shift(6)

        df['y_ma3'] = y.rolling(3).mean().shift(1)
        df['y_ma6'] = y.rolling(6).mean().shift(1)

        df['d_y_lag1'] = (y.shift(1) - y.shift(2))
        df['d_y_lag3'] = (y.shift(1) - y.shift(4))

        df['y_vol3'] = y.rolling(3).std().shift(1)
        df['y_vol6'] = y.rolling(6).std().shift(1)

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

        # === Расширенные признаки компонентов ===

        # Продовольствие
        if 'Продовольственные товары' in df.columns:
            food = df['Продовольственные товары']
            df['food_lag1'] = food.shift(1)
            df['food_lag3'] = food.shift(3)
            df['food_lag6'] = food.shift(6)
            df['food_momentum'] = food.shift(1) - food.shift(3)
        else:
            df['food_lag1'] = df['y_lag1']
            df['food_lag3'] = df['y_lag3']
            df['food_lag6'] = df['y_lag6']
            df['food_momentum'] = df['d_y_lag1']

        # Непродовольствие
        if 'Непродовольственные товары' in df.columns:
            nonfood = df['Непродовольственные товары']
            df['nonfood_lag1'] = nonfood.shift(1)
            df['nonfood_lag3'] = nonfood.shift(3)
            df['nonfood_lag6'] = nonfood.shift(6)
            df['nonfood_momentum'] = nonfood.shift(1) - nonfood.shift(3)
        else:
            df['nonfood_lag1'] = df['y_lag1']
            df['nonfood_lag3'] = df['y_lag3']
            df['nonfood_lag6'] = df['y_lag6']
            df['nonfood_momentum'] = df['d_y_lag1']

        # Услуги (особенно инертные)
        if 'Услуги' in df.columns:
            services = df['Услуги']
            df['services_lag1'] = services.shift(1)
            df['services_lag3'] = services.shift(3)
            df['services_lag6'] = services.shift(6)
            df['services_momentum'] = services.shift(1) - services.shift(3)
        else:
            df['services_lag1'] = df['y_lag1']
            df['services_lag3'] = df['y_lag3']
            df['services_lag6'] = df['y_lag6']
            df['services_momentum'] = df['d_y_lag1']

        # === Interaction features ===

        # Спред продовольствие vs услуги (волатильность vs инерция)
        df['food_services_spread'] = df['food_lag1'] - df['services_lag1']

        # Взвешенное среднее компонентов (проверка когерентности)
        w = self.COMPONENT_WEIGHTS
        df['weighted_components_lag1'] = (
            w['food'] * df['food_lag1'] +
            w['nonfood'] * df['nonfood_lag1'] +
            w['services'] * df['services_lag1']
        )

        return df

    def _compute_seasonal_norm(self, df: pd.DataFrame) -> pd.Series:
        """Сезонная норма."""
        clean_df = df[~df['year'].isin(self.OUTLIER_YEARS)]
        return clean_df.groupby('month')['Все товары и услуги'].mean()

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'RidgeComponentsForecaster':
        """Обучение модели."""
        self._validate_data(df, target_col)

        df_prep = self._prepare_features(df)
        self.seasonal_norm = self._compute_seasonal_norm(df_prep)

        df_prep['seasonal_norm'] = df_prep['month'].map(self.seasonal_norm)
        df_prep['deviation_lag1'] = df_prep['y_lag1'] - df_prep['month'].shift(1).map(self.seasonal_norm)

        self._features = self.BASE_FEATURES + self.COMPONENT_FEATURES

        # Исключаем выбросные годы
        train_df = df_prep[~df_prep['year'].isin(self.OUTLIER_YEARS)]
        train_clean = train_df.dropna(subset=self._features + [target_col])

        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных: {len(train_clean)} < {self.MIN_TRAIN_SIZE}")

        X = train_clean[self._features].values
        y = train_clean[target_col].values

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(X_scaled, y)

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Прогноз (ETS fallback)."""
        self._check_fitted()

        if self.seasonal_norm is None:
            return np.zeros(horizon)

        last_month = self._last_train_date.month if self._last_train_date else 1
        predictions = []

        for i in range(horizon):
            month = ((last_month + i) % 12) + 1
            pred = self.seasonal_norm.get(month, 100.0)
            predictions.append(pred)

        return np.array(predictions)

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Точечный прогноз."""
        self._check_fitted()

        df_prep = self._prepare_features(df)
        df_prep['seasonal_norm'] = df_prep['month'].map(self.seasonal_norm)
        df_prep['deviation_lag1'] = df_prep['y_lag1'] - df_prep['month'].shift(1).map(self.seasonal_norm)

        test_row = df_prep.loc[[target_date]]
        X_test = self.scaler.transform(test_row[self._features].values)
        pred_ridge = self.ridge.predict(X_test)[0]

        target_month = target_date.month
        pred_ets = self.seasonal_norm.get(target_month, 100.0)

        ets_weight = self.ETS_WEIGHTS.get(target_month, 0.3)
        pred_combined = (1 - ets_weight) * pred_ridge + ets_weight * pred_ets

        return {
            'date': target_date,
            'prediction': pred_combined,
            'pred_ridge': pred_ridge,
            'pred_ets': pred_ets,
            'ets_weight': ets_weight,
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
                model = RidgeComponentsForecaster(alpha=self.alpha)
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

    def get_feature_importance(self) -> pd.DataFrame:
        """Важность признаков."""
        self._check_fitted()

        importance = pd.DataFrame({
            'feature': self._features,
            'coefficient': self.ridge.coef_
        })
        importance['abs_coef'] = importance['coefficient'].abs()
        importance['is_component'] = importance['feature'].isin(self.COMPONENT_FEATURES)
        return importance.sort_values('abs_coef', ascending=False)

    def get_model_info(self) -> Dict:
        """Информация о модели."""
        return {
            'name': self.name,
            'alpha': self.alpha,
            'features_count': len(self._features) if self._features else 0,
            'component_features': len(self.COMPONENT_FEATURES),
            'is_fitted': self._is_fitted
        }
