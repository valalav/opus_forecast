"""
Ridge v3 — модель с dummy-переменными для выбросных лет
=======================================================

Вместо исключения 2010/2022 используем dummy-переменные:
- is_2010, is_2022 — модель сама научится корректировать на эти периоды
- Сохраняем все данные (не теряем 24 месяца)
- Модель может использовать паттерны выбросных лет для будущих шоков
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from typing import Dict, Optional, Any, List

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("ridge_v3")
class RidgeV3Forecaster(BaseForecaster):
    """
    Ridge v3 с dummy-переменными для выбросных лет вместо их исключения.

    Преимущества:
    - Сохраняем все данные (2010, 2022 не исключаются)
    - Модель сама определяет коррекцию на выбросные периоды
    - Лучшая адаптация к будущим кризисам
    """

    name = "ridge_v3"
    MIN_TRAIN_SIZE = 36
    ALPHA = 0.3

    # Выбросные годы как dummy, не исключаем
    OUTLIER_YEARS = [2010, 2022]

    # ETS веса по месяцам
    ETS_WEIGHTS = {
        1: 0.9, 2: 0.0, 3: 0.5, 4: 0.3,
        5: 0.9, 6: 0.5, 7: 0.0, 8: 0.5,
        9: 0.9, 10: 0.9, 11: 0.0, 12: 0.0
    }

    # Признаки (включая dummy для выбросных лет)
    BASE_FEATURES = [
        # Стандартные лаги
        'y_lag1', 'y_lag2', 'y_lag12',
        # Дополнительные лаги
        'y_lag3', 'y_lag6',
        # Скользящие средние
        'y_ma3', 'y_ma6',
        # Momentum
        'd_y_lag1', 'd_y_lag3',
        # Volatility
        'y_vol3', 'y_vol6',
        # Сезонность
        'month_sin', 'month_cos',
        'quarter_sin', 'quarter_cos',
        # Календарь
        'is_jan', 'is_dec',
        'is_tariff_month', 'is_q1', 'is_summer',
        # Компоненты
        'food_lag1', 'nonfood_lag1', 'services_lag1',
        # ETS норма
        'seasonal_norm', 'deviation_lag1',
        # === НОВОЕ: Dummy для выбросных лет ===
        'is_2010', 'is_2022',
        # Интеракции выбросных лет с месяцами (опционально)
        'is_crisis_jan',  # январь в кризисные годы
    ]

    MACRO_FEATURES = [
        'ruonia_diff_lag1',
        'spread_lag4',
        'ki_diff_lag6',
        'ki_vol',
    ]

    def __init__(self, alpha: float = None, use_macro: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha or self.ALPHA
        self.use_macro = use_macro
        self.ridge = None
        self.scaler = None
        self.seasonal_norm = None
        self._has_macro = False
        self._features = None

    @property
    def FEATURES(self) -> List[str]:
        if self._features is not None:
            return self._features
        return self.BASE_FEATURES

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков включая dummy для выбросных лет."""
        df = df.copy()

        df['month'] = df.index.month
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter

        y = df['Все товары и услуги']

        # === Стандартные лаги ===
        df['y_lag1'] = y.shift(1)
        df['y_lag2'] = y.shift(2)
        df['y_lag12'] = y.shift(12)
        df['y_lag3'] = y.shift(3)
        df['y_lag6'] = y.shift(6)

        # === Скользящие средние ===
        df['y_ma3'] = y.rolling(3).mean().shift(1)
        df['y_ma6'] = y.rolling(6).mean().shift(1)

        # === Momentum ===
        df['d_y_lag1'] = (y.shift(1) - y.shift(2))
        df['d_y_lag3'] = (y.shift(1) - y.shift(4))

        # === Volatility ===
        df['y_vol3'] = y.rolling(3).std().shift(1)
        df['y_vol6'] = y.rolling(6).std().shift(1)

        # === Сезонные признаки ===
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

        # === Календарные признаки ===
        df['is_jan'] = (df['month'] == 1).astype(int)
        df['is_dec'] = (df['month'] == 12).astype(int)
        df['is_tariff_month'] = (df['month'] == 7).astype(int)
        df['is_q1'] = (df['quarter'] == 1).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)

        # === НОВОЕ: Dummy для выбросных лет ===
        df['is_2010'] = (df['year'] == 2010).astype(int)
        df['is_2022'] = (df['year'] == 2022).astype(int)

        # Интеракция: январь в кризисные годы
        df['is_crisis_jan'] = ((df['is_jan'] == 1) &
                               (df['year'].isin(self.OUTLIER_YEARS))).astype(int)

        # === Лаги компонентов ===
        if 'Продовольственные товары' in df.columns:
            df['food_lag1'] = df['Продовольственные товары'].shift(1)
        else:
            df['food_lag1'] = df['y_lag1']

        if 'Непродовольственные товары' in df.columns:
            df['nonfood_lag1'] = df['Непродовольственные товары'].shift(1)
        else:
            df['nonfood_lag1'] = df['y_lag1']

        if 'Услуги' in df.columns:
            df['services_lag1'] = df['Услуги'].shift(1)
        else:
            df['services_lag1'] = df['y_lag1']

        return df

    def _add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавить макро-признаки Ki и Ruonia."""
        df = df.copy()

        if 'Ki' not in df.columns or 'Ruonia' not in df.columns:
            return df

        df['ruonia_diff'] = df['Ruonia'].diff()
        df['ruonia_diff_lag1'] = df['ruonia_diff'].shift(1)

        df['spread'] = df['Ki'] - df['Ruonia']
        df['spread_lag4'] = df['spread'].shift(4)

        df['ki_diff'] = df['Ki'].diff()
        df['ki_diff_lag6'] = df['ki_diff'].shift(6)

        df['ki_vol'] = df['Ki'].rolling(6).std().shift(1)

        for col in self.MACRO_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        return df

    def _compute_seasonal_norm(self, df: pd.DataFrame) -> pd.Series:
        """Вычисление сезонной нормы (всё ещё исключаем выбросы для ETS)."""
        # Для ETS нормы всё ещё исключаем выбросы
        clean_df = df[~df['year'].isin(self.OUTLIER_YEARS)]
        return clean_df.groupby('month')['Все товары и услуги'].mean()

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'RidgeV3Forecaster':
        """Обучение модели на ВСЕХ данных (выбросные годы не исключаем)."""
        series = self._validate_data(df, target_col)

        df_prep = self._prepare_features(df)

        # ETS норма по-прежнему вычисляется без выбросов
        self.seasonal_norm = self._compute_seasonal_norm(df_prep)

        df_prep['seasonal_norm'] = df_prep['month'].map(self.seasonal_norm)
        df_prep['deviation_lag1'] = df_prep['y_lag1'] - df_prep['month'].shift(1).map(self.seasonal_norm)

        self._features = self.BASE_FEATURES.copy()

        self._has_macro = False
        if self.use_macro and 'Ki' in df.columns and 'Ruonia' in df.columns:
            df_prep = self._add_macro_features(df_prep)
            available_macro = [f for f in self.MACRO_FEATURES if f in df_prep.columns]
            if available_macro:
                self._features.extend(available_macro)
                self._has_macro = True

        # === КЛЮЧЕВОЕ ОТЛИЧИЕ: НЕ исключаем выбросные годы ===
        train_clean = df_prep.dropna(subset=self._features + [target_col])

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
        """Прогноз на горизонт."""
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
        """Точечный прогноз на дату."""
        self._check_fitted()

        df_prep = self._prepare_features(df)
        df_prep['seasonal_norm'] = df_prep['month'].map(self.seasonal_norm)
        df_prep['deviation_lag1'] = df_prep['y_lag1'] - df_prep['month'].shift(1).map(self.seasonal_norm)

        if self._has_macro:
            df_prep = self._add_macro_features(df_prep)

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
            'model': self.name,
            'has_macro': self._has_macro
        }

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """Бэктестирование модели."""
        start = pd.Timestamp(start_date)

        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []

        for target_date in test_dates:
            train_df = df[df.index < target_date].copy()

            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
                continue

            try:
                model = RidgeV3Forecaster(alpha=self.alpha, use_macro=self.use_macro)
                model.fit(train_df, target_col)

                test_df = df[df.index <= target_date].copy()
                pred_result = model.predict(test_df, target_date)

                actual = df.loc[target_date, target_col]

                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred_result['prediction'],
                    'error': actual - pred_result['prediction'],
                    'pred_ridge': pred_result['pred_ridge'],
                    'pred_ets': pred_result['pred_ets'],
                    'has_macro': pred_result.get('has_macro', False)
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
        importance['is_macro'] = importance['feature'].isin(self.MACRO_FEATURES)
        importance['is_dummy'] = importance['feature'].isin(['is_2010', 'is_2022', 'is_crisis_jan'])
        return importance.sort_values('abs_coef', ascending=False)

    def get_model_info(self) -> Dict:
        """Информация о модели."""
        return {
            'name': self.name,
            'alpha': self.alpha,
            'features_count': len(self._features) if self._features else 0,
            'has_macro': self._has_macro,
            'is_fitted': self._is_fitted,
            'uses_dummy_years': True
        }
