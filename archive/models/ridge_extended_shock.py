"""
Ridge Extended + Shock Dummies
==============================

Эксперимент v4.6: Ridge Extended с shock dummy переменными из методик ЦБ.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from typing import Dict, Any, List

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("ridge_extended_shock")
class RidgeExtendedShockForecaster(BaseForecaster):
    """Ridge Extended с shock dummies."""

    name = "ridge_extended_shock"
    MIN_TRAIN_SIZE = 36

    # НЕ исключаем 2022 — используем shock dummies
    OUTLIER_YEARS = [2010]

    ALPHA = 0.3

    ETS_WEIGHTS = {
        1: 0.9, 2: 0.0, 3: 0.5, 4: 0.3,
        5: 0.9, 6: 0.5, 7: 0.0, 8: 0.5,
        9: 0.9, 10: 0.9, 11: 0.0, 12: 0.0
    }

    BASE_FEATURES = [
        'y_lag1', 'y_lag2', 'y_lag12',
        'y_lag3', 'y_lag6',
        'y_ma3', 'y_ma6',
        'd_y_lag1', 'd_y_lag3',
        'y_vol3', 'y_vol6',
        'month_sin', 'month_cos',
        'quarter_sin', 'quarter_cos',
        'is_jan', 'is_dec',
        'is_tariff_month', 'is_q1', 'is_summer',
        'food_lag1', 'nonfood_lag1', 'services_lag1',
        'seasonal_norm', 'deviation_lag1',
    ]

    SHOCK_DUMMIES = [
        'is_shock_dec2014',
        'is_shock_jan2015',
        'is_shock_mar2022',
        'is_shock_apr2022',
        'is_shock_2022',
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

    def _add_shock_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавить shock dummy переменные."""
        df = df.copy()
        df['is_shock_dec2014'] = ((df.index.year == 2014) & (df.index.month == 12)).astype(int)
        df['is_shock_jan2015'] = ((df.index.year == 2015) & (df.index.month == 1)).astype(int)
        df['is_shock_mar2022'] = ((df.index.year == 2022) & (df.index.month == 3)).astype(int)
        df['is_shock_apr2022'] = ((df.index.year == 2022) & (df.index.month == 4)).astype(int)
        df['is_shock_2022'] = (df.index.year == 2022).astype(int)
        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков."""
        df = df.copy()

        df['month'] = df.index.month
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter

        y = df['Все товары и услуги']

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

        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

        df['is_jan'] = (df['month'] == 1).astype(int)
        df['is_dec'] = (df['month'] == 12).astype(int)
        df['is_tariff_month'] = (df['month'] == 7).astype(int)
        df['is_q1'] = (df['quarter'] == 1).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)

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

        # Добавляем shock dummies
        df = self._add_shock_dummies(df)

        return df

    def _add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавить макро-признаки."""
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
        """Сезонная норма (исключаем 2022 при расчёте)."""
        clean_df = df[df['year'] != 2022]
        return clean_df.groupby('month')['Все товары и услуги'].mean()

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'RidgeExtendedShockForecaster':
        """Обучение модели."""
        self._validate_data(df, target_col)

        df_prep = self._prepare_features(df)
        self.seasonal_norm = self._compute_seasonal_norm(df_prep)

        df_prep['seasonal_norm'] = df_prep['month'].map(self.seasonal_norm)
        df_prep['deviation_lag1'] = df_prep['y_lag1'] - df_prep['month'].shift(1).map(self.seasonal_norm)

        self._features = self.BASE_FEATURES.copy() + self.SHOCK_DUMMIES

        # Макро-признаки
        self._has_macro = False
        if self.use_macro and 'Ki' in df.columns and 'Ruonia' in df.columns:
            df_prep = self._add_macro_features(df_prep)
            available_macro = [f for f in self.MACRO_FEATURES if f in df_prep.columns]
            if available_macro:
                self._features.extend(available_macro)
                self._has_macro = True

        # Исключаем только 2010 (2022 обрабатывается через shock dummies)
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

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Прогноз."""
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
            'model': self.name,
            'has_macro': self._has_macro
        }

    def forecast(self, horizon: int = 12) -> np.ndarray:
        self._check_fitted()
        last_month = self._last_train_date.month if self._last_train_date else 1
        return np.array([self.seasonal_norm.get(((last_month + i) % 12) + 1, 100.0) for i in range(horizon)])

    def backtest(self, df: pd.DataFrame, start_date: str = '2023-01-01', target_col: str = 'Все товары и услуги') -> pd.DataFrame:
        start = pd.Timestamp(start_date)
        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []
        for target_date in test_dates:
            train_df = df[df.index < target_date].copy()
            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
                continue
            try:
                model = RidgeExtendedShockForecaster(alpha=self.alpha, use_macro=self.use_macro)
                model.fit(train_df, target_col)
                test_df = df[df.index <= target_date].copy()
                pred = model.predict(test_df, target_date)
                actual = df.loc[target_date, target_col]
                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred['prediction'],
                    'error': actual - pred['prediction'],
                    'pred_ridge': pred['pred_ridge'],
                    'has_macro': pred.get('has_macro', False)
                })
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        self._check_fitted()
        importance = pd.DataFrame({
            'feature': self._features,
            'coefficient': self.ridge.coef_
        })
        importance['abs_coef'] = importance['coefficient'].abs()
        importance['is_shock'] = importance['feature'].isin(self.SHOCK_DUMMIES)
        importance['is_macro'] = importance['feature'].isin(self.MACRO_FEATURES)
        return importance.sort_values('abs_coef', ascending=False)
