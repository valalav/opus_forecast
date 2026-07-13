"""
Ridge Extended на базисных индексах
===================================

Эксперимент v4.6: Ridge Extended на базисных (кумулятивных) индексах.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from typing import Dict, Any

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("ridge_extended_base")
class RidgeExtendedBaseIndexForecaster(BaseForecaster):
    """Ridge Extended на базисных индексах."""

    name = "ridge_extended_base"
    MIN_TRAIN_SIZE = 36

    OUTLIER_YEARS = [2010, 2022]
    ALPHA = 0.3

    ETS_WEIGHTS = {
        1: 0.9, 2: 0.0, 3: 0.5, 4: 0.3,
        5: 0.9, 6: 0.5, 7: 0.0, 8: 0.5,
        9: 0.9, 10: 0.9, 11: 0.0, 12: 0.0
    }

    BASE_FEATURES = [
        'base_lag1', 'base_lag2', 'base_lag12',
        'base_lag3', 'base_lag6',
        'base_pct_lag1', 'base_pct_lag2',
        'base_ma3', 'base_ma6',
        'd_base_pct_lag1', 'd_base_pct_lag3',
        'base_vol3', 'base_vol6',
        'month_sin', 'month_cos',
        'quarter_sin', 'quarter_cos',
        'is_jan', 'is_dec',
        'is_tariff_month', 'is_q1', 'is_summer',
        'food_base_lag1', 'nonfood_base_lag1', 'services_base_lag1',
        'seasonal_base_norm', 'base_deviation_lag1',
    ]

    def __init__(self, alpha: float = None, use_log: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha or self.ALPHA
        self.use_log = use_log
        self.ridge = None
        self.scaler = None
        self.seasonal_base_norm = None
        self._features = None

    def _mom_to_base(self, mom_series: pd.Series) -> pd.Series:
        return (mom_series / 100).cumprod() * 100

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['month'] = df.index.month
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter

        df['base'] = self._mom_to_base(df['Все товары и услуги'])
        if self.use_log:
            df['base'] = np.log(df['base'])

        # Лаги
        df['base_lag1'] = df['base'].shift(1)
        df['base_lag2'] = df['base'].shift(2)
        df['base_lag3'] = df['base'].shift(3)
        df['base_lag6'] = df['base'].shift(6)
        df['base_lag12'] = df['base'].shift(12)

        # % изменение
        df['base_pct'] = df['base'].pct_change() * 100
        df['base_pct_lag1'] = df['base_pct'].shift(1)
        df['base_pct_lag2'] = df['base_pct'].shift(2)

        # MA
        df['base_ma3'] = df['base'].rolling(3).mean().shift(1)
        df['base_ma6'] = df['base'].rolling(6).mean().shift(1)

        # Momentum
        df['d_base_pct_lag1'] = df['base_pct'].shift(1) - df['base_pct'].shift(2)
        df['d_base_pct_lag3'] = df['base_pct'].shift(1) - df['base_pct'].shift(4)

        # Volatility
        df['base_vol3'] = df['base_pct'].rolling(3).std().shift(1)
        df['base_vol6'] = df['base_pct'].rolling(6).std().shift(1)

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

        # Компоненты
        for col, name in [('Продовольственные товары', 'food'),
                          ('Непродовольственные товары', 'nonfood'),
                          ('Услуги', 'services')]:
            if col in df.columns:
                comp_base = self._mom_to_base(df[col])
                if self.use_log:
                    comp_base = np.log(comp_base)
                df[f'{name}_base_lag1'] = comp_base.shift(1)
            else:
                df[f'{name}_base_lag1'] = df['base_lag1']

        return df

    def _compute_seasonal_norm(self, df: pd.DataFrame) -> pd.Series:
        clean_df = df[~df['year'].isin(self.OUTLIER_YEARS)]
        return clean_df.groupby('month')['base_pct'].mean()

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'RidgeExtendedBaseIndexForecaster':
        self._validate_data(df, target_col)

        df_prep = self._prepare_features(df)
        self.seasonal_base_norm = self._compute_seasonal_norm(df_prep)

        df_prep['seasonal_base_norm'] = df_prep['month'].map(self.seasonal_base_norm)
        df_prep['base_deviation_lag1'] = df_prep['base_pct_lag1'] - df_prep['month'].shift(1).map(self.seasonal_base_norm)

        self._features = self.BASE_FEATURES.copy()

        target = 'base_pct'
        train_df = df_prep[~df_prep['year'].isin(self.OUTLIER_YEARS)]
        train_clean = train_df.dropna(subset=self._features + [target])

        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных: {len(train_clean)} < {self.MIN_TRAIN_SIZE}")

        X = train_clean[self._features].values
        y = train_clean[target].values

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(X_scaled, y)

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        self._check_fitted()

        df_prep = self._prepare_features(df)
        df_prep['seasonal_base_norm'] = df_prep['month'].map(self.seasonal_base_norm)
        df_prep['base_deviation_lag1'] = df_prep['base_pct_lag1'] - df_prep['month'].shift(1).map(self.seasonal_base_norm)

        test_row = df_prep.loc[[target_date]]
        X_test = self.scaler.transform(test_row[self._features].values)

        pred_base_pct = self.ridge.predict(X_test)[0]

        if self.use_log:
            pred_mom = np.exp(pred_base_pct / 100) * 100
        else:
            pred_mom = pred_base_pct + 100

        target_month = target_date.month
        pred_ets = self.seasonal_base_norm.get(target_month, 0.0) + 100

        ets_weight = self.ETS_WEIGHTS.get(target_month, 0.3)
        pred_combined = (1 - ets_weight) * pred_mom + ets_weight * pred_ets

        return {
            'date': target_date,
            'prediction': pred_combined,
            'pred_ridge': pred_mom,
            'pred_ets': pred_ets,
            'model': self.name
        }

    def forecast(self, horizon: int = 12) -> np.ndarray:
        self._check_fitted()
        last_month = self._last_train_date.month if self._last_train_date else 1
        return np.array([self.seasonal_base_norm.get(((last_month + i) % 12) + 1, 0.0) + 100 for i in range(horizon)])

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
                model = RidgeExtendedBaseIndexForecaster(alpha=self.alpha, use_log=self.use_log)
                model.fit(train_df, target_col)
                test_df = df[df.index <= target_date].copy()
                pred = model.predict(test_df, target_date)
                actual = df.loc[target_date, target_col]
                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred['prediction'],
                    'error': actual - pred['prediction']
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
        return importance.sort_values('abs_coef', ascending=False)
