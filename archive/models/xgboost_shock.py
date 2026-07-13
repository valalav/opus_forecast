"""
XGBoost + Shock Dummies
=======================

Эксперимент v4.6: XGBoost с shock dummy переменными из методик ЦБ.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Dict, Any
import warnings

from .base import BaseForecaster
from .registry import ModelRegistry

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


@ModelRegistry.register("xgboost_shock")
class XGBoostShockForecaster(BaseForecaster):
    """XGBoost с shock dummies."""

    name = "xgboost_shock"
    MIN_TRAIN_SIZE = 36

    # НЕ исключаем 2022 — используем shock dummies
    OUTLIER_YEARS = [2010]

    N_ESTIMATORS = 150
    MAX_DEPTH = 3
    LEARNING_RATE = 0.05
    REG_ALPHA = 0.5
    REG_LAMBDA = 1.0
    SUBSAMPLE = 0.8
    COLSAMPLE_BYTREE = 0.8

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

    def __init__(self, n_estimators: int = None, max_depth: int = None,
                 learning_rate: float = None, **kwargs):
        super().__init__(**kwargs)

        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost не установлен")

        self.n_estimators = n_estimators or self.N_ESTIMATORS
        self.max_depth = max_depth or self.MAX_DEPTH
        self.learning_rate = learning_rate or self.LEARNING_RATE

        self.model = None
        self.scaler = None
        self.seasonal_norm = None
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

    def _compute_seasonal_norm(self, df: pd.DataFrame) -> pd.Series:
        """Сезонная норма (исключаем 2022 при расчёте)."""
        clean_df = df[df['year'] != 2022]
        return clean_df.groupby('month')['Все товары и услуги'].mean()

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'XGBoostShockForecaster':
        """Обучение модели."""
        self._validate_data(df, target_col)

        df_prep = self._prepare_features(df)
        self.seasonal_norm = self._compute_seasonal_norm(df_prep)

        df_prep['seasonal_norm'] = df_prep['month'].map(self.seasonal_norm)
        df_prep['deviation_lag1'] = df_prep['y_lag1'] - df_prep['month'].shift(1).map(self.seasonal_norm)

        self._features = self.BASE_FEATURES.copy() + self.SHOCK_DUMMIES

        # Исключаем только 2010 (2022 обрабатывается через shock dummies)
        train_df = df_prep[~df_prep['year'].isin(self.OUTLIER_YEARS)]
        train_clean = train_df.dropna(subset=self._features + [target_col])

        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных: {len(train_clean)} < {self.MIN_TRAIN_SIZE}")

        X = train_clean[self._features].values
        y = train_clean[target_col].values

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                reg_alpha=self.REG_ALPHA,
                reg_lambda=self.REG_LAMBDA,
                subsample=self.SUBSAMPLE,
                colsample_bytree=self.COLSAMPLE_BYTREE,
                objective='reg:squarederror',
                random_state=42,
                verbosity=0
            )
            self.model.fit(X_scaled, y)

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Прогноз."""
        self._check_fitted()

        df_prep = self._prepare_features(df)
        df_prep['seasonal_norm'] = df_prep['month'].map(self.seasonal_norm)
        df_prep['deviation_lag1'] = df_prep['y_lag1'] - df_prep['month'].shift(1).map(self.seasonal_norm)

        test_row = df_prep.loc[[target_date]]
        X_test = self.scaler.transform(test_row[self._features].values)

        pred_xgb = self.model.predict(X_test)[0]

        target_month = target_date.month
        pred_ets = self.seasonal_norm.get(target_month, 100.0)

        ets_weight = self.ETS_WEIGHTS.get(target_month, 0.3)
        pred_combined = (1 - ets_weight) * pred_xgb + ets_weight * pred_ets

        return {
            'date': target_date,
            'prediction': pred_combined,
            'pred_xgboost': pred_xgb,
            'pred_ets': pred_ets,
            'model': self.name
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
                model = XGBoostShockForecaster(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate)
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
            'importance': self.model.feature_importances_
        })
        importance['is_shock'] = importance['feature'].isin(self.SHOCK_DUMMIES)
        return importance.sort_values('importance', ascending=False)
