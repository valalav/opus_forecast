"""
LightGBM + Shock Dummies
========================

Эксперимент v4.6: LightGBM с shock dummy переменными из методик ЦБ.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Dict, Any, List
import warnings

warnings.filterwarnings('ignore')

from .base import BaseForecaster
from .registry import ModelRegistry

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


@ModelRegistry.register("lightgbm_shock")
class LightGBMShockForecaster(BaseForecaster):
    """LightGBM с shock dummies."""

    name = "lightgbm_shock"
    MIN_TRAIN_SIZE = 36

    # НЕ исключаем 2022 — используем shock dummies
    OUTLIER_YEARS = [2010]

    N_ESTIMATORS = 100
    MAX_DEPTH = 5
    LEARNING_RATE = 0.1
    NUM_LEAVES = 31
    MIN_CHILD_SAMPLES = 10

    BASE_FEATURES = [
        'y_lag1', 'y_lag2', 'y_lag12', 'y_ma3',
        'month_sin', 'month_cos',
        'food_lag1', 'nonfood_lag1', 'services_lag1',
        'seasonal_norm', 'deviation_lag1'
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

    def __init__(self, n_estimators: int = None, max_depth: int = None,
                 learning_rate: float = None, use_macro: bool = False, **kwargs):
        super().__init__(**kwargs)

        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM не установлен")

        self.n_estimators = n_estimators or self.N_ESTIMATORS
        self.max_depth = max_depth or self.MAX_DEPTH
        self.learning_rate = learning_rate or self.LEARNING_RATE
        self.use_macro = use_macro

        self.model = None
        self.scaler = RobustScaler()
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

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков."""
        df = df.copy()

        df['month'] = df.index.month
        df['year'] = df.index.year

        y = df['Все товары и услуги']

        df['y_lag1'] = y.shift(1)
        df['y_lag2'] = y.shift(2)
        df['y_lag12'] = y.shift(12)
        df['y_ma3'] = y.rolling(3).mean().shift(1)

        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

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

    def _compute_seasonal_norm(self, df: pd.DataFrame) -> Dict[int, float]:
        """Сезонная норма (исключаем 2022 при расчёте)."""
        clean_df = df[df['year'] != 2022]
        return clean_df.groupby('month')['Все товары и услуги'].mean().to_dict()

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'LightGBMShockForecaster':
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

        X_scaled = self.scaler.fit_transform(X)

        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.NUM_LEAVES,
            min_child_samples=self.MIN_CHILD_SAMPLES,
            verbosity=-1,
            random_state=42
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

        if self._has_macro:
            df_prep = self._add_macro_features(df_prep)

        test_row = df_prep.loc[[target_date]]
        X_test = self.scaler.transform(test_row[self._features].values)

        pred = self.model.predict(X_test)[0]

        return {
            'date': target_date,
            'prediction': pred,
            'model': self.name,
            'has_macro': self._has_macro
        }

    def forecast(self, horizon: int = 12) -> np.ndarray:
        self._check_fitted()
        last_month = self._last_train_date.month if self._last_train_date else 1
        return np.array([self.seasonal_norm.get(((last_month + i) % 12) + 1, 100.5) - 100 for i in range(horizon)])

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
                model = LightGBMShockForecaster(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    use_macro=self.use_macro
                )
                model.fit(train_df, target_col)
                test_df = df[df.index <= target_date].copy()
                pred = model.predict(test_df, target_date)
                actual = df.loc[target_date, target_col]
                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred['prediction'],
                    'error': actual - pred['prediction'],
                    'has_macro': pred.get('has_macro', False)
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
        importance['is_macro'] = importance['feature'].isin(self.MACRO_FEATURES)
        return importance.sort_values('importance', ascending=False)
