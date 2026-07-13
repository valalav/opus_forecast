"""
ElasticNet — модель с автоматическим feature selection
======================================================

ElasticNet = L1 (Lasso) + L2 (Ridge) регуляризация.

Преимущества:
- Автоматический отбор признаков через L1 (обнуляет неважные)
- Меньше переобучение при 23+ признаках на 150 точках
- ElasticNetCV автоматически подбирает alpha и l1_ratio

l1_ratio:
- 0 = чистый Ridge (L2)
- 1 = чистый Lasso (L1)
- 0.1-0.9 = смесь
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import RobustScaler
from typing import Dict, Optional, Any, List

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("elasticnet")
class ElasticNetForecaster(BaseForecaster):
    """
    ElasticNet с автоматическим подбором гиперпараметров.

    Использует ElasticNetCV для выбора оптимального alpha и l1_ratio
    через кросс-валидацию.
    """

    name = "elasticnet"
    MIN_TRAIN_SIZE = 36
    OUTLIER_YEARS = [2010]  # Только 2010 исключаем
    SAMPLE_WEIGHT_2022 = 0.25

    # ETS веса по месяцам
    ETS_WEIGHTS = {
        1: 0.9, 2: 0.0, 3: 0.5, 4: 0.3,
        5: 0.9, 6: 0.5, 7: 0.0, 8: 0.5,
        9: 0.9, 10: 0.9, 11: 0.0, 12: 0.0
    }

    # Признаки (как в Ridge Extended)
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
        'seasonal_norm', 'deviation_lag1'
    ]

    MACRO_FEATURES = [
        'ruonia_diff_lag1',
        'spread_lag4',
        'ki_diff_lag6',
        'ki_vol',
    ]

    def __init__(
        self,
        l1_ratios: List[float] = None,
        alphas: List[float] = None,
        cv: int = 5,
        use_macro: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.l1_ratios = l1_ratios or [0.1, 0.3, 0.5, 0.7, 0.9]
        self.alphas = alphas or [0.001, 0.01, 0.1, 0.3, 1.0]
        self.cv = cv
        self.use_macro = use_macro
        self.model = None
        self.scaler = None
        self.seasonal_norm = None
        self._has_macro = False
        self._features = None
        self._best_alpha = None
        self._best_l1_ratio = None

    @property
    def FEATURES(self) -> List[str]:
        if self._features is not None:
            return self._features
        return self.BASE_FEATURES

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков (как в Ridge Extended)."""
        df = df.copy()

        df['month'] = df.index.month
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter

        y = df['Все товары и услуги']

        # Лаги
        df['y_lag1'] = y.shift(1)
        df['y_lag2'] = y.shift(2)
        df['y_lag3'] = y.shift(3)
        df['y_lag6'] = y.shift(6)
        df['y_lag12'] = y.shift(12)

        # Скользящие средние
        df['y_ma3'] = y.rolling(3).mean().shift(1)
        df['y_ma6'] = y.rolling(6).mean().shift(1)

        # Momentum
        df['d_y_lag1'] = (y.shift(1) - y.shift(2))
        df['d_y_lag3'] = (y.shift(1) - y.shift(4))

        # Volatility
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

        # Компоненты
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
        """Вычисление сезонной нормы без выбросных лет."""
        clean_df = df[~df['year'].isin(self.OUTLIER_YEARS + [2022])]
        return clean_df.groupby('month')['Все товары и услуги'].mean()

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'ElasticNetForecaster':
        """Обучение ElasticNetCV."""
        series = self._validate_data(df, target_col)

        df_prep = self._prepare_features(df)
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

        train_df = df_prep[~df_prep['year'].isin(self.OUTLIER_YEARS)]
        train_clean = train_df.dropna(subset=self._features + [target_col])

        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных: {len(train_clean)} < {self.MIN_TRAIN_SIZE}")

        X = train_clean[self._features].values
        y = train_clean[target_col].values

        # Sample weighting
        sample_weights = np.ones(len(train_clean))
        years = train_clean['year'].values
        sample_weights[years == 2022] = self.SAMPLE_WEIGHT_2022

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        # ElasticNetCV с автоматическим подбором
        self.model = ElasticNetCV(
            l1_ratio=self.l1_ratios,
            alphas=self.alphas,
            cv=self.cv,
            max_iter=5000,
            random_state=42
        )
        self.model.fit(X_scaled, y, sample_weight=sample_weights)

        self._best_alpha = self.model.alpha_
        self._best_l1_ratio = self.model.l1_ratio_

        self._is_fitted = True
        self._last_train_date = df.index.max()
        self._target_col = target_col
        self._train_df = df.copy()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Прогноз на горизонт через итеративный predict()."""
        self._check_fitted()

        # Используем iterative_forecast с сохранёнными данными
        if hasattr(self, '_train_df') and self._train_df is not None:
            target_col = getattr(self, '_target_col', 'Все товары и услуги')
            return self.iterative_forecast(self._train_df, horizon, target_col)

        # Fallback на сезонную норму
        if self.seasonal_norm is None:
            return np.zeros(horizon)

        last_month = self._last_train_date.month if self._last_train_date else 1
        predictions = []

        for i in range(horizon):
            month = ((last_month + i) % 12) + 1
            pred = self.seasonal_norm.get(month, 100.0) - 100
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
        pred_elasticnet = self.model.predict(X_test)[0]

        target_month = target_date.month
        pred_ets = self.seasonal_norm.get(target_month, 100.0)

        ets_weight = self.ETS_WEIGHTS.get(target_month, 0.3)
        pred_combined = (1 - ets_weight) * pred_elasticnet + ets_weight * pred_ets

        return {
            'date': target_date,
            'prediction': pred_combined,
            'pred_elasticnet': pred_elasticnet,
            'pred_ets': pred_ets,
            'ets_weight': ets_weight,
            'model': self.name,
            'has_macro': self._has_macro,
            'best_alpha': self._best_alpha,
            'best_l1_ratio': self._best_l1_ratio
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
                model = ElasticNetForecaster(
                    l1_ratios=self.l1_ratios,
                    alphas=self.alphas,
                    cv=self.cv,
                    use_macro=self.use_macro
                )
                model.fit(train_df, target_col)

                test_df = df[df.index <= target_date].copy()
                pred_result = model.predict(test_df, target_date)

                actual = df.loc[target_date, target_col]

                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred_result['prediction'],
                    'error': actual - pred_result['prediction'],
                    'pred_elasticnet': pred_result['pred_elasticnet'],
                    'best_alpha': pred_result.get('best_alpha'),
                    'best_l1_ratio': pred_result.get('best_l1_ratio'),
                    'has_macro': pred_result.get('has_macro', False)
                })
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        """Важность признаков (ненулевые коэффициенты = отобранные)."""
        self._check_fitted()

        importance = pd.DataFrame({
            'feature': self._features,
            'coefficient': self.model.coef_
        })
        importance['abs_coef'] = importance['coefficient'].abs()
        importance['is_selected'] = importance['coefficient'] != 0
        importance['is_macro'] = importance['feature'].isin(self.MACRO_FEATURES)

        return importance.sort_values('abs_coef', ascending=False)

    def get_selected_features(self) -> List[str]:
        """Признаки, отобранные ElasticNet (с ненулевыми коэффициентами)."""
        self._check_fitted()
        importance = self.get_feature_importance()
        return importance[importance['is_selected']]['feature'].tolist()

    def get_model_info(self) -> Dict:
        """Информация о модели."""
        return {
            'name': self.name,
            'best_alpha': self._best_alpha,
            'best_l1_ratio': self._best_l1_ratio,
            'features_count': len(self._features) if self._features else 0,
            'selected_features_count': len(self.get_selected_features()) if self._is_fitted else 0,
            'has_macro': self._has_macro,
            'is_fitted': self._is_fitted
        }
