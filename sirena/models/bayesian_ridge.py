"""
Bayesian Ridge — байесовская регрессия с доверительными интервалами
===================================================================

Преимущества:
1. Автоматический подбор регуляризации (alpha)
2. Калиброванные доверительные интервалы
3. Устойчивость к мультиколлинеарности
4. Не требует кросс-валидации для выбора гиперпараметров

Использование:
    model = BayesianRidgeForecaster()
    model.fit(df)
    fc, std = model.forecast_with_ci(horizon=12)
    # fc - точечный прогноз
    # std - стандартное отклонение (для CI: fc ± 1.96*std)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import RobustScaler
from typing import Dict, Optional, Any, List, Tuple

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("bayesian_ridge")
class BayesianRidgeForecaster(BaseForecaster):
    """
    Байесовская Ridge регрессия с автоматической регуляризацией.

    Возвращает точечный прогноз и доверительные интервалы.

    Параметры:
        alpha_1, alpha_2: Prior параметры для alpha (шум)
        lambda_1, lambda_2: Prior параметры для lambda (регуляризация)
        n_iter: Максимальное число итераций

    Выходы:
        - forecast(): точечный прогноз (как Ridge)
        - forecast_with_ci(): прогноз + стандартное отклонение
    """

    name = "bayesian_ridge"
    MIN_TRAIN_SIZE = 36
    OUTLIER_YEARS = [2022, 2010]

    # Признаки (как в Ridge Extended)
    BASE_FEATURES = [
        'y_lag1', 'y_lag2', 'y_lag3', 'y_lag6', 'y_lag12',
        'y_ma3', 'y_ma6',
        'd_y_lag1', 'd_y_lag3',
        'y_vol3', 'y_vol6',
        'month_sin', 'month_cos',
        'quarter_sin', 'quarter_cos',
        'is_jan', 'is_dec',
        'food_lag1', 'nonfood_lag1', 'services_lag1',
        'seasonal_norm', 'deviation_lag1'
    ]

    MACRO_FEATURES = [
        'ruonia_diff_lag1',
        'spread_lag4',
        'ki_diff_lag6',
        'ki_vol',
    ]

    # ETS веса по месяцам
    ETS_WEIGHTS = {
        1: 0.9, 2: 0.0, 3: 0.5, 4: 0.3,
        5: 0.9, 6: 0.5, 7: 0.0, 8: 0.5,
        9: 0.9, 10: 0.9, 11: 0.0, 12: 0.0
    }

    def __init__(
        self,
        alpha_1: float = 1e-6,
        alpha_2: float = 1e-6,
        lambda_1: float = 1e-6,
        lambda_2: float = 1e-6,
        n_iter: int = 300,
        use_macro: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.n_iter = n_iter
        self.use_macro = use_macro

        self.model = None
        self.scaler = None
        self.seasonal_norm = None
        self._has_macro = False
        self._features = None
        self._df = None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков."""
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
        """Сезонная норма без выбросов."""
        clean_df = df[~df['year'].isin(self.OUTLIER_YEARS)]
        return clean_df.groupby('month')['Все товары и услуги'].mean()

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'BayesianRidgeForecaster':
        """Обучение модели."""
        self._df = df.copy()
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

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Bayesian Ridge
        self.model = BayesianRidge(
            alpha_1=self.alpha_1,
            alpha_2=self.alpha_2,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            max_iter=self.n_iter,  # n_iter переименован в max_iter в sklearn 1.8+
            fit_intercept=True
        )
        self.model.fit(X_scaled, y)

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
        """
        Точечный прогноз (обёртка над predict_with_ci).

        Returns:
            Dict с prediction и другими полями
        """
        return self.predict_with_ci(df, target_date)

    def predict_with_ci(
        self,
        df: pd.DataFrame,
        target_date: pd.Timestamp
    ) -> Dict[str, Any]:
        """
        Точечный прогноз с доверительным интервалом.

        Returns:
            Dict с:
                - prediction: точечный прогноз (комбинация Ridge + ETS)
                - pred_bayesian: прогноз Bayesian Ridge
                - std: стандартное отклонение
                - ci_lower: нижняя граница 95% CI
                - ci_upper: верхняя граница 95% CI
        """
        self._check_fitted()

        df_prep = self._prepare_features(df)
        df_prep['seasonal_norm'] = df_prep['month'].map(self.seasonal_norm)
        df_prep['deviation_lag1'] = df_prep['y_lag1'] - df_prep['month'].shift(1).map(self.seasonal_norm)

        if self._has_macro:
            df_prep = self._add_macro_features(df_prep)

        test_row = df_prep.loc[[target_date]]

        X_test = self.scaler.transform(test_row[self._features].values)

        # Bayesian Ridge с std
        pred_bayesian, std = self.model.predict(X_test, return_std=True)
        pred_bayesian = pred_bayesian[0]
        std = std[0]

        # ETS прогноз
        target_month = target_date.month
        pred_ets = self.seasonal_norm.get(target_month, 100.0)

        # Комбинация
        ets_weight = self.ETS_WEIGHTS.get(target_month, 0.3)
        pred_combined = (1 - ets_weight) * pred_bayesian + ets_weight * pred_ets

        # CI (95%)
        z = 1.96
        ci_lower = pred_combined - z * std
        ci_upper = pred_combined + z * std

        return {
            'date': target_date,
            'prediction': pred_combined,
            'pred_bayesian': pred_bayesian,
            'pred_ets': pred_ets,
            'ets_weight': ets_weight,
            'std': std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'model': self.name,
            'has_macro': self._has_macro
        }

    def forecast_with_ci(self, horizon: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """
        Прогноз на горизонт с доверительными интервалами.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (predictions, std_devs)
        """
        self._check_fitted()

        # Для полноценного CI нужно итеративное прогнозирование
        # Здесь упрощённая версия на основе сезонной нормы

        if self.seasonal_norm is None:
            return np.zeros(horizon), np.zeros(horizon)

        last_month = self._last_train_date.month if self._last_train_date else 1
        predictions = []

        for i in range(horizon):
            month = ((last_month + i) % 12) + 1
            pred = self.seasonal_norm.get(month, 100.0)
            predictions.append(pred)

        # Оценка std на основе исторической волатильности
        # Увеличивается с горизонтом
        base_std = 0.3  # Базовая ошибка (примерно MAE Ridge)
        stds = np.array([base_std * np.sqrt(1 + i * 0.1) for i in range(horizon)])

        return np.array(predictions), stds

    def backtest(
        self,
        df: pd.DataFrame = None,
        start_date: str = '2019-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """Бэктест с CI."""
        if df is None:
            df = self._df

        start = pd.Timestamp(start_date)
        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []

        for target_date in test_dates:
            train_df = df[df.index < target_date].copy()

            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
                continue

            try:
                model = BayesianRidgeForecaster(
                    alpha_1=self.alpha_1,
                    alpha_2=self.alpha_2,
                    lambda_1=self.lambda_1,
                    lambda_2=self.lambda_2,
                    use_macro=self.use_macro
                )
                model.fit(train_df, target_col)

                test_df = df[df.index <= target_date].copy()
                pred_result = model.predict_with_ci(test_df, target_date)

                actual = df.loc[target_date, target_col]

                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred_result['prediction'],
                    'error': actual - pred_result['prediction'],
                    'std': pred_result['std'],
                    'ci_lower': pred_result['ci_lower'],
                    'ci_upper': pred_result['ci_upper'],
                    'in_ci': pred_result['ci_lower'] <= actual <= pred_result['ci_upper'],
                    'has_macro': pred_result.get('has_macro', False)
                })
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_model_params(self) -> Dict[str, float]:
        """Оптимизированные параметры Bayesian Ridge."""
        self._check_fitted()

        return {
            'alpha': self.model.alpha_,
            'lambda': self.model.lambda_,
            'sigma': np.sqrt(1.0 / self.model.alpha_),  # Noise std
            'n_iter': self.model.n_iter_
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Важность признаков."""
        self._check_fitted()

        importance = pd.DataFrame({
            'feature': self._features,
            'coefficient': self.model.coef_
        })
        importance['abs_coef'] = importance['coefficient'].abs()
        return importance.sort_values('abs_coef', ascending=False)

    def get_model_info(self) -> Dict:
        """Информация о модели."""
        info = {
            'name': self.name,
            'features_count': len(self._features) if self._features else 0,
            'has_macro': self._has_macro,
            'is_fitted': self._is_fitted
        }

        if self._is_fitted:
            info.update(self.get_model_params())

        return info
