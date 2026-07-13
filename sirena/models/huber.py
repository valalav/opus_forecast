"""
Huber Regressor — робастная модель к выбросам
=============================================

HuberRegressor использует Huber loss вместо MSE:
- Для малых ошибок (< epsilon): квадратичная функция потерь (как MSE)
- Для больших ошибок (> epsilon): линейная функция потерь

Преимущества:
- Не нужно вручную исключать 2022 — модель сама понизит вес выбросов
- Автоматическая адаптация к "тяжёлым хвостам"
- Стабильнее Ridge на нестационарных данных

Параметры:
- epsilon: порог для переключения loss (default 1.35)
- alpha: L2 регуляризация (как Ridge)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import RobustScaler
from typing import Dict, Optional, Any, List

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("huber")
class HuberForecaster(BaseForecaster):
    """
    Huber Regressor — робастная к выбросам модель.

    Автоматически понижает влияние выбросов (2022 и др.)
    без необходимости исключать их из обучения.
    """

    name = "huber"
    MIN_TRAIN_SIZE = 36
    # НЕ исключаем годы — Huber сам справится с выбросами
    OUTLIER_YEARS = []

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
        epsilon: float = 1.35,
        alpha: float = 0.3,
        max_iter: int = 500,
        use_macro: bool = True,
        **kwargs
    ):
        """
        Args:
            epsilon: Порог для переключения loss функции (default 1.35)
                     Меньше = более робастный к выбросам
            alpha: L2 регуляризация (как в Ridge)
            max_iter: Максимум итераций для сходимости
            use_macro: Использовать макро-признаки (Ki, Ruonia)
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iter = max_iter
        self.use_macro = use_macro
        self.model = None
        self.scaler = None
        self.seasonal_norm = None
        self._has_macro = False
        self._features = None
        self._outliers_detected = 0

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
        """Вычисление сезонной нормы (включая все годы — Huber справится)."""
        # Для Huber не исключаем годы, но для сезонной нормы лучше исключить
        # чтобы не искажать baseline
        clean_df = df[~df['year'].isin([2010, 2022])]
        return clean_df.groupby('month')['Все товары и услуги'].mean()

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'HuberForecaster':
        """Обучение HuberRegressor (включает ВСЕ данные, включая выбросы)."""
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

        # НЕ исключаем выбросные годы — Huber сам справится
        train_clean = df_prep.dropna(subset=self._features + [target_col])

        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных: {len(train_clean)} < {self.MIN_TRAIN_SIZE}")

        X = train_clean[self._features].values
        y = train_clean[target_col].values

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        # HuberRegressor
        self.model = HuberRegressor(
            epsilon=self.epsilon,
            alpha=self.alpha,
            max_iter=self.max_iter,
            warm_start=False
        )
        self.model.fit(X_scaled, y)

        # Подсчёт выбросов (где Huber использовал линейный loss)
        # outliers — точки с |residual| > epsilon * scale
        residuals = y - self.model.predict(X_scaled)
        self._outliers_detected = np.sum(np.abs(residuals) > self.epsilon * self.model.scale_)

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
        pred_huber = self.model.predict(X_test)[0]

        target_month = target_date.month
        pred_ets = self.seasonal_norm.get(target_month, 100.0)

        ets_weight = self.ETS_WEIGHTS.get(target_month, 0.3)
        pred_combined = (1 - ets_weight) * pred_huber + ets_weight * pred_ets

        return {
            'date': target_date,
            'prediction': pred_combined,
            'pred_huber': pred_huber,
            'pred_ets': pred_ets,
            'ets_weight': ets_weight,
            'model': self.name,
            'has_macro': self._has_macro,
            'scale': self.model.scale_
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
                model = HuberForecaster(
                    epsilon=self.epsilon,
                    alpha=self.alpha,
                    max_iter=self.max_iter,
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
                    'pred_huber': pred_result['pred_huber'],
                    'scale': pred_result.get('scale'),
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
            'coefficient': self.model.coef_
        })
        importance['abs_coef'] = importance['coefficient'].abs()
        importance['is_macro'] = importance['feature'].isin(self.MACRO_FEATURES)

        return importance.sort_values('abs_coef', ascending=False)

    def get_model_info(self) -> Dict:
        """Информация о модели."""
        return {
            'name': self.name,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'scale': self.model.scale_ if self._is_fitted else None,
            'outliers_detected': self._outliers_detected,
            'features_count': len(self._features) if self._features else 0,
            'has_macro': self._has_macro,
            'is_fitted': self._is_fitted
        }
