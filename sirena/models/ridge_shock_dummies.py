"""
Ridge с фиктивными переменными для шоков
========================================

Эксперимент 3 из методик ЦБ:
Вместо исключения 2022 года целиком, используем dummy-переменные
для конкретных шоковых периодов (как в R-коде методик).

Шоковые периоды:
- Декабрь 2014 (валютный кризис)
- Январь 2015 (продолжение валютного шока)
- Июль 2017 (индексация тарифов ЖКХ) - уже есть через is_tariff_month
- Март 2022 (санкционный шок)
- Апрель 2022 (продолжение санкционного шока)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from typing import Dict, Any, List

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("ridge_shock_dummies")
class RidgeShockDummiesForecaster(BaseForecaster):
    """
    Ridge регрессия с dummy-переменными для шоков.

    Особенности:
    - НЕ исключаем выбросные годы (в отличие от базового Ridge)
    - Добавляем dummy-переменные для конкретных шоковых месяцев
    - Модель учится на шоках через коэффициенты dummy
    """

    name = "ridge_shock_dummies"
    MIN_TRAIN_SIZE = 36

    # НЕ исключаем годы - используем dummy вместо этого
    OUTLIER_YEARS = []  # Пустой список!

    ALPHA = 0.3

    # Базовые признаки
    BASE_FEATURES = [
        'y_lag1', 'y_lag2', 'y_lag12', 'y_ma3',
        'month_sin', 'month_cos',
        'food_lag1', 'nonfood_lag1', 'services_lag1',
        'seasonal_norm', 'deviation_lag1'
    ]

    # Dummy-переменные для шоков (из методик ЦБ)
    SHOCK_DUMMIES = [
        'is_shock_dec2014',
        'is_shock_jan2015',
        'is_tariff_jul2017',
        'is_shock_mar2022',
        'is_shock_apr2022',
        'is_shock_2022',  # Весь 2022 год как один шок
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

    def __init__(self, alpha: float = None, use_macro: bool = True, use_2022_dummy: bool = True, **kwargs):
        """
        Args:
            alpha: Ridge регуляризация
            use_macro: Использовать макро-признаки
            use_2022_dummy: Использовать dummy для 2022 года (иначе исключаем)
        """
        super().__init__(**kwargs)
        self.alpha = alpha or self.ALPHA
        self.use_macro = use_macro
        self.use_2022_dummy = use_2022_dummy
        self.ridge = None
        self.scaler = None
        self.seasonal_norm = None
        self._has_macro = False
        self._features = None

    def _add_shock_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавить dummy-переменные для шоковых периодов."""
        df = df.copy()

        # Декабрь 2014 - валютный шок
        df['is_shock_dec2014'] = ((df.index.year == 2014) & (df.index.month == 12)).astype(int)

        # Январь 2015 - продолжение валютного шока
        df['is_shock_jan2015'] = ((df.index.year == 2015) & (df.index.month == 1)).astype(int)

        # Июль 2017 - индексация тарифов ЖКХ (регулярный шок)
        df['is_tariff_jul2017'] = ((df.index.month == 7)).astype(int)

        # Март 2022 - санкционный шок
        df['is_shock_mar2022'] = ((df.index.year == 2022) & (df.index.month == 3)).astype(int)

        # Апрель 2022 - продолжение санкционного шока
        df['is_shock_apr2022'] = ((df.index.year == 2022) & (df.index.month == 4)).astype(int)

        # Весь 2022 год как один шок
        df['is_shock_2022'] = (df.index.year == 2022).astype(int)

        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков."""
        df = df.copy()

        df['month'] = df.index.month
        df['year'] = df.index.year

        # Лаги целевой переменной
        df['y_lag1'] = df['Все товары и услуги'].shift(1)
        df['y_lag2'] = df['Все товары и услуги'].shift(2)
        df['y_lag12'] = df['Все товары и услуги'].shift(12)

        # Скользящее среднее
        df['y_ma3'] = df['Все товары и услуги'].rolling(3).mean().shift(1)

        # Сезонные признаки
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Лаги компонентов
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
        """Вычисление сезонной нормы (исключаем только 2022 при расчёте нормы)."""
        # Даже если используем dummy, сезонную норму считаем без 2022
        clean_df = df[df['year'] != 2022]
        return clean_df.groupby('month')['Все товары и услуги'].mean()

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'RidgeShockDummiesForecaster':
        """Обучение модели с shock dummies."""
        series = self._validate_data(df, target_col)

        df_prep = self._prepare_features(df)

        # Сезонная норма
        self.seasonal_norm = self._compute_seasonal_norm(df_prep)

        df_prep['seasonal_norm'] = df_prep['month'].map(self.seasonal_norm)
        df_prep['deviation_lag1'] = df_prep['y_lag1'] - df_prep['month'].shift(1).map(self.seasonal_norm)

        # Определяем список признаков
        self._features = self.BASE_FEATURES.copy()

        # Добавляем shock dummies
        if self.use_2022_dummy:
            # Используем все shock dummies
            self._features.extend(self.SHOCK_DUMMIES)
        else:
            # Используем только dummies до 2022
            self._features.extend(['is_shock_dec2014', 'is_shock_jan2015', 'is_tariff_jul2017'])

        # Макро-признаки
        self._has_macro = False
        if self.use_macro and 'Ki' in df.columns and 'Ruonia' in df.columns:
            df_prep = self._add_macro_features(df_prep)
            available_macro = [f for f in self.MACRO_FEATURES if f in df_prep.columns]
            if available_macro:
                self._features.extend(available_macro)
                self._has_macro = True

        # Если НЕ используем 2022 dummy, исключаем 2022 год
        if not self.use_2022_dummy:
            train_df = df_prep[df_prep['year'] != 2022]
        else:
            # Используем все данные (2022 учитывается через dummy)
            train_df = df_prep

        # Очистка
        train_clean = train_df.dropna(subset=self._features + [target_col])

        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных: {len(train_clean)} < {self.MIN_TRAIN_SIZE}")

        # Обучение
        X = train_clean[self._features].values
        y = train_clean[target_col].values

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(X_scaled, y)

        self._is_fitted = True
        self._last_train_date = df.index.max()
        self._target_col = target_col
        self._train_df = df.copy()

        return self

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Точечный прогноз на дату."""
        self._check_fitted()

        df_prep = self._prepare_features(df)
        df_prep['seasonal_norm'] = df_prep['month'].map(self.seasonal_norm)
        df_prep['deviation_lag1'] = df_prep['y_lag1'] - df_prep['month'].shift(1).map(self.seasonal_norm)

        if self._has_macro:
            df_prep = self._add_macro_features(df_prep)

        test_row = df_prep.loc[[target_date]]

        # Ridge прогноз
        X_test = self.scaler.transform(test_row[self._features].values)
        pred_ridge = self.ridge.predict(X_test)[0]

        # ETS прогноз
        target_month = target_date.month
        pred_ets = self.seasonal_norm.get(target_month, 100.0)

        # Комбинация
        ets_weight = self.ETS_WEIGHTS.get(target_month, 0.3)
        pred_combined = (1 - ets_weight) * pred_ridge + ets_weight * pred_ets

        return {
            'date': target_date,
            'prediction': pred_combined,
            'pred_ridge': pred_ridge,
            'pred_ets': pred_ets,
            'ets_weight': ets_weight,
            'model': self.name,
            'has_macro': self._has_macro,
            'use_2022_dummy': self.use_2022_dummy
        }

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
                model = RidgeShockDummiesForecaster(
                    alpha=self.alpha,
                    use_macro=self.use_macro,
                    use_2022_dummy=self.use_2022_dummy
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
                    'pred_ridge': pred_result['pred_ridge'],
                    'has_macro': pred_result.get('has_macro', False)
                })
            except Exception as e:
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
        importance['is_shock'] = importance['feature'].isin(self.SHOCK_DUMMIES)
        importance['is_macro'] = importance['feature'].isin(self.MACRO_FEATURES)
        return importance.sort_values('abs_coef', ascending=False)

    def get_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Расчёт метрик качества."""
        if results.empty:
            return {'MAE': 0, 'RMSE': 0, 'KPI': 0}

        errors = results['error'].abs()
        mae = errors.mean()
        rmse = np.sqrt((results['error'] ** 2).mean())
        kpi = (errors <= 0.5).sum() / len(results) * 100

        return {
            'MAE': mae,
            'RMSE': rmse,
            'KPI': kpi
        }
