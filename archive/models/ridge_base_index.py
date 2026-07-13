"""
Ridge на базисных индексах
==========================

Эксперимент 1 из методик ЦБ:
Вместо обучения Ridge на MoM индексах, обучаем на базисных (кумулятивных)
индексах и конвертируем прогноз обратно в MoM.

Формулы преобразования:
- MoM -> Base: base[i] = cumprod(MoM/100) * 100
- Base -> MoM: MoM[i] = base[i] / base[i-1] * 100

Ожидание: более гладкий ряд может улучшить качество прогноза.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from typing import Dict, Any, List

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("ridge_base_index")
class RidgeBaseIndexForecaster(BaseForecaster):
    """
    Ridge регрессия на базисных индексах.

    Особенности:
    - Преобразование MoM -> базисный индекс
    - Обучение Ridge на log(базисный индекс)
    - Конвертация прогноза обратно в MoM
    - Исключение выбросных лет (2022, 2010)
    """

    name = "ridge_base_index"
    MIN_TRAIN_SIZE = 36

    # Годы-выбросы
    OUTLIER_YEARS = [2022, 2010]

    # Ridge регуляризация
    ALPHA = 0.3

    # Признаки на базисных индексах
    FEATURES = [
        'base_lag1', 'base_lag2', 'base_lag12',
        'base_pct_lag1', 'base_pct_lag2',  # % изменение базисного индекса
        'base_ma3',
        'month_sin', 'month_cos',
        'food_base_lag1', 'nonfood_base_lag1', 'services_base_lag1',
        'seasonal_base_norm', 'base_deviation_lag1'
    ]

    MACRO_FEATURES = [
        'ruonia_diff_lag1',
        'spread_lag4',
        'ki_diff_lag6',
        'ki_vol',
    ]

    def __init__(self, alpha: float = None, use_macro: bool = True, use_log: bool = False, **kwargs):
        """
        Args:
            alpha: Ridge регуляризация (по умолчанию 0.3)
            use_macro: Использовать макро-признаки Ki/Ruonia
            use_log: Логарифмировать базисные индексы (эксперимент 4)
        """
        super().__init__(**kwargs)
        self.alpha = alpha or self.ALPHA
        self.use_macro = use_macro
        self.use_log = use_log
        self.ridge = None
        self.scaler = None
        self.seasonal_base_norm = None
        self._has_macro = False
        self._features = None
        self._base_start_value = None  # Начальное значение базисного индекса

    def _mom_to_base(self, mom_series: pd.Series) -> pd.Series:
        """
        Преобразование MoM индексов в базисные.

        Formula: base[i] = cumprod(MoM/100) * 100
        """
        # MoM обычно в формате ~100 (100.5 = +0.5%)
        # base = накопительное произведение
        return (mom_series / 100).cumprod() * 100

    def _base_to_mom(self, base_series: pd.Series) -> pd.Series:
        """
        Преобразование базисных индексов обратно в MoM.

        Formula: MoM[i] = base[i] / base[i-1] * 100
        """
        return base_series / base_series.shift(1) * 100

    def _prepare_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков на базисных индексах."""
        df = df.copy()

        df['month'] = df.index.month
        df['year'] = df.index.year

        # Преобразуем MoM в базисные индексы
        df['base'] = self._mom_to_base(df['Все товары и услуги'])

        # Если логарифмируем
        if self.use_log:
            df['base'] = np.log(df['base'])

        # Лаги базисного индекса
        df['base_lag1'] = df['base'].shift(1)
        df['base_lag2'] = df['base'].shift(2)
        df['base_lag12'] = df['base'].shift(12)

        # Процентное изменение базисного индекса (это и есть ~MoM, но через базис)
        df['base_pct'] = df['base'].pct_change() * 100
        df['base_pct_lag1'] = df['base_pct'].shift(1)
        df['base_pct_lag2'] = df['base_pct'].shift(2)

        # Скользящее среднее базисного индекса
        df['base_ma3'] = df['base'].rolling(3).mean().shift(1)

        # Сезонные признаки
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Базисные индексы компонентов
        if 'Продовольственные товары' in df.columns:
            food_base = self._mom_to_base(df['Продовольственные товары'])
            if self.use_log:
                food_base = np.log(food_base)
            df['food_base_lag1'] = food_base.shift(1)
        else:
            df['food_base_lag1'] = df['base_lag1']

        if 'Непродовольственные товары' in df.columns:
            nonfood_base = self._mom_to_base(df['Непродовольственные товары'])
            if self.use_log:
                nonfood_base = np.log(nonfood_base)
            df['nonfood_base_lag1'] = nonfood_base.shift(1)
        else:
            df['nonfood_base_lag1'] = df['base_lag1']

        if 'Услуги' in df.columns:
            services_base = self._mom_to_base(df['Услуги'])
            if self.use_log:
                services_base = np.log(services_base)
            df['services_base_lag1'] = services_base.shift(1)
        else:
            df['services_base_lag1'] = df['base_lag1']

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

    def _compute_seasonal_base_norm(self, df: pd.DataFrame) -> pd.Series:
        """Вычисление сезонной нормы изменения базисного индекса."""
        clean_df = df[~df['year'].isin(self.OUTLIER_YEARS)]
        return clean_df.groupby('month')['base_pct'].mean()

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'RidgeBaseIndexForecaster':
        """Обучение модели на базисных индексах."""
        series = self._validate_data(df, target_col)

        df_prep = self._prepare_base_features(df)

        # Сезонная норма (изменения базисного индекса)
        self.seasonal_base_norm = self._compute_seasonal_base_norm(df_prep)

        # Добавляем сезонные признаки
        df_prep['seasonal_base_norm'] = df_prep['month'].map(self.seasonal_base_norm)
        df_prep['base_deviation_lag1'] = df_prep['base_pct_lag1'] - df_prep['month'].shift(1).map(self.seasonal_base_norm)

        # Определяем список признаков
        self._features = self.FEATURES.copy()

        # Макро-признаки
        self._has_macro = False
        if self.use_macro and 'Ki' in df.columns and 'Ruonia' in df.columns:
            df_prep = self._add_macro_features(df_prep)
            available_macro = [f for f in self.MACRO_FEATURES if f in df_prep.columns]
            if available_macro:
                self._features.extend(available_macro)
                self._has_macro = True

        # Исключаем выбросные годы
        train_df = df_prep[~df_prep['year'].isin(self.OUTLIER_YEARS)]

        # Целевая переменная - изменение базисного индекса (base_pct)
        target = 'base_pct'
        train_clean = train_df.dropna(subset=self._features + [target])

        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных: {len(train_clean)} < {self.MIN_TRAIN_SIZE}")

        # Обучение
        X = train_clean[self._features].values
        y = train_clean[target].values

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.ridge = Ridge(alpha=self.alpha)
        self.ridge.fit(X_scaled, y)

        # Сохраняем последнее значение базисного индекса для прогноза
        self._base_start_value = df_prep['base'].iloc[-1]

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Точечный прогноз на дату.

        Возвращает прогноз в MoM формате (100.X).
        """
        self._check_fitted()

        df_prep = self._prepare_base_features(df)
        df_prep['seasonal_base_norm'] = df_prep['month'].map(self.seasonal_base_norm)
        df_prep['base_deviation_lag1'] = df_prep['base_pct_lag1'] - df_prep['month'].shift(1).map(self.seasonal_base_norm)

        if self._has_macro:
            df_prep = self._add_macro_features(df_prep)

        test_row = df_prep.loc[[target_date]]

        # Ridge прогноз изменения базисного индекса
        X_test = self.scaler.transform(test_row[self._features].values)
        pred_base_pct = self.ridge.predict(X_test)[0]

        # Преобразуем в MoM
        # base_pct ≈ (base[t] / base[t-1] - 1) * 100
        # Это эквивалентно log-разности для малых изменений
        # MoM = base_pct + 100 (примерно)

        # Более точно: MoM = (1 + base_pct/100) * 100
        if self.use_log:
            # Если логарифмировали: base_pct = d(log(base))*100
            # MoM = exp(base_pct/100) * 100
            pred_mom = np.exp(pred_base_pct / 100) * 100
        else:
            # Без логарифма: base_pct = (base[t]/base[t-1] - 1)*100
            # Это просто MoM - 100, значит MoM = base_pct + 100
            pred_mom = pred_base_pct + 100

        # ETS компонента (сезонная норма)
        target_month = target_date.month
        pred_ets = self.seasonal_base_norm.get(target_month, 0.0) + 100

        # Простая комбинация (можно настроить веса)
        ets_weight = 0.3
        pred_combined = (1 - ets_weight) * pred_mom + ets_weight * pred_ets

        return {
            'date': target_date,
            'prediction': pred_combined,
            'pred_ridge_base': pred_base_pct,
            'pred_mom': pred_mom,
            'pred_ets': pred_ets,
            'model': self.name,
            'has_macro': self._has_macro,
            'use_log': self.use_log
        }

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Прогноз на горизонт."""
        self._check_fitted()

        if self.seasonal_base_norm is None:
            return np.zeros(horizon)

        last_month = self._last_train_date.month if self._last_train_date else 1
        predictions = []

        for i in range(horizon):
            month = ((last_month + i) % 12) + 1
            base_pct = self.seasonal_base_norm.get(month, 0.0)
            pred_mom = base_pct + 100  # Приблизительная конвертация
            predictions.append(pred_mom)

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
                model = RidgeBaseIndexForecaster(
                    alpha=self.alpha,
                    use_macro=self.use_macro,
                    use_log=self.use_log
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
                    'pred_base_pct': pred_result['pred_ridge_base'],
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
