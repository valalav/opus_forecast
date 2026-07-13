"""
Ridge модель для прогнозирования инфляции КБР
==============================================

Основная модель СИРЕНА-КБР v4.0 с Ridge регрессией,
комбинацией с ETS сезонной нормой и макро-признаками.

Вес в ансамбле: 40%

Новое в v4.0:
- Макро-признаки Ki и Ruonia (опционально)
- ΔRuonia_lag1 — самый сильный предиктор (r=0.477)
- Spread Ki-Ruonia — индикатор ликвидности
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from typing import Dict, Optional, Any, List

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("ridge")
class RidgeForecaster(BaseForecaster):
    """
    Ridge регрессия с сезонной ETS компонентой и макро-признаками.

    Особенности:
    - Лаговые признаки (1, 2, 12 месяцев)
    - Сезонные синус/косинус признаки
    - Комбинация с ETS по месячным весам
    - Исключение выбросных лет (2022, 2010)
    - Макро-признаки Ki и Ruonia (v4.0)
    """

    name = "ridge"
    MIN_TRAIN_SIZE = 36

    # Годы-выбросы
    OUTLIER_YEARS = [2022, 2010]

    # ETS веса по месяцам
    ETS_WEIGHTS = {
        1: 0.9, 2: 0.0, 3: 0.5, 4: 0.3,
        5: 0.9, 6: 0.5, 7: 0.0, 8: 0.5,
        9: 0.9, 10: 0.9, 11: 0.0, 12: 0.0
    }

    # Ridge регуляризация
    ALPHA = 0.3

    # Базовые признаки
    BASE_FEATURES = [
        'y_lag1', 'y_lag2', 'y_lag12', 'y_ma3',
        'month_sin', 'month_cos',
        'food_lag1', 'nonfood_lag1', 'services_lag1',
        'seasonal_norm', 'deviation_lag1'
    ]

    # Макро-признаки (добавляются если есть Ki и Ruonia)
    MACRO_FEATURES = [
        'ruonia_diff_lag1',  # r=0.477 — самый сильный!
        'spread_lag4',       # r=0.444
        'ki_diff_lag6',      # r=0.300
        'ki_vol',            # r=0.256
    ]

    def __init__(self, alpha: float = None, use_macro: bool = True, ets_weights: Dict[int, float] = None, **kwargs):
        """
        Инициализация модели.

        Args:
            alpha: Ridge регуляризация (по умолчанию 0.3)
            use_macro: Использовать макро-признаки Ki/Ruonia (по умолчанию True)
            ets_weights: Словарь весов сезонности {month: weight}. Если None, используются дефолтные.
        """
        super().__init__(**kwargs)
        self.alpha = alpha or self.ALPHA
        self.use_macro = use_macro
        self.ets_weights = ets_weights if ets_weights is not None else self.ETS_WEIGHTS
        self.ridge = None
        self.scaler = None
        self.seasonal_norm = None
        self._has_macro = False  # Будет True если данные содержат Ki и Ruonia
        self._features = None    # Актуальный список признаков

    @property
    def FEATURES(self) -> List[str]:
        """Динамический список признаков."""
        if self._features is not None:
            return self._features
        return self.BASE_FEATURES

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

        return df

    def _add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Добавить макро-признаки Ki и Ruonia.

        Признаки:
        - ruonia_diff_lag1: изменение RUONIA за месяц (lag 1) — r=0.477
        - spread_lag4: спред Ki-Ruonia (lag 4) — r=0.444
        - ki_diff_lag6: изменение Ki за месяц (lag 6) — r=0.300
        - ki_vol: волатильность Ki за 6 месяцев — r=0.256
        """
        df = df.copy()

        # Проверка наличия данных
        if 'Ki' not in df.columns or 'Ruonia' not in df.columns:
            return df

        # ΔRuonia с лагом 1 — самый сильный предиктор!
        df['ruonia_diff'] = df['Ruonia'].diff()
        df['ruonia_diff_lag1'] = df['ruonia_diff'].shift(1)

        # Спред Ki - Ruonia с лагом 4
        df['spread'] = df['Ki'] - df['Ruonia']
        df['spread_lag4'] = df['spread'].shift(4)

        # ΔKi с лагом 6 — долгосрочный эффект
        df['ki_diff'] = df['Ki'].diff()
        df['ki_diff_lag6'] = df['ki_diff'].shift(6)

        # Волатильность Ki за 6 месяцев
        df['ki_vol'] = df['Ki'].rolling(6).std().shift(1)

        # Заполняем NaN медианой для стабильности
        for col in self.MACRO_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        return df

    def _compute_seasonal_norm(self, df: pd.DataFrame) -> pd.Series:
        """Вычисление сезонной нормы без выбросных лет."""
        clean_df = df[~df['year'].isin(self.OUTLIER_YEARS)]
        return clean_df.groupby('month')['Все товары и услуги'].mean()

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'RidgeForecaster':
        """
        Обучение модели.

        Args:
            df: DataFrame с данными
            target_col: Целевая колонка

        Returns:
            self
        """
        # Валидация
        series = self._validate_data(df, target_col)

        # Подготовка признаков
        df_prep = self._prepare_features(df)

        # Сезонная норма
        self.seasonal_norm = self._compute_seasonal_norm(df_prep)

        # Добавляем сезонные признаки
        df_prep['seasonal_norm'] = df_prep['month'].map(self.seasonal_norm)
        df_prep['deviation_lag1'] = df_prep['y_lag1'] - df_prep['month'].shift(1).map(self.seasonal_norm)

        # Определяем список признаков
        self._features = self.BASE_FEATURES.copy()

        # Добавляем макро-признаки если включены и есть данные
        self._has_macro = False
        if self.use_macro and 'Ki' in df.columns and 'Ruonia' in df.columns:
            df_prep = self._add_macro_features(df_prep)
            # Проверяем что признаки созданы
            available_macro = [f for f in self.MACRO_FEATURES if f in df_prep.columns]
            if available_macro:
                self._features.extend(available_macro)
                self._has_macro = True

        # Исключаем выбросные годы
        train_df = df_prep[~df_prep['year'].isin(self.OUTLIER_YEARS)]

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

        # Сохраняем DataFrame для iterative_forecast
        self._train_df = df.copy()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Прогноз на горизонт через итеративный predict().

        Args:
            horizon: Количество месяцев

        Returns:
            numpy array с прогнозами (MoM в %)
        """
        self._check_fitted()

        # Используем iterative_forecast с сохранёнными данными
        if hasattr(self, '_train_df') and self._train_df is not None:
            target_col = getattr(self, '_target_col', 'Все товары и услуги')
            return self.iterative_forecast(self._train_df, horizon, target_col)

        # Fallback на сезонную норму (для обратной совместимости)
        if self.seasonal_norm is None:
            return np.zeros(horizon)

        last_month = self._last_train_date.month if self._last_train_date else 1
        predictions = []

        for i in range(horizon):
            month = ((last_month + i) % 12) + 1
            pred = self.seasonal_norm.get(month, 100.0) - 100  # MoM%
            predictions.append(pred)

        return np.array(predictions)

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Точечный прогноз на дату.

        Args:
            df: DataFrame с данными
            target_date: Дата прогноза

        Returns:
            Dict с прогнозом и компонентами
        """
        self._check_fitted()

        df_prep = self._prepare_features(df)
        df_prep['seasonal_norm'] = df_prep['month'].map(self.seasonal_norm)
        df_prep['deviation_lag1'] = df_prep['y_lag1'] - df_prep['month'].shift(1).map(self.seasonal_norm)

        # Добавляем макро-признаки если использовались при обучении
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
        ets_weight = self.ets_weights.get(target_month, 0.3)
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
        """
        Бэктестирование модели.

        Args:
            df: DataFrame с данными
            start_date: Начало периода
            target_col: Целевая колонка

        Returns:
            DataFrame с результатами
        """
        start = pd.Timestamp(start_date)

        # Используем реальные даты из данных, а не генерируем
        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []

        for target_date in test_dates:
            # Cutoff — все данные до текущего месяца
            cutoff = target_date - pd.DateOffset(days=1)
            train_df = df[df.index < target_date].copy()

            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
                continue

            try:
                # Создаём новую модель для чистого бэктеста
                model = RidgeForecaster(
                    alpha=self.alpha, 
                    use_macro=self.use_macro,
                    ets_weights=self.ets_weights
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
                    'pred_ets': pred_result['pred_ets'],
                    'has_macro': pred_result.get('has_macro', False)
                })
            except Exception as e:
                print(f"Ridge Error at {target_date}: {e}")
                import traceback
                traceback.print_exc()
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
