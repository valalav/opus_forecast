"""
CatBoost модель для прогнозирования инфляции КБР
================================================

CatBoost — градиентный бустинг от Яндекс, оптимизирован для:
- Работы с категориальными признаками (месяц)
- Малых выборок (~150 точек)
- Автоматической регуляризации

Преимущества перед LightGBM:
- Меньше переобучается на малых данных
- Ordered boosting для борьбы с target leakage
- Встроенная обработка категориальных признаков
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List
import warnings

from .base import BaseForecaster
from .registry import ModelRegistry

warnings.filterwarnings('ignore')


# Проверка наличия CatBoost
try:
    from catboost import CatBoostRegressor, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostRegressor = None
    Pool = None


@ModelRegistry.register("catboost")
class CatBoostForecaster(BaseForecaster):
    """
    CatBoost для прогнозирования инфляции.

    Особенности:
    - Ordered boosting (меньше переобучение)
    - Категориальный признак месяца
    - Агрессивная регуляризация для малых выборок
    - Ранняя остановка

    Parameters:
        iterations: Количество деревьев (default: 200)
        depth: Глубина дерева (default: 4, низкая для регуляризации)
        learning_rate: Шаг обучения (default: 0.05)
        l2_leaf_reg: L2 регуляризация (default: 5.0)
    """

    name = "catboost"
    MIN_TRAIN_SIZE = 36
    OUTLIER_YEARS = [2022, 2010]

    # Признаки
    NUMERICAL_FEATURES = [
        'y_lag1', 'y_lag2', 'y_lag3', 'y_lag6', 'y_lag12',
        'y_ma3', 'y_ma6',
        'y_diff1', 'y_diff3',  # momentum
        'y_vol3', 'y_vol6',    # volatility
        'food_lag1', 'nonfood_lag1', 'services_lag1',
        'month_sin', 'month_cos'
    ]

    CATEGORICAL_FEATURES = ['month', 'quarter']

    def __init__(
        self,
        iterations: int = 200,
        depth: int = 4,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 5.0,
        random_seed: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)

        if not CATBOOST_AVAILABLE:
            raise ImportError(
                "CatBoost не установлен. Установите: pip install catboost"
            )

        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed

        self.model = None
        self._df = None
        self._features = None
        self._cat_features_idx = None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков."""
        df = df.copy()

        # Временные признаки
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year

        # Лаги
        df['y_lag1'] = df['Все товары и услуги'].shift(1)
        df['y_lag2'] = df['Все товары и услуги'].shift(2)
        df['y_lag3'] = df['Все товары и услуги'].shift(3)
        df['y_lag6'] = df['Все товары и услуги'].shift(6)
        df['y_lag12'] = df['Все товары и услуги'].shift(12)

        # Скользящие средние
        df['y_ma3'] = df['Все товары и услуги'].rolling(3).mean().shift(1)
        df['y_ma6'] = df['Все товары и услуги'].rolling(6).mean().shift(1)

        # Momentum (первые разности)
        df['y_diff1'] = df['Все товары и услуги'].diff().shift(1)
        df['y_diff3'] = (df['Все товары и услуги'] - df['Все товары и услуги'].shift(3)).shift(1)

        # Volatility
        df['y_vol3'] = df['Все товары и услуги'].rolling(3).std().shift(1)
        df['y_vol6'] = df['Все товары и услуги'].rolling(6).std().shift(1)

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

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'Все товары и услуги'
    ) -> 'CatBoostForecaster':
        """
        Обучение модели.

        Args:
            df: DataFrame с данными
            target_col: Целевая колонка

        Returns:
            self
        """
        # Подготовка признаков
        df_prep = self._prepare_features(df)
        self._df = df.copy()

        # Исключаем выбросные годы
        train_df = df_prep[~df_prep['year'].isin(self.OUTLIER_YEARS)]

        # Определяем признаки
        self._features = self.NUMERICAL_FEATURES + self.CATEGORICAL_FEATURES

        # Индексы категориальных признаков
        self._cat_features_idx = [
            self._features.index(f) for f in self.CATEGORICAL_FEATURES
        ]

        # Очистка от NaN
        train_clean = train_df.dropna(subset=self._features + [target_col])

        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных: {len(train_clean)} < {self.MIN_TRAIN_SIZE}")

        # Подготовка данных
        X = train_clean[self._features].copy()
        y = train_clean[target_col].values

        # Преобразуем категориальные в строки (CatBoost требует)
        for cat_col in self.CATEGORICAL_FEATURES:
            X[cat_col] = X[cat_col].astype(int).astype(str)

        # Индексы категориальных признаков
        self._cat_features_idx = [
            self._features.index(f) for f in self.CATEGORICAL_FEATURES
        ]

        # Создаём Pool для CatBoost
        train_pool = Pool(
            X.values,
            label=y,
            cat_features=self._cat_features_idx
        )

        # Обучение CatBoost
        self.model = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_seed,
            verbose=False,
            allow_writing_files=False,
            # Регуляризация для малых выборок
            min_data_in_leaf=5,
            grow_policy='Depthwise',
            bootstrap_type='Bayesian',
            bagging_temperature=1.0
        )

        self.model.fit(train_pool)

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def _prepare_X(self, df_prep: pd.DataFrame, idx) -> Pool:
        """Подготовка Pool для предсказания."""
        X = df_prep.loc[[idx], self._features].copy()
        # Преобразуем категориальные в строки
        for cat_col in self.CATEGORICAL_FEATURES:
            X[cat_col] = X[cat_col].astype(int).astype(str)
        return Pool(X.values, cat_features=self._cat_features_idx)

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Итеративный прогноз.

        Args:
            horizon: Горизонт прогноза

        Returns:
            numpy array с прогнозами
        """
        self._check_fitted()

        df_future = self._df.copy()
        predictions = []

        last_date = self._last_train_date

        for h in range(horizon):
            # Следующая дата
            next_date = last_date + pd.DateOffset(months=1)

            # Добавляем пустую строку для прогноза
            df_future.loc[next_date, 'Все товары и услуги'] = np.nan

            # Подготавливаем признаки
            df_prep = self._prepare_features(df_future)

            # Последняя строка для прогноза
            X_test = self._prepare_X(df_prep, next_date)

            # Прогноз
            pred = self.model.predict(X_test)[0]
            predictions.append(pred)

            # Записываем прогноз для следующей итерации
            df_future.loc[next_date, 'Все товары и услуги'] = pred

            last_date = next_date

        return np.array(predictions)

    def backtest(
        self,
        df: pd.DataFrame = None,
        start_date: str = '2020-01-01',
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
        if df is None:
            df = self._df

        start = pd.Timestamp(start_date)
        test_dates = df.index[df.index >= start]
        results = []

        for target_date in test_dates:
            train_df = df[df.index < target_date].copy()

            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE + 12:
                continue

            try:
                model = CatBoostForecaster(
                    iterations=self.iterations,
                    depth=self.depth,
                    learning_rate=self.learning_rate,
                    l2_leaf_reg=self.l2_leaf_reg
                )
                model.fit(train_df, target_col)

                # Прогноз на 1 месяц
                pred = model.forecast(horizon=1)[0]
                actual = df.loc[target_date, target_col]

                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred,
                    'error': pred - actual
                })
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        """Важность признаков."""
        self._check_fitted()

        all_features = self.NUMERICAL_FEATURES + self.CATEGORICAL_FEATURES
        importance = pd.DataFrame({
            'feature': all_features,
            'importance': self.model.feature_importances_
        })
        return importance.sort_values('importance', ascending=False)

    def get_model_info(self) -> Dict:
        """Информация о модели."""
        return {
            'name': self.name,
            'iterations': self.iterations,
            'depth': self.depth,
            'learning_rate': self.learning_rate,
            'l2_leaf_reg': self.l2_leaf_reg,
            'features': self._features,
            'is_fitted': self._is_fitted
        }
