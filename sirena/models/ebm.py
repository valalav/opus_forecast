"""
EBM (Explainable Boosting Machine) модель для прогнозирования инфляции КБР
=========================================================================

Интерпретируемая модель на базе Microsoft InterpretML.
Заменяет LSTM в ансамбле СИРЕНА-КБР v4.0.

Вес в ансамбле: 5%

Преимущества:
- GAM-структура: y = f₁(x₁) + f₂(x₂) + ... — аддитивная модель
- Визуализация влияния каждого признака
- Точность на уровне gradient boosting
- Встроенные доверительные интервалы
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from typing import Dict, Any, Optional, List, Tuple
import warnings

warnings.filterwarnings('ignore')

from .base import BaseForecaster
from .registry import ModelRegistry

# Проверка InterpretML
try:
    from interpret.glassbox import ExplainableBoostingRegressor
    from interpret import show
    INTERPRET_AVAILABLE = True
except ImportError:
    INTERPRET_AVAILABLE = False


@ModelRegistry.register("ebm")
class EBMForecaster(BaseForecaster):
    """
    Explainable Boosting Machine для прогнозирования инфляции.

    Архитектура: GAM (Generalized Additive Model) с boosting
    Интерпретируемость: Полная — графики влияния каждого признака

    Особенности:
    - Аддитивная структура: сумма вкладов признаков
    - Автоматические доверительные интервалы
    - Метод explain() для интерпретации
    """

    name = "ebm"
    MIN_TRAIN_SIZE = 24

    # Годы-выбросы (как в Ridge)
    OUTLIER_YEARS = [2022, 2010]

    # Базовые признаки (совместимы с Ridge)
    BASE_FEATURES = [
        'y_lag1', 'y_lag2', 'y_lag12', 'y_ma3',
        'month_sin', 'month_cos',
        'food_lag1', 'nonfood_lag1', 'services_lag1'
    ]

    def __init__(
        self,
        max_bins: int = 256,
        max_interaction_bins: int = 32,
        interactions: int = 0,
        outer_bags: int = 8,
        inner_bags: int = 0,
        learning_rate: float = 0.01,
        min_samples_leaf: int = 2,
        max_leaves: int = 3,
        **kwargs
    ):
        """
        Инициализация EBM.

        Args:
            max_bins: Максимальное количество бинов для признаков
            max_interaction_bins: Бины для взаимодействий
            interactions: Количество взаимодействий (0 = только main effects)
            outer_bags: Количество внешних бэггинг-итераций
            inner_bags: Количество внутренних итераций
            learning_rate: Скорость обучения
            min_samples_leaf: Минимум наблюдений в листе
            max_leaves: Максимум листьев в дереве
        """
        super().__init__(**kwargs)

        self.max_bins = max_bins
        self.max_interaction_bins = max_interaction_bins
        self.interactions = interactions
        self.outer_bags = outer_bags
        self.inner_bags = inner_bags
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.max_leaves = max_leaves

        self.model = None
        self.scaler = None
        self._features = None
        self._feature_names = None
        self.last_X = None
        self.seasonal_mean = None

        # Fallback если InterpretML недоступен
        self._use_fallback = not INTERPRET_AVAILABLE
        self.fallback_model = None

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков (аналогично Ridge)."""
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

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'EBMForecaster':
        """
        Обучение EBM.

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

        # Сезонное среднее для fallback
        clean_df = df_prep[~df_prep['year'].isin(self.OUTLIER_YEARS)]
        self.seasonal_mean = clean_df.groupby('month')[target_col].mean()

        # Исключаем выбросные годы
        train_df = df_prep[~df_prep['year'].isin(self.OUTLIER_YEARS)]

        # Определяем признаки
        self._features = self.BASE_FEATURES.copy()

        # Очистка
        train_clean = train_df.dropna(subset=self._features + [target_col])

        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных: {len(train_clean)} < {self.MIN_TRAIN_SIZE}")

        # Обучение
        X = train_clean[self._features].values
        y = train_clean[target_col].values

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        self._feature_names = self._features.copy()

        if self._use_fallback:
            # Fallback на Ridge если InterpretML недоступен
            self._fit_fallback(X_scaled, y)
        else:
            # EBM
            self.model = ExplainableBoostingRegressor(
                max_bins=self.max_bins,
                max_interaction_bins=self.max_interaction_bins,
                interactions=self.interactions,
                outer_bags=self.outer_bags,
                inner_bags=self.inner_bags,
                learning_rate=self.learning_rate,
                min_samples_leaf=self.min_samples_leaf,
                max_leaves=self.max_leaves,
                feature_names=self._feature_names
            )
            self.model.fit(X_scaled, y)

        # Сохраняем последние признаки для прогноза
        self.last_X = df_prep[self._features].iloc[-1:].values
        self._last_values = df_prep.iloc[-1].to_dict()

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def _fit_fallback(self, X: np.ndarray, y: np.ndarray):
        """Fallback обучение через Ridge."""
        from sklearn.linear_model import Ridge
        self.fallback_model = Ridge(alpha=0.3)
        self.fallback_model.fit(X, y)

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Прогноз на горизонт.

        Note: Для полного прогноза используйте predict() с данными.
        Этот метод возвращает сезонное среднее как baseline.

        Args:
            horizon: Количество месяцев

        Returns:
            numpy array с прогнозами
        """
        self._check_fitted()

        # Для EBM возвращаем сезонное среднее как baseline
        if self.seasonal_mean is None:
            return np.zeros(horizon)

        last_month = self._last_train_date.month if self._last_train_date else 1
        predictions = []

        for i in range(horizon):
            month = ((last_month + i) % 12) + 1
            pred = self.seasonal_mean.get(month, 100.0)
            predictions.append(pred)

        return np.array(predictions)

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Точечный прогноз на дату.

        Args:
            df: DataFrame с данными
            target_date: Дата прогноза

        Returns:
            Dict с прогнозом и объяснением
        """
        self._check_fitted()

        df_prep = self._prepare_features(df)
        test_row = df_prep.loc[[target_date]]

        X_test = self.scaler.transform(test_row[self._features].values)

        if self._use_fallback:
            pred = self.fallback_model.predict(X_test)[0]
            explanation = None
        else:
            pred = self.model.predict(X_test)[0]
            explanation = self._get_local_explanation(X_test, test_row)

        return {
            'date': target_date,
            'prediction': pred,
            'model': self.name,
            'explanation': explanation
        }

    def _get_local_explanation(self, X: np.ndarray, df_row: pd.DataFrame) -> Dict[str, Any]:
        """
        Локальное объяснение прогноза.

        Args:
            X: Масштабированные признаки
            df_row: DataFrame с исходными значениями

        Returns:
            Dict с вкладами признаков
        """
        if self._use_fallback or self.model is None:
            return None

        # Получаем локальные объяснения от EBM
        local_expl = self.model.explain_local(X)

        # Извлекаем вклады
        contributions = {}
        intercept = local_expl.data(0)['extra']['scores'][0]  # Intercept

        for i, feature_name in enumerate(self._feature_names):
            score = local_expl.data(0)['scores'][i]
            raw_value = df_row[feature_name].values[0]
            contributions[feature_name] = {
                'value': raw_value,
                'contribution': score
            }

        return {
            'intercept': intercept,
            'contributions': contributions,
            'total': sum(c['contribution'] for c in contributions.values()) + intercept
        }

    def explain(self) -> Dict[str, Any]:
        """
        Глобальное объяснение модели.

        Returns:
            Dict с важностью признаков и графиками
        """
        self._check_fitted()

        if self._use_fallback:
            # Fallback: коэффициенты Ridge
            return {
                'type': 'fallback_ridge',
                'feature_importance': dict(zip(
                    self._feature_names,
                    np.abs(self.fallback_model.coef_)
                ))
            }

        # EBM глобальное объяснение
        global_expl = self.model.explain_global()

        # Извлекаем важность признаков
        feature_importance = {}
        feature_shapes = {}

        for i, name in enumerate(global_expl.data()['names']):
            if name in self._feature_names:
                importance = global_expl.data()['scores'][i]
                feature_importance[name] = importance

                # Форма влияния (shape function)
                feature_shapes[name] = {
                    'x': global_expl.data(i)['names'],
                    'y': global_expl.data(i)['scores'],
                    'density': global_expl.data(i).get('density', None)
                }

        return {
            'type': 'ebm',
            'feature_importance': feature_importance,
            'feature_shapes': feature_shapes,
            'intercept': self.model.intercept_
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Важность признаков как DataFrame.

        Returns:
            DataFrame с важностью признаков
        """
        explanation = self.explain()

        if explanation['type'] == 'fallback_ridge':
            importance = pd.DataFrame({
                'feature': list(explanation['feature_importance'].keys()),
                'importance': list(explanation['feature_importance'].values())
            })
        else:
            importance = pd.DataFrame({
                'feature': list(explanation['feature_importance'].keys()),
                'importance': list(explanation['feature_importance'].values())
            })

        importance = importance.sort_values('importance', ascending=False)
        return importance

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """
        Бэктестирование EBM.

        Args:
            df: DataFrame с данными
            start_date: Начало периода
            target_col: Целевая колонка

        Returns:
            DataFrame с результатами
        """
        start = pd.Timestamp(start_date)

        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []

        for target_date in test_dates:
            train_df = df[df.index < target_date].copy()

            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
                continue

            try:
                # Создаём новую модель для чистого бэктеста
                model = EBMForecaster(
                    max_bins=self.max_bins,
                    interactions=self.interactions,
                    outer_bags=self.outer_bags
                )
                model.fit(train_df, target_col)

                test_df = df[df.index <= target_date].copy()
                pred_result = model.predict(test_df, target_date)

                actual = df.loc[target_date, target_col]

                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred_result['prediction'],
                    'error': actual - pred_result['prediction']
                })
            except Exception as e:
                continue

        return pd.DataFrame(results)

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

    def plot_feature_effects(self, feature_name: str = None):
        """
        Визуализация влияния признака.

        Args:
            feature_name: Название признака (если None — все)

        Returns:
            Объект визуализации InterpretML или None
        """
        if self._use_fallback:
            return None

        global_expl = self.model.explain_global()

        if feature_name:
            idx = self._feature_names.index(feature_name)
            return show(global_expl, idx)

        return show(global_expl)


# Фабрика для совместимости
def create_ebm_model(**kwargs):
    """Фабрика для EBM модели."""
    return EBMForecaster(**kwargs)
