"""
Hierarchical Forecast с MinTrace Reconciliation
================================================

Прогнозирует все уровни иерархии отдельно, затем согласовывает:
- Total (Все товары и услуги)
- Food (Продовольственные товары)
- NonFood (Непродовольственные товары)
- Services (Услуги)

Условие согласования: Total = Food + NonFood + Services (по весам)

MinTrace минимизирует дисперсию ошибок прогнозов при согласовании.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from typing import Dict, Optional, Tuple
import warnings

from .base import BaseForecaster
from .registry import ModelRegistry

warnings.filterwarnings('ignore')


# Веса компонентов (из micro_sprav.csv)
COMPONENT_WEIGHTS = {
    'Продовольственные товары': 0.3948,
    'Непродовольственные товары': 0.3653,
    'Услуги': 0.2342
}

COMPONENTS = list(COMPONENT_WEIGHTS.keys())
TOTAL_COL = 'Все товары и услуги'


@ModelRegistry.register("hierarchical")
class HierarchicalForecaster(BaseForecaster):
    """
    Hierarchical Forecast с MinTrace reconciliation.

    Иерархия:
    ```
              Total
             /  |  \\
         Food NonFood Services
    ```

    Условие: Total = w1*Food + w2*NonFood + w3*Services

    MinTrace находит оптимальные согласованные прогнозы, минимизируя
    дисперсию ошибок.
    """

    name = "hierarchical"

    # Базовая модель для прогнозирования каждого ряда
    BASE_MODEL = 'ridge'
    MIN_TRAIN_SIZE = 36

    def __init__(
        self,
        base_model: str = None,
        weights: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        Args:
            base_model: Модель для прогнозирования (default: ridge)
            weights: Веса компонентов (default: из micro_sprav)
        """
        super().__init__(**kwargs)

        self.base_model_name = base_model or self.BASE_MODEL
        self.weights = weights or COMPONENT_WEIGHTS.copy()

        # Модели для каждого ряда
        self._models: Dict[str, BaseForecaster] = {}

        # Ковариационная матрица ошибок (для MinTrace)
        self._W: Optional[np.ndarray] = None

        # Матрица суммирования
        self._S: Optional[np.ndarray] = None

        # Исторические данные
        self._df: Optional[pd.DataFrame] = None

    def _build_summing_matrix(self) -> np.ndarray:
        """
        Строит матрицу суммирования S.

        S @ bottom_level = all_levels

        Для нашей иерархии (Total, Food, NonFood, Services):
        S = [[w1, w2, w3],   # Total = w1*Food + w2*NonFood + w3*Services
             [1,  0,  0],    # Food
             [0,  1,  0],    # NonFood
             [0,  0,  1]]    # Services
        """
        w = list(self.weights.values())
        S = np.array([
            w,           # Total (взвешенная сумма)
            [1, 0, 0],   # Food
            [0, 1, 0],   # NonFood
            [0, 0, 1]    # Services
        ])
        return S

    def _estimate_covariance(self, df: pd.DataFrame) -> np.ndarray:
        """
        Оценивает ковариационную матрицу ошибок прогнозов.

        Используем исторические ошибки для оценки W.
        """
        # Упрощённая оценка: диагональная матрица с дисперсиями
        # каждого ряда (Total, Food, NonFood, Services)

        cols = [TOTAL_COL] + COMPONENTS
        variances = []

        for col in cols:
            if col in df.columns:
                # Дисперсия первых разностей (proxy для дисперсии ошибок)
                var = df[col].diff().var()
                variances.append(var if not np.isnan(var) else 1.0)
            else:
                variances.append(1.0)

        return np.diag(variances)

    def _mintrace_reconcile(
        self,
        base_forecasts: np.ndarray,
        S: np.ndarray,
        W: np.ndarray
    ) -> np.ndarray:
        """
        MinTrace reconciliation.

        Формула: ỹ = S @ (S' @ W⁻¹ @ S)⁻¹ @ S' @ W⁻¹ @ ŷ

        Args:
            base_forecasts: Базовые прогнозы [Total, Food, NonFood, Services]
            S: Матрица суммирования (4 x 3)
            W: Ковариационная матрица (4 x 4)

        Returns:
            Согласованные прогнозы [Total, Food, NonFood, Services]
        """
        # W⁻¹
        W_inv = np.linalg.inv(W + 1e-6 * np.eye(W.shape[0]))

        # S' @ W⁻¹ @ S
        StWinvS = S.T @ W_inv @ S

        # (S' @ W⁻¹ @ S)⁻¹
        StWinvS_inv = np.linalg.inv(StWinvS + 1e-6 * np.eye(StWinvS.shape[0]))

        # P = S @ (S' @ W⁻¹ @ S)⁻¹ @ S' @ W⁻¹
        P = S @ StWinvS_inv @ S.T @ W_inv

        # Согласованные прогнозы
        reconciled = P @ base_forecasts

        return reconciled

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'Все товары и услуги'
    ) -> 'HierarchicalForecaster':
        """
        Обучение иерархической модели.

        1. Обучает базовую модель для каждого ряда (Total, Food, NonFood, Services)
        2. Оценивает ковариационную матрицу W
        3. Строит матрицу суммирования S
        """
        self._df = df.copy()

        # Проверяем наличие компонентов
        cols_needed = [TOTAL_COL] + COMPONENTS
        missing = [c for c in cols_needed if c not in df.columns]
        if missing:
            raise ValueError(f"Отсутствуют колонки: {missing}")

        # Обучаем модель для каждого ряда
        for col in cols_needed:
            model = ModelRegistry.get(self.base_model_name)
            model.fit(df, target_col=col)
            self._models[col] = model

        # Строим матрицы
        self._S = self._build_summing_matrix()
        self._W = self._estimate_covariance(df)

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Генерирует согласованный прогноз Total.

        1. Прогнозирует все ряды базовой моделью
        2. Применяет MinTrace reconciliation
        3. Возвращает прогноз Total
        """
        self._check_fitted()

        # Прогнозы базовых моделей
        cols = [TOTAL_COL] + COMPONENTS
        base_fc = np.zeros((horizon, len(cols)))

        for i, col in enumerate(cols):
            if col in self._models:
                base_fc[:, i] = self._models[col].forecast(horizon)

        # Применяем MinTrace для каждого горизонта
        reconciled_fc = np.zeros((horizon, len(cols)))

        for h in range(horizon):
            reconciled_fc[h, :] = self._mintrace_reconcile(
                base_fc[h, :],
                self._S,
                self._W
            )

        # Возвращаем Total (первая колонка)
        return reconciled_fc[:, 0]

    def forecast_all(self, horizon: int = 12) -> Dict[str, np.ndarray]:
        """
        Возвращает согласованные прогнозы для всех рядов.

        Returns:
            Dict с ключами: 'total', 'food', 'nonfood', 'services'
        """
        self._check_fitted()

        cols = [TOTAL_COL] + COMPONENTS
        base_fc = np.zeros((horizon, len(cols)))

        for i, col in enumerate(cols):
            if col in self._models:
                base_fc[:, i] = self._models[col].forecast(horizon)

        reconciled_fc = np.zeros((horizon, len(cols)))
        for h in range(horizon):
            reconciled_fc[h, :] = self._mintrace_reconcile(
                base_fc[h, :],
                self._S,
                self._W
            )

        return {
            'total': reconciled_fc[:, 0],
            'food': reconciled_fc[:, 1],
            'nonfood': reconciled_fc[:, 2],
            'services': reconciled_fc[:, 3],
            # Также базовые для сравнения
            'base_total': base_fc[:, 0],
            'base_food': base_fc[:, 1],
            'base_nonfood': base_fc[:, 2],
            'base_services': base_fc[:, 3],
        }

    def get_reconciliation_adjustment(self, horizon: int = 12) -> pd.DataFrame:
        """
        Показывает корректировку от reconciliation.

        Returns:
            DataFrame с колонками: horizon, series, base, reconciled, adjustment
        """
        self._check_fitted()

        fc = self.forecast_all(horizon)

        rows = []
        for h in range(horizon):
            for series in ['total', 'food', 'nonfood', 'services']:
                base_val = fc[f'base_{series}'][h]
                rec_val = fc[series][h]
                rows.append({
                    'horizon': h + 1,
                    'series': series,
                    'base': base_val,
                    'reconciled': rec_val,
                    'adjustment': rec_val - base_val
                })

        return pd.DataFrame(rows)

    def backtest(
        self,
        df: pd.DataFrame = None,
        start_date: str = '2020-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """Бэктест иерархической модели."""
        if df is None:
            if self._df is not None:
                df = self._df
            else:
                raise ValueError("Нужны данные для бэктеста")

        results = []
        start = pd.to_datetime(start_date)
        test_dates = df.index[df.index >= start]

        for date in test_dates:
            cutoff = date - pd.DateOffset(months=1)
            train_data = df[df.index <= cutoff]

            if len(train_data) < self.MIN_TRAIN_SIZE + 12:
                continue

            try:
                self.fit(train_data, target_col)
                pred = self.forecast(horizon=1)[0]
                actual = df.loc[date, target_col]

                results.append({
                    'date': date,
                    'actual': actual,
                    'prediction': pred,
                    'error': pred - actual
                })
            except Exception:
                continue

        return pd.DataFrame(results)

    def check_coherence(self, horizon: int = 1) -> Dict:
        """
        Проверяет когерентность прогнозов.

        Когерентность: Total ≈ w1*Food + w2*NonFood + w3*Services
        """
        self._check_fitted()

        fc = self.forecast_all(horizon)

        # Расчёт суммы компонентов
        w = list(self.weights.values())
        sum_components = (
            w[0] * fc['food'] +
            w[1] * fc['nonfood'] +
            w[2] * fc['services']
        )

        # Разница с Total
        diff = fc['total'] - sum_components

        return {
            'total': fc['total'],
            'sum_components': sum_components,
            'difference': diff,
            'is_coherent': np.allclose(diff, 0, atol=1e-6)
        }

    def get_model_info(self) -> Dict:
        """Информация о модели."""
        return {
            'name': self.name,
            'base_model': self.base_model_name,
            'weights': self.weights,
            'is_fitted': self._is_fitted,
            'last_train_date': self._last_train_date,
        }
