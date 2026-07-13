"""
Dynamic Model Averaging (DMA) — адаптивные веса ансамбля
========================================================

Идея: веса моделей адаптируются на основе их недавней точности.
Модели которые были точнее недавно получают больший вес.

Метод:
1. Для каждой точки прогноза вычисляем веса моделей
2. Вес пропорционален 1/MAE за последние N месяцев
3. Веса нормализуются для суммы = 1
4. Прогноз = взвешенное среднее прогнозов моделей
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List
import warnings

from .base import BaseForecaster
from .registry import ModelRegistry


@ModelRegistry.register("dma")
class DMAForecaster(BaseForecaster):
    """
    Dynamic Model Averaging — ансамбль с адаптивными весами.

    Веса моделей обновляются на основе их недавней точности.
    """

    name = "dma"
    MIN_TRAIN_SIZE = 36
    OUTLIER_YEARS = [2010, 2022]

    # Lookback для расчёта весов
    LOOKBACK_MONTHS = 12

    # Минимальный вес для любой модели (для робастности)
    MIN_WEIGHT = 0.05

    # Модели в ансамбле
    BASE_MODELS = ['ridge', 'ngboost', 'ridge_extended', 'elasticnet', 'xgboost']

    def __init__(self,
                 lookback_months: int = None,
                 base_models: List[str] = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.lookback_months = lookback_months or self.LOOKBACK_MONTHS
        self.base_models = base_models or self.BASE_MODELS.copy()

        self.models = {}
        self.model_weights = {}
        self._df_train = None

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'DMAForecaster':
        """Обучение базовых моделей."""
        self._validate_data(df, target_col)

        self._df_train = df.copy()
        self.models = {}

        # Обучаем все базовые модели
        for model_name in self.base_models:
            try:
                model = ModelRegistry.get(model_name)
                model.fit(df, target_col)
                self.models[model_name] = model
            except Exception as e:
                # Модель не доступна или ошибка
                continue

        if len(self.models) == 0:
            raise ValueError("Не удалось обучить ни одну модель")

        # Инициализируем равные веса
        n_models = len(self.models)
        self.model_weights = {name: 1.0 / n_models for name in self.models}

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def _compute_adaptive_weights(
        self,
        df: pd.DataFrame,
        target_date: pd.Timestamp,
        target_col: str = 'Все товары и услуги'
    ) -> Dict[str, float]:
        """Вычисление адаптивных весов на основе недавней точности."""

        # Период для оценки: последние N месяцев до target_date
        lookback_start = target_date - pd.DateOffset(months=self.lookback_months)
        eval_dates = df.loc[lookback_start:target_date].dropna(subset=[target_col]).index
        eval_dates = eval_dates[eval_dates < target_date]  # исключаем target_date

        if len(eval_dates) < 3:
            # Недостаточно данных для оценки - возвращаем равные веса
            n_models = len(self.models)
            return {name: 1.0 / n_models for name in self.models}

        # Вычисляем MAE каждой модели за период
        model_errors = {name: [] for name in self.models}

        for eval_date in eval_dates:
            actual = df.loc[eval_date, target_col]

            for model_name, model in self.models.items():
                try:
                    # Получаем прогноз модели
                    pred_result = model.predict(df[df.index <= eval_date], eval_date)
                    pred = pred_result['prediction']
                    error = abs(actual - pred)
                    model_errors[model_name].append(error)
                except Exception:
                    # Если ошибка - пропускаем
                    continue

        # Вычисляем MAE для каждой модели
        model_mae = {}
        for name, errors in model_errors.items():
            if len(errors) > 0:
                model_mae[name] = np.mean(errors)
            else:
                model_mae[name] = 1.0  # большой MAE по умолчанию

        # Веса пропорциональны 1/MAE^2
        raw_weights = {}
        for name, mae in model_mae.items():
            raw_weights[name] = 1.0 / (mae ** 2 + 1e-6)

        # Нормализуем веса
        total_weight = sum(raw_weights.values())
        weights = {name: max(w / total_weight, self.MIN_WEIGHT)
                   for name, w in raw_weights.items()}

        # Перенормализуем после min_weight
        total_weight = sum(weights.values())
        weights = {name: w / total_weight for name, w in weights.items()}

        return weights

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Прогноз на горизонт (равновзвешенный)."""
        self._check_fitted()

        predictions = []
        for h in range(horizon):
            model_preds = []
            for model in self.models.values():
                try:
                    fc = model.forecast(h + 1)
                    model_preds.append(fc[-1])
                except Exception:
                    continue

            if model_preds:
                predictions.append(np.mean(model_preds))
            else:
                predictions.append(100.0)

        return np.array(predictions)

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Прогноз с адаптивными весами."""
        self._check_fitted()

        # Вычисляем адаптивные веса
        weights = self._compute_adaptive_weights(df, target_date)

        # Получаем прогнозы от всех моделей
        model_predictions = {}
        for model_name, model in self.models.items():
            try:
                pred_result = model.predict(df, target_date)
                model_predictions[model_name] = pred_result['prediction']
            except Exception:
                # Если модель не сработала - используем сезонную норму
                model_predictions[model_name] = 100.0

        # Взвешенное среднее
        weighted_sum = 0
        for name, pred in model_predictions.items():
            weighted_sum += weights.get(name, 0) * pred

        return {
            'date': target_date,
            'prediction': weighted_sum,
            'model_predictions': model_predictions,
            'weights': weights,
            'model': self.name
        }

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """Бэктест с динамическими весами."""
        start = pd.Timestamp(start_date)
        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []

        for target_date in test_dates:
            train_df = df[df.index < target_date].copy()

            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
                continue

            try:
                # Обучаем DMA на train данных
                model = DMAForecaster(
                    lookback_months=self.lookback_months,
                    base_models=self.base_models
                )
                model.fit(train_df, target_col)

                # Прогноз
                test_df = df[df.index <= target_date].copy()
                pred_result = model.predict(test_df, target_date)

                actual = df.loc[target_date, target_col]

                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred_result['prediction'],
                    'error': actual - pred_result['prediction'],
                    'weights': str(pred_result['weights'])
                })
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_model_info(self) -> Dict:
        """Информация о модели."""
        return {
            'name': self.name,
            'lookback_months': self.lookback_months,
            'base_models': list(self.models.keys()),
            'current_weights': self.model_weights,
            'is_fitted': self._is_fitted
        }
