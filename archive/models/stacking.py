"""
Стакинг (Meta-Learning) модель
==============================

Двухуровневая модель:
1. Базовые модели (Ridge, BVAR, LightGBM, Prophet, SARIMA, ETS, EBM)
   генерируют OOF-прогнозы
2. Meta-Ridge обучается на комбинации прогнозов

Преимущества:
- Автоматически подбирает оптимальные веса
- Адаптируется к сильным/слабым сторонам каждой модели
- Может использовать нелинейные комбинации
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
import warnings

from .base import BaseForecaster
from .registry import ModelRegistry

# Подавляем предупреждения
warnings.filterwarnings('ignore')


@ModelRegistry.register("stacking")
class StackingForecaster(BaseForecaster):
    """
    Двухуровневая стакинг-модель.

    Уровень 1: Базовые модели генерируют прогнозы
    Уровень 2: Meta-Ridge комбинирует прогнозы

    Attributes:
        BASE_MODELS: Список базовых моделей
        META_ALPHA: Регуляризация meta-модели
        OOF_START: Начало периода для OOF-прогнозов
    """

    name = "stacking"

    # Базовые модели (без stacking, чтобы избежать рекурсии)
    BASE_MODELS = ['ridge', 'bvar', 'lightgbm', 'prophet', 'sarima', 'ets', 'ebm']

    # Параметры
    META_ALPHA = 1.0  # Сильная регуляризация чтобы не переобучиться
    OOF_START = '2019-01-01'  # Начало OOF-периода
    MIN_OOF_SAMPLES = 24  # Минимум наблюдений для meta-модели

    def __init__(
        self,
        base_models: Optional[List[str]] = None,
        meta_alpha: float = None,
        oof_start: str = None,
        **kwargs
    ):
        """
        Args:
            base_models: Список базовых моделей (None = все)
            meta_alpha: Регуляризация meta-Ridge
            oof_start: Начало OOF-периода
        """
        super().__init__(**kwargs)

        self.base_model_names = base_models or self.BASE_MODELS.copy()
        self.meta_alpha = meta_alpha or self.META_ALPHA
        self.oof_start = oof_start or self.OOF_START

        # Компоненты
        self._base_models: Dict[str, BaseForecaster] = {}
        self._meta_model: Optional[Ridge] = None
        self._scaler: Optional[StandardScaler] = None

        # OOF данные
        self._oof_predictions: Optional[pd.DataFrame] = None
        self._oof_actuals: Optional[pd.Series] = None

        # Для прогноза
        self._last_df: Optional[pd.DataFrame] = None

    def _get_base_model(self, name: str) -> BaseForecaster:
        """Получить экземпляр базовой модели."""
        return ModelRegistry.get(name)

    def _generate_oof_predictions(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Генерирует OOF (Out-of-Fold) прогнозы через expanding window.

        Returns:
            (oof_predictions DataFrame, actuals Series)
        """
        start_date = pd.to_datetime(self.oof_start)
        test_dates = df.index[df.index >= start_date]

        oof_records = []

        for date in test_dates:
            cutoff = date - pd.DateOffset(months=1)
            train_data = df[df.index <= cutoff]

            if len(train_data) < 36:  # MIN_TRAIN_SIZE для Ridge
                continue

            actual = df.loc[date, target_col] if target_col in df.columns else df.loc[date].iloc[0]

            row = {'date': date, 'actual': actual}

            # Прогноз каждой базовой модели
            for model_name in self.base_model_names:
                try:
                    model = self._get_base_model(model_name)
                    model.fit(train_data, target_col)
                    pred = model.forecast(horizon=1)[0]
                    row[model_name] = pred
                except Exception:
                    row[model_name] = np.nan

            oof_records.append(row)

        if len(oof_records) < self.MIN_OOF_SAMPLES:
            raise ValueError(
                f"Недостаточно OOF-прогнозов: {len(oof_records)} < {self.MIN_OOF_SAMPLES}"
            )

        oof_df = pd.DataFrame(oof_records).set_index('date')
        actuals = oof_df['actual']
        predictions = oof_df.drop(columns=['actual'])

        # Заполняем пропуски медианой по строке
        predictions = predictions.fillna(predictions.median())

        return predictions, actuals

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'Все товары и услуги'
    ) -> 'StackingForecaster':
        """
        Обучение стакинг-модели.

        1. Генерируем OOF-прогнозы от базовых моделей
        2. Обучаем meta-Ridge на OOF-прогнозах

        Args:
            df: DataFrame с данными
            target_col: Целевая колонка
        """
        self._last_df = df.copy()

        # 1. Генерируем OOF
        self._oof_predictions, self._oof_actuals = self._generate_oof_predictions(
            df, target_col
        )

        # 2. Подготовка данных для meta-модели
        X = self._oof_predictions.values
        y = self._oof_actuals.values

        # Добавляем дополнительные признаки
        # - Среднее прогнозов
        # - Месяц (sin/cos)
        months = self._oof_predictions.index.month
        month_sin = np.sin(2 * np.pi * months / 12).values.reshape(-1, 1)
        month_cos = np.cos(2 * np.pi * months / 12).values.reshape(-1, 1)
        X_avg = X.mean(axis=1, keepdims=True)

        X_full = np.hstack([X, X_avg, month_sin, month_cos])

        # 3. Скейлинг
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_full)

        # 4. Обучаем meta-Ridge
        self._meta_model = Ridge(alpha=self.meta_alpha)
        self._meta_model.fit(X_scaled, y)

        # 5. Обучаем базовые модели на полных данных
        for model_name in self.base_model_names:
            try:
                model = self._get_base_model(model_name)
                model.fit(df, target_col)
                self._base_models[model_name] = model
            except Exception:
                pass

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Генерирует прогноз через стакинг.

        1. Получает прогнозы от базовых моделей
        2. Комбинирует через meta-модель
        """
        self._check_fitted()

        forecasts = np.zeros((horizon, len(self.base_model_names)))

        # Получаем прогнозы базовых моделей
        for i, model_name in enumerate(self.base_model_names):
            if model_name in self._base_models:
                try:
                    fc = self._base_models[model_name].forecast(horizon)
                    forecasts[:, i] = fc
                except Exception:
                    forecasts[:, i] = np.nan
            else:
                forecasts[:, i] = np.nan

        # Заполняем пропуски средним
        for h in range(horizon):
            row = forecasts[h, :]
            nan_mask = np.isnan(row)
            if nan_mask.any() and not nan_mask.all():
                forecasts[h, nan_mask] = np.nanmean(row)

        # Добавляем дополнительные признаки
        last_month = self._last_train_date.month
        months = [(last_month + h) % 12 + 1 for h in range(horizon)]
        month_sin = np.sin(2 * np.pi * np.array(months) / 12).reshape(-1, 1)
        month_cos = np.cos(2 * np.pi * np.array(months) / 12).reshape(-1, 1)
        X_avg = forecasts.mean(axis=1, keepdims=True)

        X_full = np.hstack([forecasts, X_avg, month_sin, month_cos])

        # Скейлинг и прогноз meta-модели
        X_scaled = self._scaler.transform(X_full)
        stacked_forecast = self._meta_model.predict(X_scaled)

        return stacked_forecast

    def get_base_forecasts(self, horizon: int = 12) -> Dict[str, np.ndarray]:
        """Возвращает прогнозы базовых моделей."""
        self._check_fitted()

        result = {}
        for model_name in self.base_model_names:
            if model_name in self._base_models:
                try:
                    result[model_name] = self._base_models[model_name].forecast(horizon)
                except Exception:
                    pass

        result['stacking'] = self.forecast(horizon)
        return result

    def get_meta_weights(self) -> Dict[str, float]:
        """
        Возвращает веса meta-модели (коэффициенты Ridge).

        Note: Это приблизительные "веса" — реальная комбинация нелинейна
        из-за дополнительных признаков.
        """
        self._check_fitted()

        n_base = len(self.base_model_names)
        coefs = self._meta_model.coef_[:n_base]

        # Нормализуем для интерпретации
        abs_coefs = np.abs(coefs)
        weights = abs_coefs / abs_coefs.sum()

        return dict(zip(self.base_model_names, weights))

    def backtest(
        self,
        df: pd.DataFrame = None,
        start_date: str = '2020-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """
        Бэктест стакинг-модели.

        Note: Полный бэктест очень медленный (O(n * k * m) где n=периоды,
        k=базовые модели, m=OOF периоды). Используем упрощённую версию.
        """
        if df is None:
            if self._last_df is not None:
                df = self._last_df
            else:
                raise ValueError("Нужны данные для бэктеста")

        results = []
        start = pd.to_datetime(start_date)
        test_dates = df.index[df.index >= start]

        for date in test_dates:
            cutoff = date - pd.DateOffset(months=1)
            train_data = df[df.index <= cutoff]

            if len(train_data) < 60:  # Нужно больше данных для стакинга
                continue

            try:
                self.fit(train_data, target_col)
                pred = self.forecast(horizon=1)[0]
                actual = df.loc[date, target_col] if target_col in df.columns else df.loc[date].iloc[0]

                results.append({
                    'date': date,
                    'actual': actual,
                    'prediction': pred,
                    'error': pred - actual
                })
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_model_info(self) -> Dict:
        """Информация о модели."""
        info = {
            'name': self.name,
            'base_models': self.base_model_names,
            'meta_alpha': self.meta_alpha,
            'oof_start': self.oof_start,
            'is_fitted': self._is_fitted,
        }

        if self._is_fitted:
            info['meta_weights'] = self.get_meta_weights()
            info['n_oof_samples'] = len(self._oof_predictions) if self._oof_predictions is not None else 0

        return info
