"""
Базовый класс для моделей прогнозирования СИРЕНА-КБР
====================================================

Все модели должны наследоваться от BaseForecaster и реализовывать
абстрактные методы fit(), forecast(), backtest().
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np


class BaseForecaster(ABC):
    """
    Абстрактный базовый класс для всех моделей прогнозирования.

    Пример использования:
        class MyModel(BaseForecaster):
            name = "my_model"

            def fit(self, df):
                # обучение
                return self

            def forecast(self, horizon=12):
                # прогноз
                return np.array([...])

            def backtest(self, df, start_date):
                # бэктест
                return pd.DataFrame(...)
    """

    # Имя модели для регистрации
    name: str = "base"

    # Минимальное количество наблюдений для обучения
    MIN_TRAIN_SIZE: int = 24

    def __init__(self, **kwargs):
        """Инициализация модели с параметрами."""
        self.params = kwargs
        self._is_fitted = False
        self._last_train_date: Optional[pd.Timestamp] = None

    @abstractmethod
    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'BaseForecaster':
        """
        Обучение модели на данных.

        Args:
            df: DataFrame с временным индексом и целевой переменной
            target_col: Название целевой колонки

        Returns:
            self для цепочки вызовов
        """
        pass

    @abstractmethod
    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Прогноз на заданный горизонт.

        Args:
            horizon: Количество периодов для прогноза

        Returns:
            numpy array с прогнозами (MoM в %)
        """
        pass

    @abstractmethod
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
            start_date: Начальная дата бэктеста
            target_col: Целевая переменная

        Returns:
            DataFrame с колонками: date, actual, prediction, error
        """
        pass

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Прогноз на конкретную дату.

        Args:
            df: DataFrame с данными до target_date
            target_date: Дата прогноза

        Returns:
            dict с прогнозом и метаданными
        """
        self.fit(df[df.index < target_date])
        fc = self.forecast(horizon=1)

        return {
            'date': target_date,
            'prediction': fc[0],
            'model': self.name
        }

    def iterative_forecast(
        self,
        df: pd.DataFrame,
        horizon: int = 12,
        target_col: str = 'Все товары и услуги'
    ) -> np.ndarray:
        """
        Итеративный прогноз через predict().

        Для каждого горизонта:
        1. Вызывает predict() для получения точечного прогноза
        2. Добавляет прогноз в DataFrame
        3. Переобучает модель (опционально)
        4. Прогнозирует следующий горизонт

        Args:
            df: DataFrame с историческими данными
            horizon: Горизонт прогноза в месяцах
            target_col: Целевая переменная

        Returns:
            numpy array с прогнозами (MoM в %, не индекс)
        """
        self._check_fitted()

        predictions = []
        last_date = df.index.max()
        df_work = df.copy()

        for h in range(horizon):
            target_date = last_date + pd.DateOffset(months=h + 1)

            # Создаём строку для прогноза, копируя предыдущую строку
            if target_date not in df_work.index:
                # Копируем последнюю строку как шаблон
                prev_date = df_work.index.max()
                df_work.loc[target_date] = df_work.loc[prev_date].copy()
                # Целевую переменную ставим NaN (будем прогнозировать)
                df_work.loc[target_date, target_col] = np.nan
                # Сортируем по индексу
                df_work = df_work.sort_index()

            # Получаем прогноз через predict()
            try:
                pred_result = self.predict(df_work, target_date)
                pred = pred_result['prediction']

                # Конвертируем из индекса (100.xx) в проценты (0.xx) если нужно
                if abs(pred) > 50:
                    pred = pred - 100

                predictions.append(pred)

                # Обновляем DataFrame для следующего шага
                df_work.loc[target_date, target_col] = pred + 100  # храним как индекс

            except Exception as e:
                # Fallback на сезонную норму если есть
                sn = getattr(self, 'seasonal_norm', None)
                if sn is not None and (isinstance(sn, dict) or (hasattr(sn, '__len__') and len(sn) > 0)):
                    month = target_date.month
                    if isinstance(sn, dict):
                        pred = sn.get(month, 100.0) - 100
                    else:
                        pred = sn.get(month, 100.0) - 100  # pd.Series with .get
                    predictions.append(pred)
                    df_work.loc[target_date, target_col] = pred + 100
                else:
                    # Последнее значение
                    last_val = df_work[target_col].dropna().iloc[-1]
                    if abs(last_val) > 50:
                        pred = last_val - 100
                    else:
                        pred = last_val
                    predictions.append(pred)
                    df_work.loc[target_date, target_col] = pred + 100

        return np.array(predictions)

    def get_params(self) -> Dict[str, Any]:
        """Получить параметры модели."""
        return self.params.copy()

    def set_params(self, **params) -> 'BaseForecaster':
        """Установить параметры модели."""
        self.params.update(params)
        return self

    @property
    def is_fitted(self) -> bool:
        """Проверка, обучена ли модель."""
        return self._is_fitted

    def _check_fitted(self):
        """Проверка что модель обучена перед прогнозом."""
        if not self._is_fitted:
            raise ValueError(f"Модель {self.name} не обучена. Вызовите fit() сначала.")

    def _validate_data(self, df: pd.DataFrame, target_col: str) -> pd.Series:
        """
        Валидация входных данных.

        Args:
            df: DataFrame
            target_col: Целевая колонка

        Returns:
            Series с целевой переменной
        """
        if df.empty:
            raise ValueError("DataFrame пустой")

        if target_col not in df.columns:
            raise ValueError(f"Колонка '{target_col}' не найдена в данных")

        series = df[target_col].dropna()

        if len(series) < self.MIN_TRAIN_SIZE:
            raise ValueError(
                f"Недостаточно данных: {len(series)} < {self.MIN_TRAIN_SIZE}"
            )

        return series

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', fitted={self._is_fitted})"


class ForecastResult:
    """
    Результат прогнозирования с метаданными.

    Attributes:
        values: numpy array с прогнозами
        dates: даты прогноза
        model: название модели
        lower: нижняя граница интервала (если есть)
        upper: верхняя граница интервала (если есть)
    """

    def __init__(
        self,
        values: np.ndarray,
        dates: pd.DatetimeIndex,
        model: str,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None
    ):
        self.values = values
        self.dates = dates
        self.model = model
        self.lower = lower
        self.upper = upper

    def to_dataframe(self) -> pd.DataFrame:
        """Конвертация в DataFrame."""
        df = pd.DataFrame({
            'Date': self.dates,
            'Forecast': self.values
        })

        if self.lower is not None:
            df['Lower'] = self.lower
        if self.upper is not None:
            df['Upper'] = self.upper

        return df

    def __repr__(self) -> str:
        return f"ForecastResult(model='{self.model}', horizon={len(self.values)})"
