"""
Загрузчик прогнозов микрокомпонентной ARIMA модели из micro_test.csv

Формат файла:
- Строки (Date): целевые месяцы (на какой месяц прогноз)
- Столбцы: месяц когда был сделан прогноз (по какие данные были доступны)

Для h=1: прогноз на 01.01.2025 берется из столбца 01.12.2024
Для h=2: прогноз на 01.02.2025 берется из столбца 01.12.2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any


class MicroARIMAForecaster:
    """
    Загрузчик прогнозов из micro_test.csv.

    Это внешняя модель пользователя (микрокомпонентная ARIMA).
    Прогнозы уже готовы, нужно только извлечь для заданного горизонта.
    """

    name = "micro_arima"

    def __init__(self, horizon: int = 1, file_path: str = 'micro_test.csv'):
        self.horizon = horizon
        self.file_path = Path(file_path)
        self._forecasts = None
        self._is_fitted = False

    def _load_forecasts(self):
        """Загрузить матрицу прогнозов из CSV."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {self.file_path}")

        # Читаем CSV с разделителем ';' и decimal ','
        df = pd.read_csv(self.file_path, sep=';', decimal=',', encoding='utf-8-sig')

        # Первый столбец - целевые даты (Date)
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
        df = df.set_index('Date')

        # Остальные столбцы - месяцы когда был сделан прогноз
        # Переименуем столбцы в Timestamp
        new_cols = {}
        for col in df.columns:
            try:
                new_cols[col] = pd.to_datetime(col, format='%d.%m.%Y')
            except:
                pass

        df = df.rename(columns=new_cols)

        # Нормализуем индекс к началу месяца
        df.index = df.index.to_period('M').to_timestamp()

        self._forecasts = df

    def fit(self, df: pd.DataFrame = None, target_col: str = None):
        """
        Загрузка прогнозов (fit не нужен, только загрузка файла).
        """
        self._load_forecasts()
        self._is_fitted = True
        return self

    def get_forecast(self, target_date: pd.Timestamp, cutoff_date: pd.Timestamp = None) -> Optional[float]:
        """
        Получить прогноз для target_date по данным до cutoff_date.

        Args:
            target_date: Дата на которую нужен прогноз
            cutoff_date: Дата до которой были доступны данные (если None, вычисляется как target_date - horizon)

        Returns:
            Прогноз MoM (в формате 100.xx)
        """
        if not self._is_fitted:
            self._load_forecasts()
            self._is_fitted = True

        # Нормализуем даты
        target_date = pd.Timestamp(target_date).to_period('M').to_timestamp()

        if cutoff_date is None:
            cutoff_date = target_date - pd.DateOffset(months=self.horizon)
        else:
            cutoff_date = pd.Timestamp(cutoff_date).to_period('M').to_timestamp()

        # Ищем столбец с cutoff_date
        matching_cols = [c for c in self._forecasts.columns
                         if isinstance(c, pd.Timestamp) and c == cutoff_date]

        if not matching_cols:
            return None

        col = matching_cols[0]

        # Ищем строку с target_date
        if target_date not in self._forecasts.index:
            return None

        value = self._forecasts.loc[target_date, col]

        if pd.isna(value):
            return None

        return float(value)

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Получить прогноз в формате стандартного интерфейса.

        Returns:
            Dict с 'prediction' (в формате 100 + %)
        """
        # Вычисляем cutoff из данных
        cutoff = target_date - pd.DateOffset(months=self.horizon)

        value = self.get_forecast(target_date, cutoff)

        if value is None:
            return {'prediction': np.nan}

        return {'prediction': value}

    def forecast(self, horizon: int = 12, start_date: pd.Timestamp = None) -> np.ndarray:
        """
        Траектория прогнозов (для совместимости с интерфейсом).
        """
        # Для h=12 используем фиксированный cutoff
        # Здесь просто возвращаем NaN, т.к. нужен контекст данных
        return np.full(horizon, np.nan)

    def get_available_dates(self) -> Dict[str, list]:
        """Получить доступные целевые даты и cutoff даты."""
        if not self._is_fitted:
            self._load_forecasts()
            self._is_fitted = True

        return {
            'target_dates': list(self._forecasts.index),
            'cutoff_dates': [c for c in self._forecasts.columns if isinstance(c, pd.Timestamp)]
        }

    def get_forecasts_matrix(self) -> pd.DataFrame:
        """Получить полную матрицу прогнозов."""
        if not self._is_fitted:
            self._load_forecasts()
            self._is_fitted = True

        return self._forecasts.copy()
