"""
Ridge SA модель: Bottom-Up прогнозирование на SA данных
=======================================================

Прогнозирует 3 компонента (Прод/Непрод/Услуги) отдельно на
сезонно-скорректированных данных, затем агрегирует по весам.

Преимущества SA подхода:
- Данные уже без сезонности — модель фокусируется на трендах
- Bottom-up: можно анализировать вклад каждого компонента
- Меньше шума от сезонных колебаний

Данные: data/sa_fl.csv (январь 2016 — октябрь 2025)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from typing import Dict, Optional, List, Tuple
from pathlib import Path

from .base import BaseForecaster, ForecastResult
from .registry import ModelRegistry


# Веса компонентов (из micro_sprav.csv, нормализованы до 100%)
COMPONENT_WEIGHTS = {
    'Продовольственные товары': 0.3972,      # 39.48% / 99.43%
    'Непродовольственные товары': 0.3674,    # 36.53% / 99.43%
    'Услуги': 0.2355                          # 23.42% / 99.43%
}

COMPONENTS = list(COMPONENT_WEIGHTS.keys())


@ModelRegistry.register("ridge_sa")
class RidgeSAForecaster(BaseForecaster):
    """
    Bottom-Up Ridge модель на сезонно-скорректированных данных.

    Алгоритм:
    1. Загружает SA данные для 3 компонентов
    2. Обучает отдельную Ridge модель для каждого компонента
    3. Прогнозирует каждый компонент независимо
    4. Агрегирует по весам → общий ИПЦ

    Признаки (упрощённые, т.к. SA данные без сезонности):
    - y_lag1, y_lag2, y_lag3: лаги целевой переменной
    - y_ma3: скользящее среднее
    - trend: линейный тренд
    """

    name = "ridge_sa"
    MIN_TRAIN_SIZE = 24  # Минимум наблюдений

    # Ridge регуляризация
    ALPHA = 0.5

    # Признаки для SA модели (без сезонных!)
    FEATURES = [
        'y_lag1', 'y_lag2', 'y_lag3',
        'y_ma3', 'y_ma6',
        'trend'
    ]

    def __init__(self, alpha: float = None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha or self.ALPHA
        self.models: Dict[str, Ridge] = {}
        self.scalers: Dict[str, RobustScaler] = {}
        self._last_values: Dict[str, pd.Series] = {}
        self._sa_data: Optional[pd.DataFrame] = None

    def _load_sa_data(self) -> pd.DataFrame:
        """Загрузка SA данных."""
        data_path = Path(__file__).parent.parent.parent / 'data' / 'sa_fl.csv'

        df = pd.read_csv(data_path, sep=';', decimal=',', encoding='utf-8-sig')
        df['Дата'] = pd.to_datetime(df['Дата'], format='%d.%m.%Y')
        df['Значение'] = pd.to_numeric(
            df['Значение'].astype(str).str.replace(',', '.'),
            errors='coerce'
        )

        pivot = df.pivot_table(
            index='Дата',
            columns='Товар',
            values='Значение',
            aggfunc='first'
        ).sort_index()

        pivot.index.name = 'Date'
        return pivot

    def _prepare_features(self, series: pd.Series) -> pd.DataFrame:
        """Подготовка признаков для одного компонента."""
        df = pd.DataFrame({'y': series})

        # Лаги
        df['y_lag1'] = df['y'].shift(1)
        df['y_lag2'] = df['y'].shift(2)
        df['y_lag3'] = df['y'].shift(3)

        # Скользящие средние
        df['y_ma3'] = df['y'].rolling(3).mean().shift(1)
        df['y_ma6'] = df['y'].rolling(6).mean().shift(1)

        # Линейный тренд
        df['trend'] = np.arange(len(df))

        return df

    def fit(self, df: pd.DataFrame = None, target_col: str = 'Все товары и услуги') -> 'RidgeSAForecaster':
        """
        Обучение модели на SA данных.

        Args:
            df: Если None, загружает SA данные автоматически.
                Если передан DataFrame, использует его (должен содержать SA данные).
            target_col: Игнорируется (модель всегда прогнозирует по компонентам)
        """
        # Загружаем SA данные
        if df is None:
            self._sa_data = self._load_sa_data()
        else:
            self._sa_data = df

        # Обучаем модель для каждого компонента
        for component in COMPONENTS:
            if component not in self._sa_data.columns:
                raise ValueError(f"Component '{component}' not found in SA data")

            series = self._sa_data[component].dropna()

            # Подготовка признаков
            df_prep = self._prepare_features(series)
            df_clean = df_prep.dropna()

            if len(df_clean) < self.MIN_TRAIN_SIZE:
                raise ValueError(f"Not enough data for {component}: {len(df_clean)} < {self.MIN_TRAIN_SIZE}")

            # Обучение
            X = df_clean[self.FEATURES].values
            y = df_clean['y'].values

            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)

            model = Ridge(alpha=self.alpha)
            model.fit(X_scaled, y)

            self.models[component] = model
            self.scalers[component] = scaler
            self._last_values[component] = series.tail(12)  # Сохраняем последние значения для прогноза

        self._is_fitted = True
        self._last_train_date = self._sa_data.index.max()

        return self

    def _forecast_component(self, component: str, horizon: int) -> np.ndarray:
        """Прогноз для одного компонента."""
        model = self.models[component]
        scaler = self.scalers[component]
        last_vals = self._last_values[component].values

        forecasts = []
        # Буфер для итеративного прогноза
        buffer = list(last_vals[-12:])  # Последние 12 значений

        trend_start = len(self._sa_data)

        for h in range(horizon):
            # Формируем признаки
            y_lag1 = buffer[-1]
            y_lag2 = buffer[-2]
            y_lag3 = buffer[-3]
            y_ma3 = np.mean(buffer[-3:])
            y_ma6 = np.mean(buffer[-6:])
            trend = trend_start + h

            X = np.array([[y_lag1, y_lag2, y_lag3, y_ma3, y_ma6, trend]])
            X_scaled = scaler.transform(X)

            pred = model.predict(X_scaled)[0]
            forecasts.append(pred)
            buffer.append(pred)

        return np.array(forecasts)

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Bottom-Up прогноз: прогнозируем компоненты, агрегируем.

        Args:
            horizon: Горизонт прогноза в месяцах

        Returns:
            numpy array с прогнозами общего ИПЦ (индекс ~100)
        """
        self._check_fitted()

        # Прогнозируем каждый компонент
        component_forecasts = {}
        for component in COMPONENTS:
            component_forecasts[component] = self._forecast_component(component, horizon)

        # Агрегируем по весам
        total_forecast = np.zeros(horizon)
        for component, weight in COMPONENT_WEIGHTS.items():
            total_forecast += component_forecasts[component] * weight

        return total_forecast

    def forecast_components(self, horizon: int = 12) -> Dict[str, np.ndarray]:
        """
        Прогноз по компонентам (для анализа).

        Returns:
            Dict с прогнозами для каждого компонента
        """
        self._check_fitted()

        result = {}
        for component in COMPONENTS:
            result[component] = self._forecast_component(component, horizon)

        # Добавляем агрегированный
        result['Все товары и услуги'] = self.forecast(horizon)

        return result

    def backtest(self, df: pd.DataFrame = None, start_date: str = '2019-01-01',
                 target_col: str = 'Все товары и услуги') -> pd.DataFrame:
        """
        Бэктест модели.

        Args:
            df: SA данные (если None, загружает автоматически)
            start_date: Начало тестового периода
            target_col: Игнорируется

        Returns:
            DataFrame с колонками: date, actual, prediction, error
        """
        if df is None:
            sa_data = self._load_sa_data()
        else:
            sa_data = df

        results = []
        start = pd.to_datetime(start_date)

        # Получаем фактические значения общего ИПЦ
        if 'Все товары и услуги' in sa_data.columns:
            actual_total = sa_data['Все товары и услуги']
        else:
            # Агрегируем из компонентов
            actual_total = sum(
                sa_data[comp] * weight
                for comp, weight in COMPONENT_WEIGHTS.items()
            )

        test_dates = sa_data.index[sa_data.index >= start]

        for date in test_dates:
            cutoff = date - pd.DateOffset(months=1)
            train_data = sa_data[sa_data.index <= cutoff]

            if len(train_data) < self.MIN_TRAIN_SIZE + 12:
                continue

            try:
                # Обучаем на данных до cutoff
                self.fit(train_data)

                # Прогноз на 1 месяц
                pred = self.forecast(horizon=1)[0]
                actual = actual_total.loc[date]

                results.append({
                    'date': date,
                    'actual': actual,
                    'prediction': pred,
                    'error': pred - actual
                })
            except Exception as e:
                continue

        return pd.DataFrame(results)

    def get_component_contributions(self, horizon: int = 12) -> pd.DataFrame:
        """
        Вклад каждого компонента в прогноз.

        Returns:
            DataFrame с колонками: month, component, value, contribution
        """
        self._check_fitted()

        forecasts = self.forecast_components(horizon)

        rows = []
        for h in range(horizon):
            for component in COMPONENTS:
                value = forecasts[component][h]
                weight = COMPONENT_WEIGHTS[component]
                contribution = (value - 100) * weight  # Вклад в MoM

                rows.append({
                    'month': h + 1,
                    'component': component,
                    'value': value,
                    'weight': weight,
                    'contribution': contribution
                })

        return pd.DataFrame(rows)
