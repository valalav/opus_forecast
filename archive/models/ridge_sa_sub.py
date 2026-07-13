"""
Ridge SA Subcomponent: Bottom-Up по субкомпонентам
==================================================

Прогнозирует ~47 субкомпонентов (35 товарных + 12 услуг) отдельно,
затем агрегирует по весам.

Преимущества:
- Более детальная декомпозиция прогноза
- Можно анализировать драйверы инфляции
- Каждый субкомпонент прогнозируется независимо
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from typing import Dict, Optional, List
from pathlib import Path

from .base import BaseForecaster
from .registry import ModelRegistry


# Веса субкомпонентов товаров (из micro_sprav.csv)
GOODS_SUBCOMPONENTS = {
    # Продовольственные товары
    'Мясопродукты': 0.0990,
    'Плодоовощная продукция, включая картофель': 0.0589,
    'Другие продовольственные товары': 0.0367,
    'Молоко и молочная продукция': 0.0305,
    'Кондитерские изделия': 0.0290,
    'Рыбопродукты': 0.0226,
    'Масло и жиры': 0.0173,
    'Макаронные и крупяные изделия': 0.0169,
    'Общественное питание': 0.0158,
    'Чай, кофе, какао': 0.0152,
    'Сыр': 0.0137,
    'Алкогольные напитки': 0.0137,
    'Хлеб и хлебобулочные изделия': 0.0136,
    'Яйца': 0.0061,
    'Сахар': 0.0057,
    # Непродовольственные товары
    'Одежда и белье': 0.0647,
    'Другие непродовольственные товары': 0.0445,
    'Топливо моторное': 0.0330,
    'Легковые автомобили': 0.0307,
    'Обувь кожаная, текстильная и комбинированная': 0.0281,
    'Медицинские товары': 0.0225,
    'Мебель': 0.0207,
    'Парфюмерно-косметические товары': 0.0198,
    'Трикотажные изделия': 0.0160,
    'Электротовары и другие бытовые приборы': 0.0159,
    'Строительные материалы': 0.0148,
    'Галантерея': 0.0148,
    'Моющие и чистящие средства': 0.0112,
    'Средства связи': 0.0091,
    'Табачные изделия': 0.0086,
    'Персональные компьютеры': 0.0049,
    'Телерадиотовары': 0.0025,
    'Меха и меховые изделия': 0.0012,
    'Печатные издания': 0.0011,
    'Инструменты и оборудование': 0.0011,
}

# Веса категорий услуг (сопоставлены с SA данными)
SERVICE_SUBCOMPONENTS = {
    'Другие услуги': 0.0876,
    'Жилищно-коммунальные услуги': 0.0671,
    'Услуги телекоммуникационные (связи)': 0.0202,
    'Бытовые услуги': 0.0186,
    'Услуги пассажирского транспорта': 0.0171,
    'Услуги в системе образования': 0.0097,
    'Медицинские услуги': 0.0041,
    'Железнодорожный транспорт': 0.0039,
    'Услуги гостиниц и прочих мест проживания': 0.0024,
    'Услуги дошкольного воспитания': 0.0019,
    'Санаторно-оздоровительные услуги': 0.0010,
    'Услуги организаций культуры': 0.0007,
}

# Объединённый словарь
ALL_SUBCOMPONENTS = {**GOODS_SUBCOMPONENTS, **SERVICE_SUBCOMPONENTS}


@ModelRegistry.register("ridge_sa_sub")
class RidgeSASubForecaster(BaseForecaster):
    """
    Bottom-Up Ridge по субкомпонентам на SA данных.

    Прогнозирует каждый субкомпонент отдельно, агрегирует по весам.
    """

    name = "ridge_sa_sub"
    MIN_TRAIN_SIZE = 24
    ALPHA = 0.5

    FEATURES = ['y_lag1', 'y_lag2', 'y_lag3', 'y_ma3', 'y_ma6', 'trend']

    def __init__(self, alpha: float = None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha or self.ALPHA
        self.models: Dict[str, Ridge] = {}
        self.scalers: Dict[str, RobustScaler] = {}
        self._last_values: Dict[str, pd.Series] = {}
        self._sa_data: Optional[pd.DataFrame] = None
        self._available_subs: List[str] = []
        self._weights: Dict[str, float] = {}

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
            index='Дата', columns='Товар', values='Значение', aggfunc='first'
        ).sort_index()
        pivot.index.name = 'Date'

        return pivot

    def _prepare_features(self, series: pd.Series) -> pd.DataFrame:
        """Признаки для одного субкомпонента."""
        df = pd.DataFrame({'y': series})
        df['y_lag1'] = df['y'].shift(1)
        df['y_lag2'] = df['y'].shift(2)
        df['y_lag3'] = df['y'].shift(3)
        df['y_ma3'] = df['y'].rolling(3).mean().shift(1)
        df['y_ma6'] = df['y'].rolling(6).mean().shift(1)
        df['trend'] = np.arange(len(df))
        return df

    def fit(self, df: pd.DataFrame = None, target_col: str = None) -> 'RidgeSASubForecaster':
        """Обучение моделей для всех субкомпонентов."""
        if df is None:
            self._sa_data = self._load_sa_data()
        else:
            self._sa_data = df

        # Определяем доступные субкомпоненты
        self._available_subs = []
        self._weights = {}

        for sub, weight in ALL_SUBCOMPONENTS.items():
            if sub in self._sa_data.columns:
                series = self._sa_data[sub].dropna()
                # Проверяем достаточно ли данных
                if len(series) >= self.MIN_TRAIN_SIZE + 6:
                    self._available_subs.append(sub)
                    self._weights[sub] = weight

        # Нормализуем веса
        total_weight = sum(self._weights.values())
        self._weights = {k: v / total_weight for k, v in self._weights.items()}

        # Обучаем модель для каждого субкомпонента
        for sub in self._available_subs:
            series = self._sa_data[sub].dropna()
            df_prep = self._prepare_features(series)
            df_clean = df_prep.dropna()

            X = df_clean[self.FEATURES].values
            y = df_clean['y'].values

            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)

            model = Ridge(alpha=self.alpha)
            model.fit(X_scaled, y)

            self.models[sub] = model
            self.scalers[sub] = scaler
            self._last_values[sub] = series.tail(12)

        self._is_fitted = True
        self._last_train_date = self._sa_data.index.max()

        return self

    def _forecast_subcomponent(self, sub: str, horizon: int) -> np.ndarray:
        """Прогноз для одного субкомпонента."""
        model = self.models[sub]
        scaler = self.scalers[sub]
        buffer = list(self._last_values[sub].values[-12:])
        trend_start = len(self._sa_data)

        forecasts = []
        for h in range(horizon):
            X = np.array([[
                buffer[-1], buffer[-2], buffer[-3],
                np.mean(buffer[-3:]), np.mean(buffer[-6:]),
                trend_start + h
            ]])
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            forecasts.append(pred)
            buffer.append(pred)

        return np.array(forecasts)

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Bottom-Up прогноз: субкомпоненты → агрегат."""
        self._check_fitted()

        total = np.zeros(horizon)
        for sub in self._available_subs:
            fc = self._forecast_subcomponent(sub, horizon)
            total += fc * self._weights[sub]

        return total

    def forecast_subcomponents(self, horizon: int = 12) -> Dict[str, np.ndarray]:
        """Прогнозы по всем субкомпонентам."""
        self._check_fitted()

        result = {}
        for sub in self._available_subs:
            result[sub] = self._forecast_subcomponent(sub, horizon)
        result['Все товары и услуги'] = self.forecast(horizon)

        return result

    def get_contributions(self, horizon: int = 12) -> pd.DataFrame:
        """Вклад каждого субкомпонента в прогноз."""
        self._check_fitted()

        forecasts = self.forecast_subcomponents(horizon)

        rows = []
        for h in range(horizon):
            for sub in self._available_subs:
                value = forecasts[sub][h]
                weight = self._weights[sub]
                contribution = (value - 100) * weight

                rows.append({
                    'month': h + 1,
                    'subcomponent': sub,
                    'value': value,
                    'weight': weight,
                    'contribution_pp': contribution
                })

        return pd.DataFrame(rows)

    def backtest(self, df: pd.DataFrame = None, start_date: str = '2020-01-01',
                 target_col: str = None) -> pd.DataFrame:
        """Бэктест модели."""
        if df is None:
            sa_data = self._load_sa_data()
        else:
            sa_data = df

        # Фактический общий ИПЦ
        if 'Все товары и услуги' in sa_data.columns:
            actual_total = sa_data['Все товары и услуги']
        else:
            raise ValueError("No 'Все товары и услуги' in data")

        results = []
        start = pd.to_datetime(start_date)
        test_dates = sa_data.index[sa_data.index >= start]

        for date in test_dates:
            cutoff = date - pd.DateOffset(months=1)
            train_data = sa_data[sa_data.index <= cutoff]

            if len(train_data) < self.MIN_TRAIN_SIZE + 12:
                continue

            try:
                self.fit(train_data)
                pred = self.forecast(horizon=1)[0]
                actual = actual_total.loc[date]

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
        return {
            'name': self.name,
            'n_subcomponents': len(self._available_subs),
            'subcomponents': self._available_subs,
            'weights': self._weights,
            'total_weight_coverage': sum(self._weights.values()),
            'last_train_date': self._last_train_date
        }
