"""
Модель прогнозирования недельных цен СИРЕНА-КБР
================================================

WeeklyForecaster - прогнозирование недельной динамики цен
на основе данных Росстата по КБР.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any, Union
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import warnings

from .base import BaseForecaster
from .registry import ModelRegistry

warnings.filterwarnings('ignore')


@ModelRegistry.register("weekly")
class WeeklyForecaster(BaseForecaster):
    """
    Модель прогнозирования недельных цен.

    Использует Ridge регрессию с сезонными признаками для прогноза
    недельной динамики цен (WoW - week-over-week).

    Поддерживает:
    - Агрегированный прогноз (средний WoW по корзине)
    - Прогноз отдельных товаров
    - Прогноз групп товаров (продовольствие, непродовольствие)

    Example:
        model = WeeklyForecaster()
        model.fit(df)
        forecast = model.forecast(horizon=4)  # 4 недели
    """

    name = "weekly"
    MIN_TRAIN_SIZE = 52  # Минимум год данных

    # Ключевые товары для агрегата
    KEY_PRODUCTS = [
        'Говядина (кроме бескостного мяса), кг',
        'Свинина (кроме бескостного мяса), кг',
        'Куры охлажденные и мороженые, кг',
        'Молоко пастеризованное, л',
        'Масло сливочное, кг',
        'Сыры сычужные твердые и мягкие, кг',
        'Яйца куриные, 10 шт.',
        'Сахар-песок, кг',
        'Масло подсолнечное, кг',
        'Мука пшеничная, кг',
        'Хлеб и булочные изделия из пшеничной муки, кг',
        'Рис шлифованный, кг',
        'Картофель, кг',
        'Капуста белокочанная свежая, кг',
        'Лук репчатый, кг',
        'Морковь, кг',
        'Яблоки, кг',
        'Бананы, кг',
        'Помидоры свежие, кг',
        'Огурцы свежие, кг',
    ]

    # Плодоовощи (высокая волатильность)
    FRUITS_VEG = [
        'Картофель, кг',
        'Капуста белокочанная свежая, кг',
        'Лук репчатый, кг',
        'Морковь, кг',
        'Яблоки, кг',
        'Бананы, кг',
        'Помидоры свежие, кг',
        'Огурцы свежие, кг',
    ]

    def __init__(self, alpha: float = 0.5, **kwargs):
        """
        Инициализация модели.

        Args:
            alpha: Параметр регуляризации Ridge
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.ridge = Ridge(alpha=alpha)
        self.scaler = RobustScaler()

        self.df_wide: Optional[pd.DataFrame] = None
        self.df_wow: Optional[pd.DataFrame] = None
        self.seasonal_pattern: Optional[pd.Series] = None
        self.last_values: Optional[pd.Series] = None
        self.available_products: List[str] = []

    def load_data(self, path: str = 'data/weekly_prices.csv') -> pd.DataFrame:
        """
        Загрузка данных из CSV.

        Args:
            path: Путь к файлу с недельными ценами

        Returns:
            DataFrame в wide формате (индекс - даты, колонки - товары)
        """
        df = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8-sig')

        # Преобразуем в wide format
        df_wide = df.pivot_table(
            index='Сведено',
            columns='Товары',
            values='Значение',
            aggfunc='first'
        )

        # Преобразуем индекс YYYY_WW в даты
        dates = []
        for idx in df_wide.index:
            try:
                year, week = str(idx).split('_')
                # ISO week to date (Monday of that week)
                date = pd.Timestamp.fromisocalendar(int(year), int(week), 1)
                dates.append(date)
            except:
                dates.append(pd.NaT)

        df_wide.index = pd.DatetimeIndex(dates)
        df_wide = df_wide.dropna(how='all')
        df_wide = df_wide.sort_index()

        return df_wide

    def fit(
        self,
        df: Optional[pd.DataFrame] = None,
        target_col: str = 'aggregate',
        data_path: str = 'data/weekly_prices.csv'
    ) -> 'WeeklyForecaster':
        """
        Обучение модели.

        Args:
            df: DataFrame в wide формате (опционально, иначе загружается из файла)
            target_col: Целевая колонка или 'aggregate' для среднего по корзине
            data_path: Путь к данным (если df не передан)

        Returns:
            self
        """
        # Загружаем данные
        if df is None:
            self.df_wide = self.load_data(data_path)
        else:
            self.df_wide = df.copy()

        self.available_products = [p for p in self.KEY_PRODUCTS if p in self.df_wide.columns]

        # Вычисляем WoW (week-over-week) изменения в %
        self.df_wow = self.df_wide.pct_change() * 100
        self.df_wow = self.df_wow.replace([np.inf, -np.inf], np.nan)

        # Создаём агрегированный ряд (медиана по ключевым товарам)
        if self.available_products:
            self.df_wow['aggregate'] = self.df_wow[self.available_products].median(axis=1)
        else:
            # Fallback: медиана по всем товарам
            self.df_wow['aggregate'] = self.df_wow.median(axis=1)

        # Плодоовощной индекс
        fv_cols = [p for p in self.FRUITS_VEG if p in self.df_wide.columns]
        if fv_cols:
            self.df_wow['fruits_veg'] = self.df_wow[fv_cols].median(axis=1)

        # Целевой ряд
        target = target_col if target_col in self.df_wow.columns else 'aggregate'
        y = self.df_wow[target].dropna()

        if len(y) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных: {len(y)} < {self.MIN_TRAIN_SIZE}")

        # Сезонный паттерн (по неделям года)
        self.seasonal_pattern = y.groupby(y.index.isocalendar().week).median()

        # Признаки для Ridge
        X = self._create_features(y)
        y_clean = y.loc[X.index]

        # Обучаем
        X_scaled = self.scaler.fit_transform(X)
        self.ridge.fit(X_scaled, y_clean)

        # Сохраняем последние значения для прогноза
        self.last_values = self.df_wow.iloc[-1]
        self._last_train_date = self.df_wow.index[-1]
        self._is_fitted = True

        return self

    def _create_features(self, y: pd.Series) -> pd.DataFrame:
        """Создание признаков для модели."""
        df = pd.DataFrame(index=y.index)

        # Лаги
        df['lag1'] = y.shift(1)
        df['lag2'] = y.shift(2)
        df['lag4'] = y.shift(4)  # Месяц назад
        df['lag52'] = y.shift(52)  # Год назад

        # Скользящие средние
        df['ma4'] = y.rolling(4).mean()
        df['ma12'] = y.rolling(12).mean()

        # Сезонность (неделя года)
        week_num = y.index.isocalendar().week
        df['week_sin'] = np.sin(2 * np.pi * week_num / 52)
        df['week_cos'] = np.cos(2 * np.pi * week_num / 52)

        # Месяц
        month = y.index.month
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)

        # Сезонная норма
        df['seasonal_norm'] = week_num.map(self.seasonal_pattern).values

        return df.dropna()

    def forecast(
        self,
        horizon: int = 4,
        target_col: str = 'aggregate'
    ) -> np.ndarray:
        """
        Прогноз на заданный горизонт (в неделях).

        Args:
            horizon: Количество недель для прогноза
            target_col: 'aggregate', 'fruits_veg' или название товара

        Returns:
            numpy array с прогнозами WoW в %
        """
        self._check_fitted()

        target = target_col if target_col in self.df_wow.columns else 'aggregate'
        y = self.df_wow[target].dropna()

        forecasts = []
        current_values = y.values.copy()
        last_date = y.index[-1]

        for h in range(horizon):
            # Следующая дата
            next_date = last_date + pd.Timedelta(weeks=h+1)
            week_num = next_date.isocalendar().week
            month = next_date.month

            # Признаки для прогноза
            features = {
                'lag1': current_values[-1] if len(current_values) > 0 else 0,
                'lag2': current_values[-2] if len(current_values) > 1 else 0,
                'lag4': current_values[-4] if len(current_values) > 3 else 0,
                'lag52': current_values[-52] if len(current_values) > 51 else 0,
                'ma4': np.mean(current_values[-4:]) if len(current_values) >= 4 else np.mean(current_values),
                'ma12': np.mean(current_values[-12:]) if len(current_values) >= 12 else np.mean(current_values),
                'week_sin': np.sin(2 * np.pi * week_num / 52),
                'week_cos': np.cos(2 * np.pi * week_num / 52),
                'month_sin': np.sin(2 * np.pi * month / 12),
                'month_cos': np.cos(2 * np.pi * month / 12),
                'seasonal_norm': self.seasonal_pattern.get(week_num, 0),
            }

            X = pd.DataFrame([features])
            X_scaled = self.scaler.transform(X)
            pred = self.ridge.predict(X_scaled)[0]

            forecasts.append(pred)
            current_values = np.append(current_values, pred)

        return np.array(forecasts)

    def forecast_products(
        self,
        products: Optional[List[str]] = None,
        horizon: int = 4
    ) -> Dict[str, np.ndarray]:
        """
        Прогноз для отдельных товаров.

        Args:
            products: Список товаров (None = ключевые товары)
            horizon: Горизонт в неделях

        Returns:
            Dict с прогнозами по товарам
        """
        self._check_fitted()

        if products is None:
            products = self.available_products

        results = {}
        for product in products:
            if product in self.df_wow.columns:
                y = self.df_wow[product].dropna()
                if len(y) >= self.MIN_TRAIN_SIZE:
                    # Простой прогноз: сезонная норма + MA
                    forecasts = []
                    last_date = y.index[-1]

                    for h in range(horizon):
                        next_date = last_date + pd.Timedelta(weeks=h+1)
                        week_num = next_date.isocalendar().week

                        seasonal = self.seasonal_pattern.get(week_num, 0)
                        ma = y.iloc[-4:].mean()
                        pred = 0.5 * seasonal + 0.5 * ma
                        forecasts.append(pred)

                    results[product] = np.array(forecasts)

        return results

    def backtest(
        self,
        df: Optional[pd.DataFrame] = None,
        start_date: str = '2024-01-01',
        target_col: str = 'aggregate'
    ) -> pd.DataFrame:
        """
        Бэктестирование модели.

        Args:
            df: DataFrame (опционально)
            start_date: Начальная дата бэктеста
            target_col: Целевая колонка

        Returns:
            DataFrame с результатами
        """
        if df is None and self.df_wow is None:
            raise ValueError("Нет данных для бэктеста. Вызовите fit() сначала.")

        wow = self.df_wow if df is None else df
        target = target_col if target_col in wow.columns else 'aggregate'
        y = wow[target].dropna()

        start = pd.Timestamp(start_date)
        results = []

        for i, date in enumerate(y.index):
            if date < start:
                continue
            if i < self.MIN_TRAIN_SIZE:
                continue

            # Обучаем на данных до date
            train_data = self.df_wide.loc[:date - pd.Timedelta(weeks=1)]
            if len(train_data) < self.MIN_TRAIN_SIZE:
                continue

            try:
                self.fit(train_data)
                pred = self.forecast(horizon=1)[0]
                actual = y.loc[date]

                results.append({
                    'date': date,
                    'actual': actual,
                    'prediction': pred,
                    'error': pred - actual
                })
            except Exception as e:
                continue

        return pd.DataFrame(results)

    def get_metrics(self, backtest_results: pd.DataFrame) -> Dict[str, float]:
        """Расчёт метрик качества."""
        if backtest_results.empty:
            return {'MAE': np.nan, 'RMSE': np.nan, 'KPI': np.nan}

        errors = backtest_results['error'].abs()

        return {
            'MAE': errors.mean(),
            'RMSE': np.sqrt((backtest_results['error'] ** 2).mean()),
            'KPI': (errors <= 0.5).mean() * 100,  # % прогнозов с ошибкой ≤ 0.5%
            'n_obs': len(backtest_results)
        }

    def get_current_signal(self) -> Dict[str, Any]:
        """
        Получить текущий сигнал для nowcasting месячной инфляции.

        Returns:
            Dict с сигналами от недельных данных
        """
        self._check_fitted()

        # Последние 4 недели (текущий месяц)
        last_4w = self.df_wow.iloc[-4:]

        # Агрегированный сигнал
        agg_signal = last_4w['aggregate'].mean() if 'aggregate' in last_4w.columns else np.nan

        # Плодоовощной сигнал
        fv_signal = last_4w['fruits_veg'].mean() if 'fruits_veg' in last_4w.columns else np.nan

        # Прогноз на следующие 4 недели
        forecast_4w = self.forecast(horizon=4)

        return {
            'last_date': self._last_train_date,
            'last_4w_avg': agg_signal,
            'fruits_veg_4w': fv_signal,
            'forecast_4w': forecast_4w.tolist(),
            'forecast_sum': forecast_4w.sum(),
            'monthly_proxy': agg_signal * 4.33,  # Аппроксимация месячного прогноза
        }

    def __repr__(self) -> str:
        n_products = len(self.available_products) if self.available_products else 0
        return f"WeeklyForecaster(fitted={self._is_fitted}, products={n_products})"
