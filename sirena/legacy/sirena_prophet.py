"""
Prophet модель для прогнозирования инфляции КБР
===============================================

Facebook Prophet - модель для автоматического определения
тренда и сезонности с устойчивостью к выбросам.

Автор: СИРЕНА-КБР
Версия: 1.0
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not installed. Run: pip install prophet")


class SirenaProphet:
    """
    Prophet модель для прогнозирования инфляции.

    Prophet автоматически обрабатывает:
    - Сезонность (годовая, недельная)
    - Тренд (линейный или логистический)
    - Праздники и аномалии
    """

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = False,
        daily_seasonality: bool = False,
        seasonality_mode: str = 'additive',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        outlier_years: list = None
    ):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not installed")

        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.outlier_years = outlier_years or [2022]

        self.model = None
        self.last_date = None

    def _prepare_prophet_df(
        self,
        df: pd.DataFrame,
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """Преобразование данных в формат Prophet (ds, y)."""
        if target_col in df.columns:
            series = df[target_col].dropna()
        else:
            series = df.dropna()

        # Конвертируем в проценты если нужно
        if series.mean() > 50:
            series = series - 100

        prophet_df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })

        return prophet_df

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'Все товары и услуги'
    ):
        """
        Обучение модели.

        Args:
            df: DataFrame с данными
            target_col: Целевая колонка

        Returns:
            self
        """
        prophet_df = self._prepare_prophet_df(df, target_col)

        # Исключаем годы-выбросы
        prophet_df['year'] = prophet_df['ds'].dt.year
        prophet_df = prophet_df[~prophet_df['year'].isin(self.outlier_years)]
        prophet_df = prophet_df.drop('year', axis=1)

        self.last_date = prophet_df['ds'].max()

        # Создаём и обучаем модель
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale
        )

        # Добавляем кастомную месячную сезонность
        self.model.add_seasonality(
            name='monthly',
            period=30.5,
            fourier_order=5
        )

        # Отключаем вывод
        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

        self.model.fit(prophet_df)

        return self

    def forecast(self, horizon: int = 12) -> dict:
        """
        Прогноз на заданный горизонт.

        Args:
            horizon: Количество месяцев

        Returns:
            dict с mean, lower, upper
        """
        if self.model is None:
            raise ValueError("Модель не обучена")

        # Создаём future dataframe
        future = self.model.make_future_dataframe(
            periods=horizon,
            freq='MS'  # Month Start
        )

        # Прогноз
        forecast = self.model.predict(future)

        # Берём только будущие периоды
        forecast_future = forecast[forecast['ds'] > self.last_date]

        return {
            'mean': forecast_future['yhat'].values,
            'lower': forecast_future['yhat_lower'].values,
            'upper': forecast_future['yhat_upper'].values,
            'dates': forecast_future['ds'].values,
            'trend': forecast_future['trend'].values,
            'yearly': forecast_future['yearly'].values if 'yearly' in forecast_future.columns else None
        }

    def get_components(self) -> pd.DataFrame:
        """Получить компоненты модели."""
        if self.model is None:
            return pd.DataFrame()

        future = self.model.make_future_dataframe(periods=0, freq='MS')
        forecast = self.model.predict(future)

        return forecast[['ds', 'trend', 'yearly', 'yhat']]

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
            start_date: Начальная дата
            target_col: Целевая колонка

        Returns:
            DataFrame с результатами
        """
        if target_col in df.columns:
            series = df[target_col].dropna()
        else:
            series = df.dropna()

        # Конвертируем если нужно
        original_values = series.copy()
        if series.mean() > 50:
            series = series - 100

        test_dates = series[series.index >= start_date].index
        results = []

        for target_date in test_dates:
            cutoff = target_date - pd.DateOffset(months=1)
            train_series = series[series.index <= cutoff]

            if len(train_series) < 24:
                continue

            try:
                # Создаём DataFrame для обучения
                train_df = pd.DataFrame({
                    target_col: train_series.values + 100  # Возвращаем в исходный формат
                }, index=train_series.index)

                model = SirenaProphet(
                    yearly_seasonality=self.yearly_seasonality,
                    seasonality_mode=self.seasonality_mode,
                    outlier_years=self.outlier_years
                )
                model.fit(train_df, target_col)
                fc = model.forecast(horizon=1)

                prediction = fc['mean'][0]
                actual = series.loc[target_date]

                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': prediction,
                    'error': actual - prediction
                })
            except Exception:
                continue

        return pd.DataFrame(results)

    def plot_components(self):
        """Визуализация компонентов (требует matplotlib)."""
        if self.model is None:
            return None

        import matplotlib.pyplot as plt

        future = self.model.make_future_dataframe(periods=12, freq='MS')
        forecast = self.model.predict(future)

        fig = self.model.plot_components(forecast)
        return fig


# Тестирование
if __name__ == "__main__":
    if not PROPHET_AVAILABLE:
        print("Prophet not available")
        exit()

    # Загрузка данных
    df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal='.')

    try:
        df_raw['Date'] = pd.to_datetime(df_raw['Day'], format='%d.%m.%Y')
    except:
        df_raw['Date'] = pd.to_datetime(df_raw['Day'])

    if 'Товар' in df_raw.columns:
        df = df_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
    else:
        df = df_raw.set_index('Date')

    df = df.sort_index()

    print("Обучение Prophet модели...")
    model = SirenaProphet()
    model.fit(df, 'Все товары и услуги')

    # Прогноз
    fc = model.forecast(12)
    print("\nProphet Прогноз на 12 месяцев (MoM %):")
    for i, (d, v) in enumerate(zip(fc['dates'], fc['mean'])):
        print(f"  {pd.Timestamp(d).strftime('%Y-%m')}: {v:.2f}%")

    # Бэктест
    print("\nЗапуск бэктеста (может занять время)...")
    bt = model.backtest(df, start_date='2024-01-01')  # Короткий бэктест
    if not bt.empty:
        mae = bt['error'].abs().mean()
        print(f"\nMAE на бэктесте: {mae:.3f}")
