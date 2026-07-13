"""
ETS (Exponential Smoothing) модель для прогнозирования инфляции КБР
===================================================================

Использует Holt-Winters экспоненциальное сглаживание из statsmodels.
Поддерживает сезонность и тренд.

Автор: СИРЕНА-КБР
Версия: 1.0
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings('ignore')


class SirenaETS:
    """
    ETS (Error-Trend-Seasonality) модель.

    Параметры:
        trend: 'add' или 'mul' (аддитивный/мультипликативный тренд)
        seasonal: 'add' или 'mul' (аддитивная/мультипликативная сезонность)
        seasonal_periods: период сезонности (12 для месячных данных)
    """

    def __init__(
        self,
        trend: str = 'add',
        seasonal: str = 'add',
        seasonal_periods: int = 12,
        damped_trend: bool = False
    ):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.model = None
        self.fit_result = None
        self.last_values = None

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги'):
        """
        Обучение модели.

        Args:
            df: DataFrame с индексом datetime и целевой переменной
            target_col: Название колонки с целевой переменной

        Returns:
            self
        """
        # Извлекаем временной ряд
        if target_col in df.columns:
            series = df[target_col].dropna()
        else:
            # Предполагаем, что df - это Series
            series = df.dropna()

        # Преобразуем в MoM (если индексы > 50, значит это индексы типа 100.5)
        if series.mean() > 50:
            series = series - 100  # Конвертируем в проценты

        self.last_values = series.values

        # Для ETS нужно минимум 2 полных сезона
        if len(series) < 2 * self.seasonal_periods:
            # Fallback на простую модель без сезонности
            self.model = ExponentialSmoothing(
                series,
                trend=self.trend,
                seasonal=None,
                damped_trend=self.damped_trend
            )
        else:
            self.model = ExponentialSmoothing(
                series,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                damped_trend=self.damped_trend
            )

        self.fit_result = self.model.fit(optimized=True)
        return self

    def forecast(self, horizon: int = 12) -> dict:
        """
        Прогноз на заданный горизонт.

        Args:
            horizon: Количество периодов для прогноза

        Returns:
            dict с ключами: mean, lower, upper (95% CI)
        """
        if self.fit_result is None:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")

        # Прогноз
        forecast = self.fit_result.forecast(steps=horizon)

        # ETS не даёт готовых интервалов, оцениваем через residuals
        residuals = self.fit_result.resid
        std_resid = np.std(residuals)

        # 95% доверительный интервал (примерно)
        z = 1.96
        lower = forecast - z * std_resid * np.sqrt(np.arange(1, horizon + 1))
        upper = forecast + z * std_resid * np.sqrt(np.arange(1, horizon + 1))

        return {
            'mean': forecast.values,
            'lower': lower.values,
            'upper': upper.values,
            'aic': self.fit_result.aic if hasattr(self.fit_result, 'aic') else None
        }

    def get_components(self) -> dict:
        """
        Получить компоненты модели (тренд, сезонность, уровень).

        Returns:
            dict с компонентами
        """
        if self.fit_result is None:
            return {}

        return {
            'level': self.fit_result.level,
            'trend': self.fit_result.trend if hasattr(self.fit_result, 'trend') else None,
            'season': self.fit_result.season if hasattr(self.fit_result, 'season') else None,
            'resid': self.fit_result.resid
        }

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
            DataFrame с результатами бэктеста
        """
        if target_col in df.columns:
            series = df[target_col].dropna()
        else:
            series = df.dropna()

        # Конвертируем если нужно
        if series.mean() > 50:
            series = series - 100

        test_dates = series[series.index >= start_date].index
        results = []

        for target_date in test_dates:
            # Обучаемся на данных до target_date
            cutoff = target_date - pd.DateOffset(months=1)
            train_data = series[series.index <= cutoff]

            if len(train_data) < 24:  # Минимум 2 года
                continue

            try:
                # Создаём новый экземпляр для чистого бэктеста
                model = SirenaETS(
                    trend=self.trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonal_periods
                )
                model.fit(train_data)
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


# Тестирование
if __name__ == "__main__":
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

    # Обучение
    model = SirenaETS()
    model.fit(df, 'Все товары и услуги')

    # Прогноз
    fc = model.forecast(12)
    print("ETS Прогноз на 12 месяцев:")
    print(fc['mean'])

    # Бэктест
    bt = model.backtest(df, start_date='2023-01-01')
    if not bt.empty:
        mae = bt['error'].abs().mean()
        print(f"\nMAE на бэктесте: {mae:.3f}")
