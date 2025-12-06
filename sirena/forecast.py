"""
Модуль ансамблевого прогнозирования СИРЕНА-КБР
=============================================
"""

from typing import Optional, Dict, Any
from datetime import datetime

import pandas as pd
import numpy as np

from logger import get_logger

logger = get_logger(__name__)


class EnsembleForecaster:
    """Ансамблевый прогноз (Ridge + BVAR + SARIMA)."""

    def __init__(
        self,
        ridge_weight: float = 0.6,
        bvar_weight: float = 0.3,
        sarima_weight: float = 0.1
    ):
        self.ridge_weight = ridge_weight
        self.bvar_weight = bvar_weight
        self.sarima_weight = sarima_weight

        self._ridge_model = None
        self._bvar_model = None
        self._sarima_model = None

    def run_bvar_forecast(
        self,
        horizon: int = 12,
        cutoff_date: Optional[pd.Timestamp] = None
    ) -> Optional[pd.DataFrame]:
        """
        Запуск BVAR прогноза.

        Args:
            horizon: Горизонт прогноза в месяцах
            cutoff_date: Дата отсечки данных

        Returns:
            DataFrame с колонками Date, BVAR
        """
        try:
            from sirena_bvar import BayesianVAR

            # Загрузка данных
            df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')

            cols_to_fix = ['mom', 'Prod', 'Nonprod', 'Serv', 'usd_nom_i', 'Ruonia']
            for col in cols_to_fix:
                if col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str).str.replace(',', '.')
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
            if df['Date'].isna().any():
                df['Date'] = pd.to_datetime(df['Date'])

            df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
            df = df.set_index('Date').sort_index()

            # Подготовка данных
            data = pd.DataFrame()
            data['CPI'] = df['mom'] - 100
            data['Food'] = df['Prod'] - 100
            data['NonFood'] = df['Nonprod'] - 100
            data['Services'] = df['Serv'] - 100
            data['USD'] = df['usd_nom_i'] - 100
            data['RUONIA'] = df['Ruonia']
            data = data.dropna()

            if cutoff_date is not None:
                data = data[data.index <= cutoff_date]

            # Обучение
            model = BayesianVAR(data, ['CPI', 'Food', 'USD', 'RUONIA'], lags=4)
            model.fit(lambda1=1.0, lambda2=0.5, lambda3=1.0)

            # Прогноз
            fc = model.forecast(h=horizon, n_draws=2000)
            median_path = fc['median'][:, 0]

            last_date = data.index.max()
            dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=horizon,
                freq='MS'
            )

            logger.info(f"BVAR прогноз рассчитан на {horizon} месяцев")
            return pd.DataFrame({'Date': dates, 'BVAR': median_path})

        except Exception as e:
            logger.warning(f"BVAR недоступен: {e}")
            return None

    def run_sarima_forecast(
        self,
        horizon: int = 12,
        cutoff_date: Optional[pd.Timestamp] = None
    ) -> Optional[pd.DataFrame]:
        """
        Запуск SARIMA прогноза.

        Args:
            horizon: Горизонт прогноза в месяцах
            cutoff_date: Дата отсечки данных

        Returns:
            DataFrame с колонками Date, SARIMA
        """
        try:
            from sirena_arima import SirenaARIMA

            # Загрузка данных
            df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')

            if 'MoM' in df_raw.columns:
                if df_raw['MoM'].dtype == object:
                    df_raw['MoM'] = df_raw['MoM'].astype(str).str.replace(',', '.')
                df_raw['MoM'] = pd.to_numeric(df_raw['MoM'], errors='coerce')

            if 'Day' in df_raw.columns:
                df_raw['Date'] = pd.to_datetime(df_raw['Day'], format='%d.%m.%Y', errors='coerce')
            elif 'Date' in df_raw.columns:
                df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')

            if 'Товар' in df_raw.columns and 'MoM' in df_raw.columns:
                df = df_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
            else:
                df = df_raw.set_index('Date')

            df = df.sort_index()

            if cutoff_date is not None:
                df = df[df.index <= cutoff_date]

            ts = df['Все товары и услуги'].dropna() - 100

            # Обучение
            model = SirenaARIMA()
            model.fit_sarima(ts)
            fc = model.forecast(steps=horizon)

            last_date = ts.index.max()
            dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=horizon,
                freq='MS'
            )

            logger.info(f"SARIMA прогноз рассчитан на {horizon} месяцев")
            return pd.DataFrame({'Date': dates, 'SARIMA': fc['mean'].values})

        except Exception as e:
            logger.warning(f"SARIMA недоступна: {e}")
            return None

    def combine_forecasts(
        self,
        ridge_forecast: pd.DataFrame,
        bvar_forecast: Optional[pd.DataFrame],
        sarima_forecast: Optional[pd.DataFrame]
    ) -> Optional[np.ndarray]:
        """
        Объединение прогнозов в ансамбль.

        Args:
            ridge_forecast: DataFrame с прогнозом Ridge (колонка MoM)
            bvar_forecast: DataFrame с прогнозом BVAR
            sarima_forecast: DataFrame с прогнозом SARIMA

        Returns:
            numpy array с ансамблевым прогнозом
        """
        if bvar_forecast is None or sarima_forecast is None:
            logger.warning("Не все модели доступны, ансамбль не рассчитан")
            return None

        ridge_vals = ridge_forecast['MoM'].values
        bvar_vals = bvar_forecast['BVAR'].values
        sarima_vals = sarima_forecast['SARIMA'].values

        if len(ridge_vals) != len(bvar_vals) or len(ridge_vals) != len(sarima_vals):
            logger.error("Несовпадение длины прогнозов")
            return None

        ensemble = (
            ridge_vals * self.ridge_weight +
            bvar_vals * self.bvar_weight +
            sarima_vals * self.sarima_weight
        )

        logger.info(f"Ансамбль рассчитан: Ridge({self.ridge_weight}) + "
                   f"BVAR({self.bvar_weight}) + SARIMA({self.sarima_weight})")
        return ensemble


def calculate_cumulative_inflation(mom_values: np.ndarray) -> float:
    """
    Расчёт накопленной инфляции.

    Args:
        mom_values: Массив месячных индексов (MoM в формате индекса, например 100.5)

    Returns:
        Накопленная инфляция в процентах
    """
    return (np.prod(mom_values / 100) - 1) * 100


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray
) -> Dict[str, float]:
    """
    Расчёт метрик качества прогноза.

    Args:
        actual: Фактические значения
        predicted: Прогнозные значения

    Returns:
        Словарь с MAE, RMSE, KPI
    """
    errors = actual - predicted
    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())
    kpi_count = (np.abs(errors) <= 0.5).sum()
    kpi_pct = kpi_count / len(errors) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'KPI_count': kpi_count,
        'KPI_total': len(errors),
        'KPI_pct': kpi_pct
    }
