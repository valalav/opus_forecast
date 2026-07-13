"""
Расчёт YoY (year-over-year) инфляции
====================================

Модуль для конвертации MoM (month-over-month) прогнозов в YoY.

Формула:
    YoY(t) = Π(1 + MoM_i/100) - 1  для i = t-11, ..., t

Пример:
    >>> mom_series = pd.Series([0.5, 0.3, 0.8, ...])  # 12 месяцев MoM %
    >>> yoy = calculate_yoy(mom_series)
    >>> print(f"YoY инфляция: {yoy:.2f}%")
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List


def calculate_yoy(mom_series: pd.Series) -> float:
    """
    Расчёт YoY из последних 12 MoM значений.

    Args:
        mom_series: Series с MoM в процентах (0.5 = 0.5%)

    Returns:
        YoY в процентах (5.0 = 5.0%)

    Example:
        >>> mom = pd.Series([0.84, 1.34, 0.81, 0.53, -0.07, -0.04,
        ...                  0.36, -0.57, 1.0, 0.53, 0.15, 0.19])
        >>> yoy = calculate_yoy(mom)
        >>> print(f"{yoy:.2f}%")  # ~5.17%
    """
    if len(mom_series) < 12:
        raise ValueError(f"Нужно минимум 12 месяцев, получено {len(mom_series)}")

    last_12 = mom_series.tail(12).values
    indices = 1 + last_12 / 100  # MoM% → индекс
    yoy = np.prod(indices) - 1  # кумулятивный рост
    return yoy * 100  # в процентах


def calculate_yoy_from_mom(mom_values: Union[List[float], np.ndarray]) -> float:
    """
    Расчёт YoY из массива MoM значений.

    Args:
        mom_values: Массив MoM в процентах (12 значений)

    Returns:
        YoY в процентах
    """
    mom_arr = np.array(mom_values)
    if len(mom_arr) != 12:
        raise ValueError(f"Нужно ровно 12 месяцев, получено {len(mom_arr)}")

    indices = 1 + mom_arr / 100
    yoy = np.prod(indices) - 1
    return yoy * 100


def forecast_yoy(
    mom_history: pd.Series,
    mom_forecast: Union[List[float], np.ndarray],
    return_full: bool = True
) -> Union[pd.Series, float]:
    """
    Расчёт YoY на прогнозном горизонте.

    Объединяет историю MoM с прогнозом и рассчитывает rolling YoY.

    Args:
        mom_history: Исторические MoM значения (индекс = даты)
        mom_forecast: Прогнозные MoM значения
        return_full: True = вернуть Series с YoY для каждого месяца,
                     False = вернуть только последний YoY

    Returns:
        Series с YoY или float (последний YoY)

    Example:
        >>> history = df['MoM'].dropna()  # до ноября 2025
        >>> forecast = [0.19, 0.5, 0.6, ...]  # декабрь 2025+
        >>> yoy_series = forecast_yoy(history, forecast)
    """
    # Создаём индекс для прогнозных дат
    last_date = mom_history.index.max()
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=len(mom_forecast),
        freq='MS'
    )

    # Объединяем историю и прогноз
    forecast_series = pd.Series(mom_forecast, index=forecast_dates)
    combined = pd.concat([mom_history, forecast_series])

    # Rolling YoY (последние 12 месяцев)
    yoy = combined.rolling(12).apply(
        lambda x: (np.prod(1 + x / 100) - 1) * 100,
        raw=True
    )

    if return_full:
        # Возвращаем только прогнозную часть YoY
        return yoy.loc[forecast_dates]
    else:
        return yoy.iloc[-1]


def mom_to_yoy_series(mom_series: pd.Series) -> pd.Series:
    """
    Конвертация всего ряда MoM в YoY (rolling 12 месяцев).

    Args:
        mom_series: Series с MoM значениями

    Returns:
        Series с YoY значениями (первые 11 = NaN)

    Example:
        >>> yoy = mom_to_yoy_series(df['Все товары и услуги'])
        >>> yoy.plot(title='YoY инфляция')
    """
    yoy = mom_series.rolling(12).apply(
        lambda x: (np.prod(1 + x / 100) - 1) * 100,
        raw=True
    )
    return yoy


def calculate_cumulative_index(mom_series: pd.Series, base: float = 100.0) -> pd.Series:
    """
    Расчёт кумулятивного индекса из MoM.

    Args:
        mom_series: Series с MoM в процентах
        base: Базовое значение индекса (по умолчанию 100)

    Returns:
        Series с кумулятивным индексом

    Example:
        >>> idx = calculate_cumulative_index(mom_series, base=100)
        >>> # idx[0] = 100, idx[1] = 100.5, idx[2] = 100.8, ...
    """
    indices = 1 + mom_series / 100
    cumulative = indices.cumprod() * base / indices.iloc[0]
    return cumulative


def yoy_at_horizon(
    mom_history: pd.Series,
    mom_forecast: Union[List[float], np.ndarray],
    horizon: int
) -> float:
    """
    YoY на конкретном горизонте прогноза.

    Args:
        mom_history: Исторические MoM
        mom_forecast: Прогнозные MoM
        horizon: Горизонт (1, 2, 3, 6, 12 месяцев)

    Returns:
        YoY на заданном горизонте в процентах

    Example:
        >>> yoy_h12 = yoy_at_horizon(history, forecast, horizon=12)
    """
    if horizon > len(mom_forecast):
        raise ValueError(f"Горизонт {horizon} > длина прогноза {len(mom_forecast)}")

    # Берём только нужное количество прогнозных месяцев
    forecast_trimmed = mom_forecast[:horizon]

    # Формируем полный ряд: история + прогноз
    last_date = mom_history.index.max()
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=horizon,
        freq='MS'
    )
    forecast_series = pd.Series(forecast_trimmed, index=forecast_dates)
    combined = pd.concat([mom_history, forecast_series])

    # Берём последние 12 месяцев (заканчивая на горизонте)
    target_date = forecast_dates[-1]
    start_date = target_date - pd.DateOffset(months=11)

    last_12 = combined.loc[start_date:target_date]
    if len(last_12) < 12:
        # Если не хватает данных, берём последние 12
        last_12 = combined.tail(12)

    yoy = (np.prod(1 + last_12.values / 100) - 1) * 100
    return yoy


# === Вспомогательные функции ===

def format_yoy_table(
    mom_history: pd.Series,
    mom_forecast: np.ndarray,
    horizons: List[int] = [1, 2, 3, 6, 12]
) -> pd.DataFrame:
    """
    Создание таблицы YoY для разных горизонтов.

    Args:
        mom_history: Исторические MoM
        mom_forecast: Прогнозные MoM (минимум max(horizons) месяцев)
        horizons: Список горизонтов

    Returns:
        DataFrame с колонками: Горизонт, Дата, MoM%, YoY%
    """
    last_date = mom_history.index.max()
    results = []

    for h in horizons:
        if h > len(mom_forecast):
            continue

        target_date = last_date + pd.DateOffset(months=h)
        mom_h = mom_forecast[h - 1]
        yoy_h = yoy_at_horizon(mom_history, mom_forecast, h)

        results.append({
            'Горизонт': f'h={h}',
            'Дата': target_date.strftime('%Y-%m'),
            'MoM%': mom_h,
            'YoY%': yoy_h
        })

    return pd.DataFrame(results)
