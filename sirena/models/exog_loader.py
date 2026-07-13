#!/usr/bin/env python3
"""
ЗАГРУЗЧИК ЭКЗОГЕННЫХ ПЕРЕМЕННЫХ
===============================

Логика загрузки:
1. Если есть exog_manual.csv — использовать его (ручные корректировки)
2. Иначе использовать exog_forecast.csv (автопрогноз)

Файлы:
- data/exog_forecast.csv — автоматический прогноз от ExogForecaster
- data/exog_manual.csv — ручные корректировки пользователя

Использование:
    from sirena.models.exog_loader import load_exog_data, get_exog_for_date

    # Загрузить данные (manual если есть, иначе forecast)
    exog_df = load_exog_data()

    # Получить значения для конкретной даты
    values = get_exog_for_date(target_date)
    # {'Ki': 16.0, 'Ruonia': 15.75, 'USD_ABS': 78.0, 'Brent': 60.0}
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import shutil


# Paths
DATA_DIR = Path(__file__).parent.parent.parent / 'data'
FORECAST_FILE = DATA_DIR / 'exog_forecast.csv'
MANUAL_FILE = DATA_DIR / 'exog_manual.csv'


def load_exog_data(prefer_manual: bool = True) -> Optional[pd.DataFrame]:
    """
    Загрузка экзогенных данных.

    Args:
        prefer_manual: Если True, использовать manual файл если он существует

    Returns:
        DataFrame с экзогенными данными или None если файлов нет
    """
    # Determine which file to use
    if prefer_manual and MANUAL_FILE.exists():
        filepath = MANUAL_FILE
        source = 'manual'
    elif FORECAST_FILE.exists():
        filepath = FORECAST_FILE
        source = 'forecast'
    else:
        return None

    # Load data
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    # Add source column
    df['_source'] = source

    return df


def get_exog_for_date(target_date: pd.Timestamp,
                       prefer_manual: bool = True) -> Optional[Dict[str, float]]:
    """
    Получить значения экзогенных для конкретной даты.

    Args:
        target_date: Целевая дата
        prefer_manual: Использовать manual файл если есть

    Returns:
        Dict с Ki, Ruonia, USD_ABS, Brent или None
    """
    df = load_exog_data(prefer_manual)
    if df is None:
        return None

    # Normalize date to first of month
    target_norm = target_date.to_period('M').to_timestamp()

    if target_norm in df.index:
        row = df.loc[target_norm]
        return {
            'Ki': row.get('Ki', np.nan),
            'Ruonia': row.get('Ruonia', np.nan),
            'USD_ABS': row.get('USD_ABS', np.nan),
            'Brent': row.get('Brent', np.nan),
        }

    # Fallback: use last available value
    if len(df) > 0:
        row = df.iloc[-1]
        return {
            'Ki': row.get('Ki', np.nan),
            'Ruonia': row.get('Ruonia', np.nan),
            'USD_ABS': row.get('USD_ABS', np.nan),
            'Brent': row.get('Brent', np.nan),
        }

    return None


def copy_forecast_to_manual(overwrite: bool = False) -> bool:
    """
    Скопировать forecast в manual файл.

    Args:
        overwrite: Перезаписать если manual уже существует

    Returns:
        True если успешно
    """
    if not FORECAST_FILE.exists():
        return False

    if MANUAL_FILE.exists() and not overwrite:
        return False

    shutil.copy(FORECAST_FILE, MANUAL_FILE)
    return True


def clear_manual_file() -> bool:
    """Удалить manual файл (вернуться к автопрогнозу)."""
    if MANUAL_FILE.exists():
        MANUAL_FILE.unlink()
        return True
    return False


def manual_file_exists() -> bool:
    """Проверить существует ли manual файл."""
    return MANUAL_FILE.exists()


def get_exog_source() -> str:
    """Получить источник данных ('manual' или 'forecast')."""
    if MANUAL_FILE.exists():
        return 'manual'
    elif FORECAST_FILE.exists():
        return 'forecast'
    return 'none'


def save_manual_data(df: pd.DataFrame) -> bool:
    """
    Сохранить данные в manual файл.

    Args:
        df: DataFrame с колонками Date, Ki, Ruonia, USD_ABS, Brent, Type

    Returns:
        True если успешно
    """
    try:
        # Ensure required columns
        required_cols = ['Date', 'Ki', 'Ruonia', 'USD_ABS', 'Brent', 'Type']

        # Reset index if Date is index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={'index': 'Date'})

        # Convert Date to string if needed
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

        # Select and order columns
        cols = [c for c in required_cols if c in df.columns]
        df = df[cols]

        # Save
        df.to_csv(MANUAL_FILE, index=False, float_format='%.2f')
        return True
    except Exception as e:
        print(f"Error saving manual data: {e}")
        return False


def update_manual_row(date: str, values: Dict[str, float]) -> bool:
    """
    Обновить одну строку в manual файле.

    Args:
        date: Дата в формате 'YYYY-MM-DD'
        values: Dict с Ki, Ruonia, USD_ABS, Brent

    Returns:
        True если успешно
    """
    # Load or create manual file
    if MANUAL_FILE.exists():
        df = pd.read_csv(MANUAL_FILE)
    elif FORECAST_FILE.exists():
        df = pd.read_csv(FORECAST_FILE)
    else:
        return False

    df['Date'] = pd.to_datetime(df['Date'])
    date_ts = pd.to_datetime(date)

    # Find and update row
    mask = df['Date'] == date_ts
    if mask.any():
        for key, val in values.items():
            if key in df.columns:
                df.loc[mask, key] = val
    else:
        # Add new row
        new_row = {'Date': date_ts, 'Type': 'Manual', **values}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df = df.sort_values('Date')

    # Save
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df.to_csv(MANUAL_FILE, index=False, float_format='%.2f')
    return True
