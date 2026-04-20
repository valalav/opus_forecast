"""
Модуль загрузки данных для СИРЕНА-КБР
=====================================
"""

import os
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Загрузчик данных для моделей прогнозирования."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self._monthly_data: Optional[pd.DataFrame] = None
        self._weekly_data: Optional[pd.DataFrame] = None
        self._inflation_data: Optional[pd.DataFrame] = None

    def load_monthly_kbr(self) -> Optional[pd.DataFrame]:
        """
        Загрузка месячных данных ИПЦ КБР из infl_kbr.csv.

        Returns:
            DataFrame с колонками: 'Все товары и услуги', 'Продовольственные товары',
            'Непродовольственные товары', 'Услуги'. Индекс - datetime.
        """
        path = self.data_dir / "infl_kbr.csv"

        if not path.exists():
            logger.error(f"Файл не найден: {path}")
            return None

        try:
            df_raw = pd.read_csv(path, sep=';', decimal='.')

            # Парсинг даты (поддержка разных форматов)
            if 'Day' in df_raw.columns:
                try:
                    df_raw['Date'] = pd.to_datetime(df_raw['Day'], format='%d.%m.%Y')
                except ValueError:
                    df_raw['Date'] = pd.to_datetime(df_raw['Day'], format='%Y-%m-%d', errors='coerce')
                    if df_raw['Date'].isna().all():
                        df_raw['Date'] = pd.to_datetime(df_raw['Day'])

            # Pivot если нужно
            if 'Товар' in df_raw.columns and 'MoM' in df_raw.columns:
                df = df_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
            else:
                df = df_raw.set_index('Date')

            # Оставляем нужные колонки
            required_cols = ['Все товары и услуги', 'Продовольственные товары',
                           'Непродовольственные товары', 'Услуги']
            df = df[required_cols].copy()
            df = df.sort_index()

            self._monthly_data = df
            logger.info(f"Загружено {len(df)} месяцев данных КБР")
            return df

        except Exception as e:
            logger.error(f"Ошибка загрузки infl_kbr.csv: {e}")
            return None

    def load_weekly_prices(self) -> Optional[pd.DataFrame]:
        """
        Загрузка недельных цен из weekly_prices.csv.

        Returns:
            DataFrame с недельными ценами товаров.
        """
        path = self.data_dir / "weekly_prices.csv"

        if not path.exists():
            logger.warning(f"Недельные данные не найдены: {path}")
            return None

        try:
            w = pd.read_csv(path, sep=';', decimal=',')

            if 'Товары' not in w.columns:
                w = pd.read_csv(path, sep=';', decimal='.')

            if 'Сведено' in w.columns:
                w[['year', 'week']] = w['Сведено'].str.split('_', expand=True).astype(int)
                w['month'] = pd.to_datetime(
                    w['year'].astype(str) + w['week'].astype(str) + '1',
                    format='%Y%W%w'
                ).dt.month

            self._weekly_data = w
            logger.info(f"Загружено {len(w)} записей недельных цен")
            return w

        except Exception as e:
            logger.warning(f"Ошибка загрузки weekly_prices.csv: {e}")
            return None

    def load_inflation_data(self) -> Optional[pd.DataFrame]:
        """
        Загрузка расширенных данных инфляции (с макропоказателями).

        Returns:
            DataFrame с ИПЦ, USD, RUONIA и компонентами.
        """
        path = self.data_dir / "inflation_data.csv"

        if not path.exists():
            logger.error(f"Файл не найден: {path}")
            return None

        try:
            df = pd.read_csv(path, sep=';', decimal=',')

            # Исправление типов
            cols_to_fix = ['mom', 'Prod', 'Nonprod', 'Serv', 'usd_nom_i', 'Ruonia', 'Ki', 'Ki_i']
            for col in cols_to_fix:
                if col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str).str.replace(',', '.')
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Парсинг даты
            df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
            if df['Date'].isna().any():
                df['Date'] = pd.to_datetime(df['Date'])

            # Нормализация к началу месяца
            df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
            df = df.set_index('Date').sort_index()

            self._inflation_data = df
            logger.info(f"Загружено {len(df)} месяцев макроданных")
            return df

        except Exception as e:
            logger.error(f"Ошибка загрузки inflation_data.csv: {e}")
            return None

    def load_all(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Загрузка всех данных.

        Returns:
            Tuple (monthly_data, weekly_data)
        """
        monthly = self.load_monthly_kbr()
        weekly = self.load_weekly_prices()
        return monthly, weekly

    @property
    def monthly_data(self) -> Optional[pd.DataFrame]:
        """Месячные данные (lazy load)."""
        if self._monthly_data is None:
            self.load_monthly_kbr()
        return self._monthly_data

    @property
    def weekly_data(self) -> Optional[pd.DataFrame]:
        """Недельные данные (lazy load)."""
        if self._weekly_data is None:
            self.load_weekly_prices()
        return self._weekly_data

    def get_last_date(self) -> Optional[pd.Timestamp]:
        """Последняя дата с фактическими данными."""
        if self.monthly_data is not None:
            valid = self.monthly_data.dropna(subset=['Все товары и услуги'])
            return valid.index.max()
        return None

    def get_date_range(self) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """Диапазон дат в данных."""
        if self.monthly_data is not None:
            return self.monthly_data.index.min(), self.monthly_data.index.max()
        return None, None


# Синглтон загрузчика
_loader: Optional[DataLoader] = None


def get_data_loader() -> DataLoader:
    """Получить глобальный загрузчик данных."""
    global _loader
    if _loader is None:
        _loader = DataLoader()
    return _loader
