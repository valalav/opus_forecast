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
        """Load the latest canonical unadjusted monthly indices; fail on gaps."""
        self._monthly_data = load_model_data('raw', data_dir=self.data_dir)
        return self._monthly_data

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


# Shared input contract for live forecasts, backtests and public loaders.
MONTHLY_COLUMNS = {
    'mom': 'Все товары и услуги',
    'Prod': 'Продовольственные товары',
    'Nonprod': 'Непродовольственные товары',
    'Serv': 'Услуги',
}
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / 'data'


class DataFreshnessError(ValueError):
    """A required observation/representation is unavailable at the forecast origin."""


def require_observations(frame, columns, through, source, trailing_months=1):
    """Reject stale/partial/non-finite data, including gaps inside a lag window."""
    through = pd.Timestamp(through).to_period('M').to_timestamp()
    dates = pd.date_range(end=through, periods=trailing_months, freq='MS')
    missing = []
    for col in columns:
        values = (pd.to_numeric(frame[col], errors='coerce').reindex(dates)
                  if col in frame else pd.Series(index=dates, dtype=float))
        bad = values.index[~np.isfinite(values)]
        if len(bad):
            last = frame[col].last_valid_index() if col in frame else None
            missing.append(f'{col}: missing {bad[0]:%Y-%m}, last valid {last}')
    if missing:
        raise DataFreshnessError(f'{source}: required through {through:%Y-%m}; ' + '; '.join(missing[:5]) + (f'; and {len(missing)-5} more series' if len(missing)>5 else ''))


def load_model_data(representation, data_dir=None, cutoff=None, include_macro=False):
    """Return a homogeneous raw or SA vintage, never splice representations.

    Cutoff is an observation-month boundary. Current revised files are NOT historical
    publication vintages; backtests using them remain pseudo-out-of-sample.
    """
    import hashlib
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    if representation not in ('raw', 'sa'):
        raise ValueError('representation must be explicitly raw or sa')
    raw_path = data_dir / 'inflation_data.csv'
    raw = pd.read_csv(raw_path, sep=';', decimal=',', encoding='utf-8-sig')
    raw['Date'] = pd.to_datetime(raw['Date'], format='%d.%m.%Y', errors='raise').dt.to_period('M').dt.to_timestamp()
    raw = raw.set_index('Date').sort_index()
    if raw.index.has_duplicates:
        raise DataFreshnessError(f'{raw_path}: duplicate observation months')
    for col in raw:
        raw[col] = pd.to_numeric(raw[col].astype(str).str.replace(',', '.', regex=False), errors='coerce')
    raw = raw.loc[raw[list(MONTHLY_COLUMNS)].notna().any(axis=1)]
    if cutoff is not None:
        cutoff = pd.Timestamp(cutoff).to_period('M').to_timestamp()
        raw = raw.loc[:cutoff]
    if raw.empty:
        raise DataFreshnessError(f'{raw_path}: no monthly observations')
    required = cutoff if cutoff is not None else raw.index.max()
    require_observations(raw, MONTHLY_COLUMNS, required, raw_path)
    path = raw_path
    if representation == 'raw':
        frame = raw[list(MONTHLY_COLUMNS)].rename(columns=MONTHLY_COLUMNS)
    else:
        path = data_dir / 'sa_fl.csv'
        sa = pd.read_csv(path, sep=';', decimal=',', encoding='utf-8-sig')
        sa['Дата'] = pd.to_datetime(sa['Дата'], format='%d.%m.%Y', errors='raise').dt.to_period('M').dt.to_timestamp()
        sa['Значение'] = pd.to_numeric(sa['Значение'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
        if sa.duplicated(['Дата', 'Товар']).any():
            raise DataFreshnessError(f'{path}: duplicate series/month')
        frame = sa.pivot(index='Дата', columns='Товар', values='Значение').sort_index()
        frame = frame.loc[:required, list(MONTHLY_COLUMNS.values())]
    require_observations(frame, MONTHLY_COLUMNS.values(), required, path,
                         trailing_months=min(12, len(frame)))
    if include_macro:
        for col in ('usd_nom_i', 'Ki', 'Ruonia', 'Ki_i'):
            require_observations(raw, [col], required, raw_path,
                                 trailing_months=min(12, len(raw)))
            frame[col] = raw[col]
    frame.index.name = 'Date'
    frame.attrs['input_contract'] = {
        'representation': representation, 'units': 'mom_index_100',
        'source': str(path), 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'last_observation': frame.index.max().strftime('%Y-%m-%d'),
        'required_through': required.strftime('%Y-%m-%d'),
        'macro_included': include_macro,
    }
    return frame


def forecast_source_manifest(data_dir=None):
    """Content fingerprints invalidate a forecast after any registered input changes."""
    import hashlib
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    names = ['inflation_data.csv', 'sa_fl.csv', 'kbr_weekly_prices_2008_2026.csv',
             'kbr_micro_full.csv', 'micro_sprav.csv', 'raw/infostat.csv',
             'external/micro_cpi_region_export/micro_test_statsmodels.csv',
             'nowcast_policy.json', 'Сравнение еженедельных цен_01.csv',
             'weekly_accounting_month_overrides.csv', 'sa_hor.csv', 'mom_sa_kbr.csv',
             'raw/subcomp.csv', 'raw/sub_mom.csv', 'sa_source_manifest.json',
             'access_source_manifest.json', 'raw/infostat_source_manifest.json',
             'kbr_indices.csv', 'access_weights.csv', 'items_structure.csv']
    return {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
            if (data_dir / name).exists() else None for name in names}


def validate_forecast_cache(payload, data_dir=None):
    expected = forecast_source_manifest(data_dir)
    actual = payload.get('source_manifest', {})
    changed = [name for name in expected if name not in actual or actual[name] != expected[name]]
    if changed:
        raise DataFreshnessError('Forecast cache is stale; rerun scripts/precompute_forecasts.py. Changed/missing inputs: ' + ', '.join(changed))


FORECAST_CACHE_ALIASES = {
    'NGBoost_Shock': 'NGBoostShock', 'Ridge_Shock': 'RidgeShockDummies',
    'Ridge_Ext': 'RidgeExtended',
}


def cached_forecast_point(payload, model_name, target_date, cutoff):
    """Select the actual dated horizon; never refit or relabel a one-step value."""
    cutoff = pd.Timestamp(cutoff).to_period('M').to_timestamp()
    if pd.Timestamp(payload['last_data_date']).to_period('M').to_timestamp() != cutoff:
        raise DataFreshnessError('Forecast cache cutoff differs from dashboard observations')
    dates = pd.DatetimeIndex(pd.to_datetime(payload['forecast_dates']))
    if dates.has_duplicates:
        raise DataFreshnessError('Duplicate forecast target dates')
    target = pd.Timestamp(target_date).to_period('M').to_timestamp()
    positions = (dates == target).nonzero()[0]
    if len(positions) != 1 or target <= cutoff:
        raise DataFreshnessError(f'No forecast for requested target {target:%Y-%m}')
    name = FORECAST_CACHE_ALIASES.get(model_name, model_name)
    values = payload['forecasts'].get(name)
    step = int(positions[0])
    if values is None or len(values) <= step or values[step] is None:
        return float('nan')
    value = float(values[step])
    if not np.isfinite(value):
        raise DataFreshnessError(f'Non-finite cached forecast: {name}')
    return value


def load_common_backtest(horizon, report_dir=None, data_dir=None):
    """Held-out, matched-origin scores for the explicitly selected Micro policy.

    A revised source invalidates this comparison; legacy files remain separate.
    """
    import json
    report_dir = Path(report_dir) if report_dir is not None else DEFAULT_DATA_DIR.parent / 'archive/results/micro_policy_20260909'
    summary_path = report_dir / 'common_summary.json'
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    if horizon not in summary['horizons'] or summary['source_manifest'] != forecast_source_manifest(data_dir):
        return None
    frame = pd.read_csv(report_dir / 'common_predictions.csv', parse_dates=['Date', 'cutoff'])
    frame = frame.loc[(frame.horizon == horizon) & (frame.Date >= pd.Timestamp(summary['validation_start']))].copy()
    selected = summary['selected_on_development_h1']
    candidates = summary['models']
    valid = np.isfinite(frame[['Actual', *candidates]].to_numpy(dtype=float)).all(axis=1)
    frame = frame.loc[valid, ['Date', 'Actual', 'Ridge', 'Huber', 'Ridge_ProdProxy', selected]].rename(columns={selected: 'Micro'})
    frame.attrs['evaluation'] = 'Общие целевые месяцы: август 2025 — июль 2026. Политика Micro выбрана на предшествующих 12 месяцах. Последовательная проверка на пересмотренных данных; публикационные vintages не восстановлены.'
    frame.attrs['common_origins'] = True
    return frame.reset_index(drop=True)
