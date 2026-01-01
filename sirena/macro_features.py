"""
Макроэкономические признаки для моделей прогнозирования.
=========================================================

Признаки на основе Ki (ключевая ставка) и Ruonia.
Найдены через корреляционный анализ с инфляцией.

Лучшие признаки:
- ΔRuonia_lag1 (r=0.477) — изменение RUONIA за месяц
- Spread_lag4 (r=0.444) — спред Ki-Ruonia
- ΔKi_lag6 (r=0.300) — изменение ставки за 6 мес
- Ki_vol (r=0.288) — волатильность ставки
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def add_macro_features(
    df: pd.DataFrame,
    ki_col: str = 'Ki',
    ruonia_col: str = 'Ruonia',
    features: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Добавить макроэкономические признаки в DataFrame.

    Args:
        df: DataFrame с колонками Ki и Ruonia
        ki_col: Название колонки ключевой ставки
        ruonia_col: Название колонки RUONIA
        features: Список признаков для добавления (None = все)

    Returns:
        DataFrame с новыми признаками
    """
    df = df.copy()

    # Проверка наличия колонок
    if ki_col not in df.columns:
        raise ValueError(f"Колонка {ki_col} не найдена в данных")
    if ruonia_col not in df.columns:
        raise ValueError(f"Колонка {ruonia_col} не найдена в данных")

    # Все доступные признаки
    all_features = [
        # Изменения RUONIA (самые сильные предикторы)
        'ruonia_diff_lag1',
        'ruonia_diff_lag2',

        # Спред Ki-Ruonia (индикатор ликвидности)
        'spread_lag3',
        'spread_lag4',

        # Изменения Ki (долгосрочный эффект)
        'ki_diff_lag6',
        'ki_diff3_lag3',

        # Уровни с лагами
        'ki_lag3',
        'ruonia_lag3',

        # Волатильность и отклонения
        'ki_vol',
        'ki_deviation',

        # Нормализованные
        'ki_norm',
        'ruonia_norm',
    ]

    if features is None:
        features = all_features

    # === ИЗМЕНЕНИЯ RUONIA ===
    if 'ruonia_diff_lag1' in features:
        df['ruonia_diff'] = df[ruonia_col].diff()
        df['ruonia_diff_lag1'] = df['ruonia_diff'].shift(1)

    if 'ruonia_diff_lag2' in features:
        if 'ruonia_diff' not in df.columns:
            df['ruonia_diff'] = df[ruonia_col].diff()
        df['ruonia_diff_lag2'] = df['ruonia_diff'].shift(2)

    # === СПРЕД Ki - Ruonia ===
    if 'spread_lag3' in features or 'spread_lag4' in features:
        df['spread'] = df[ki_col] - df[ruonia_col]

    if 'spread_lag3' in features:
        df['spread_lag3'] = df['spread'].shift(3)

    if 'spread_lag4' in features:
        df['spread_lag4'] = df['spread'].shift(4)

    # === ИЗМЕНЕНИЯ Ki ===
    if 'ki_diff_lag6' in features:
        df['ki_diff'] = df[ki_col].diff()
        df['ki_diff_lag6'] = df['ki_diff'].shift(6)

    if 'ki_diff3_lag3' in features:
        df['ki_diff3'] = df[ki_col].diff(3)  # изменение за 3 месяца
        df['ki_diff3_lag3'] = df['ki_diff3'].shift(3)

    # === УРОВНИ С ЛАГАМИ ===
    if 'ki_lag3' in features:
        df['ki_lag3'] = df[ki_col].shift(3)

    if 'ruonia_lag3' in features:
        df['ruonia_lag3'] = df[ruonia_col].shift(3)

    # === ВОЛАТИЛЬНОСТЬ И ОТКЛОНЕНИЯ ===
    if 'ki_vol' in features:
        df['ki_vol'] = df[ki_col].rolling(6).std().shift(1)

    if 'ki_deviation' in features:
        ki_trend = df[ki_col].rolling(12).mean()
        df['ki_deviation'] = (df[ki_col] - ki_trend).shift(3)

    # === НОРМАЛИЗОВАННЫЕ (для Ridge) ===
    if 'ki_norm' in features:
        # Нормализация: (x - mean) / std за последние 24 месяца
        ki_mean = df[ki_col].rolling(24, min_periods=12).mean()
        ki_std = df[ki_col].rolling(24, min_periods=12).std()
        df['ki_norm'] = ((df[ki_col] - ki_mean) / ki_std).shift(1)

    if 'ruonia_norm' in features:
        ruonia_mean = df[ruonia_col].rolling(24, min_periods=12).mean()
        ruonia_std = df[ruonia_col].rolling(24, min_periods=12).std()
        df['ruonia_norm'] = ((df[ruonia_col] - ruonia_mean) / ruonia_std).shift(1)

    # Удаляем промежуточные колонки
    for col in ['ruonia_diff', 'spread', 'ki_diff', 'ki_diff3']:
        if col in df.columns and col not in features:
            df = df.drop(columns=[col])

    return df


def get_best_macro_features() -> List[str]:
    """
    Вернуть список лучших признаков по корреляции.

    Returns:
        Список названий признаков
    """
    return [
        'ruonia_diff_lag1',  # r=0.477
        'spread_lag4',       # r=0.444
        'ki_diff_lag6',      # r=0.300
        'ki_vol',            # r=0.288
        'ki_deviation',      # r=0.256
    ]


def get_minimal_macro_features() -> List[str]:
    """
    Минимальный набор признаков (топ-3).
    """
    return [
        'ruonia_diff_lag1',
        'spread_lag4',
        'ki_diff_lag6',
    ]


# Для совместимости
MACRO_FEATURES_BEST = get_best_macro_features()
MACRO_FEATURES_MINIMAL = get_minimal_macro_features()


# ============================================================================
# НЕФТЬ BRENT
# ============================================================================

def load_brent_prices(
    start_date: str = '2010-01-01',
    end_date: Optional[str] = None,
    cache_file: str = 'data/brent_prices.csv'
) -> pd.DataFrame:
    """
    Загружает месячные цены нефти Brent.

    Сначала пробует загрузить из кеша, затем из Yahoo Finance.

    Args:
        start_date: Начальная дата
        end_date: Конечная дата (None = сегодня)
        cache_file: Путь к файлу кеша

    Returns:
        DataFrame с колонками: Date (index), brent (цена), brent_pct (% изменение)
    """
    from pathlib import Path

    cache_path = Path(cache_file)

    # Пробуем загрузить из кеша
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path, parse_dates=['Date'], index_col='Date')
            # Проверяем актуальность (не старше 7 дней)
            if len(df) > 0:
                last_date = df.index.max()
                if (pd.Timestamp.now() - last_date).days < 7:
                    return df
        except Exception:
            pass

    # Загружаем из Yahoo Finance
    try:
        import yfinance as yf

        ticker = 'BZ=F'  # Brent Crude Oil Futures
        brent = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval='1mo',
            progress=False
        )

        if len(brent) == 0:
            raise ValueError("Нет данных по Brent")

        # Берём Close цену, ресемплируем на месяц
        df = pd.DataFrame()
        df['brent'] = brent['Close'].resample('MS').last()
        df['brent_pct'] = df['brent'].pct_change() * 100  # % изменение
        df.index.name = 'Date'

        # Сохраняем в кеш
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)

        return df

    except ImportError:
        raise ImportError("Установите yfinance: pip install yfinance")
    except Exception as e:
        raise ValueError(f"Ошибка загрузки Brent: {e}")


def add_brent_features(
    df: pd.DataFrame,
    brent_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Добавляет признаки на основе цен нефти Brent.

    Признаки:
    - brent_lag3: цена 3 месяца назад
    - brent_lag6: цена 6 месяцев назад
    - brent_pct_lag3: % изменение цены с лагом 3

    Args:
        df: DataFrame с инфляцией (индекс = Date)
        brent_df: DataFrame с ценами Brent (если None, загружается)

    Returns:
        DataFrame с новыми признаками
    """
    df = df.copy()

    # Загружаем Brent если не передан
    if brent_df is None:
        try:
            brent_df = load_brent_prices()
        except Exception as e:
            # Если не удалось загрузить, возвращаем без изменений
            print(f"Не удалось загрузить данные Brent: {e}")
            return df

    # Объединяем по индексу
    df = df.join(brent_df[['brent', 'brent_pct']], how='left')

    # Создаём лаговые признаки
    df['brent_lag3'] = df['brent'].shift(3)
    df['brent_lag6'] = df['brent'].shift(6)
    df['brent_pct_lag3'] = df['brent_pct'].shift(3)
    df['brent_pct_lag6'] = df['brent_pct'].shift(6)

    # Удаляем исходные колонки (оставляем только лаги)
    df = df.drop(columns=['brent', 'brent_pct'], errors='ignore')

    return df


def get_brent_features() -> List[str]:
    """Список признаков Brent."""
    return [
        'brent_lag3',
        'brent_lag6',
        'brent_pct_lag3',
        'brent_pct_lag6',
    ]


BRENT_FEATURES = get_brent_features()


# =============================================================================
# Production Proxy Features (Torg, pp) — demand indicators
# =============================================================================

def load_production_proxies(data_dir: str = 'data/raw') -> pd.DataFrame:
    """
    Загрузка производственных индикаторов из infostat.csv.

    Колонки:
    - Torg: Индекс торгового оборота (demand proxy, % к пред. месяцу)
    - pp: Индекс платных услуг (services demand proxy, % к пред. месяцу)

    Args:
        data_dir: Путь к директории с данными

    Returns:
        DataFrame с Date index и колонками Torg, pp
    """
    from pathlib import Path

    file_path = Path(data_dir) / 'infostat.csv'
    if not file_path.exists():
        raise FileNotFoundError(f"Файл {file_path} не найден")

    df = pd.read_csv(file_path, sep=';', decimal=',', encoding='utf-8-sig')

    # Конвертация типов
    for col in ['Torg', 'pp']:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

    # Парсинг даты
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    df = df.set_index('Date').sort_index()

    # Нормализация индекса (месяц)
    df.index = df.index.to_period('M').to_timestamp()

    return df


def add_production_features(df: pd.DataFrame, data_dir: str = 'data/raw') -> pd.DataFrame:
    """
    Добавить production proxy features в DataFrame.

    Признаки:
    - torg_lag3: Торговля с лагом 3 мес (demand indicator)
    - torg_lag6: Торговля с лагом 6 мес
    - torg_diff_lag3: Изменение торговли с лагом 3
    - pp_lag3: Услуги с лагом 3 мес
    - pp_lag6: Услуги с лагом 6 мес
    - pp_diff_lag3: Изменение услуг с лагом 3
    - torg_ma3: MA(3) торговли

    Args:
        df: DataFrame с данными (должен иметь DatetimeIndex)
        data_dir: Путь к директории с данными

    Returns:
        DataFrame с новыми признаками
    """
    try:
        production = load_production_proxies(data_dir)
    except FileNotFoundError:
        return df  # Если файл не найден, возвращаем без изменений

    df = df.copy()

    # Нормализуем индекс df
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    # Приводим к месячному индексу для объединения
    df_monthly_idx = df.index.to_period('M').to_timestamp()

    # Создаём временный DataFrame для merge
    production_aligned = production.reindex(df_monthly_idx)

    # Лаговые признаки
    if 'Torg' in production_aligned.columns:
        torg = production_aligned['Torg']
        df['torg_lag3'] = torg.shift(3).values
        df['torg_lag6'] = torg.shift(6).values
        df['torg_diff_lag3'] = torg.diff().shift(3).values
        df['torg_ma3'] = torg.rolling(3).mean().shift(1).values

    if 'pp' in production_aligned.columns:
        pp = production_aligned['pp']
        df['pp_lag3'] = pp.shift(3).values
        df['pp_lag6'] = pp.shift(6).values
        df['pp_diff_lag3'] = pp.diff().shift(3).values

    return df


def get_production_features() -> List[str]:
    """Список production proxy признаков."""
    return [
        'torg_lag3',
        'torg_lag6',
        'torg_diff_lag3',
        'torg_ma3',
        'pp_lag3',
        'pp_lag6',
        'pp_diff_lag3',
    ]


PRODUCTION_FEATURES = get_production_features()
