"""
Загрузчик сезонно-скорректированных (SA) данных
===============================================

Данные из файлов:
- data/sa_fl.csv - вертикальный формат (Код;Товар;Дата;Значение)
- data/sa_hor.csv - горизонтальный формат (колонки = товары)
- data/micro_sprav.csv - веса товаров по компонентам/субкомпонентам
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

# Путь к данным
DATA_DIR = Path(__file__).parent.parent / 'data'


def load_sa_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Загружает SA данные из sa_fl.csv и возвращает pivot таблицу.

    Returns:
        DataFrame с индексом Date и колонками = названия товаров/компонентов
        Значения - индексы MoM (около 100)
    """
    if file_path is None:
        file_path = DATA_DIR / 'sa_fl.csv'

    df = pd.read_csv(file_path, sep=';', decimal=',', encoding='utf-8-sig')

    # Конвертируем типы
    df['Дата'] = pd.to_datetime(df['Дата'], format='%d.%m.%Y')
    df['Значение'] = pd.to_numeric(
        df['Значение'].astype(str).str.replace(',', '.'),
        errors='coerce'
    )

    # Pivot: строки = даты, колонки = товары
    pivot = df.pivot_table(
        index='Дата',
        columns='Товар',
        values='Значение',
        aggfunc='first'
    ).sort_index()

    pivot.index.name = 'Date'

    return pivot


def load_sa_horizontal(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Загружает SA данные из sa_hor.csv (горизонтальный формат).

    Returns:
        DataFrame с индексом = номер строки (месяц), колонки = "Код_Товар"
    """
    if file_path is None:
        file_path = DATA_DIR / 'sa_hor.csv'

    df = pd.read_csv(file_path, sep=';', decimal=',', encoding='utf-8-sig')

    # Конвертируем все колонки в числа
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '.'),
            errors='coerce'
        )

    return df


def load_component_weights(file_path: Optional[str] = None) -> Dict[str, float]:
    """
    Загружает веса 3 основных компонентов из micro_sprav.csv.

    Returns:
        Dict с весами:
        - 'Продовольственные товары': 0.3948
        - 'Непродовольственные товары': 0.3653
        - 'Услуги': 0.2342
    """
    if file_path is None:
        file_path = DATA_DIR / 'micro_sprav.csv'

    df = pd.read_csv(file_path, sep=';', encoding='utf-8-sig')
    df['Weight'] = pd.to_numeric(
        df['Weight'].astype(str).str.replace(',', '.'),
        errors='coerce'
    )

    # Агрегируем по компонентам
    weights = df.groupby('Компонент')['Weight'].sum().to_dict()

    return weights


def load_subcomponent_weights(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Загружает веса субкомпонентов из micro_sprav.csv.

    Returns:
        DataFrame с колонками: Субкомпонент, Компонент, Weight
    """
    if file_path is None:
        file_path = DATA_DIR / 'micro_sprav.csv'

    df = pd.read_csv(file_path, sep=';', encoding='utf-8-sig')
    df['Weight'] = pd.to_numeric(
        df['Weight'].astype(str).str.replace(',', '.'),
        errors='coerce'
    )

    # Агрегируем по субкомпонентам
    weights = df.groupby(['Субкомпонент', 'Компонент'])['Weight'].sum().reset_index()

    return weights


def get_sa_components() -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Загружает SA данные и веса для 3 компонентов.

    Returns:
        (sa_data, weights)
        - sa_data: DataFrame с колонками для компонентов
        - weights: Dict с нормализованными весами
    """
    sa_data = load_sa_data()
    raw_weights = load_component_weights()

    # Компоненты которые нам нужны
    components = [
        'Продовольственные товары',
        'Непродовольственные товары',
        'Услуги'
    ]

    # Проверяем наличие колонок
    missing = [c for c in components if c not in sa_data.columns]
    if missing:
        raise ValueError(f"Missing components in SA data: {missing}")

    # Нормализуем веса (сумма = 1)
    total = sum(raw_weights.get(c, 0) for c in components)
    weights = {c: raw_weights.get(c, 0) / total for c in components}

    return sa_data[components], weights


def get_sa_with_total() -> pd.DataFrame:
    """
    Загружает SA данные включая общий ИПЦ.

    Returns:
        DataFrame с колонками:
        - 'Все товары и услуги' (общий SA ИПЦ)
        - 'Продовольственные товары'
        - 'Непродовольственные товары'
        - 'Услуги'
    """
    sa_data = load_sa_data()

    cols = [
        'Все товары и услуги',
        'Продовольственные товары',
        'Непродовольственные товары',
        'Услуги'
    ]

    available = [c for c in cols if c in sa_data.columns]

    return sa_data[available]


# Константы для быстрого доступа
COMPONENT_WEIGHTS = {
    'Продовольственные товары': 0.3948,
    'Непродовольственные товары': 0.3653,
    'Услуги': 0.2342
}

COMPONENTS = list(COMPONENT_WEIGHTS.keys())


if __name__ == '__main__':
    # Тест загрузки
    print("Loading SA data...")
    sa = load_sa_data()
    print(f"Shape: {sa.shape}")
    print(f"Date range: {sa.index.min()} - {sa.index.max()}")
    print(f"Columns: {list(sa.columns[:10])}...")
    print()

    print("Component weights:")
    weights = load_component_weights()
    for comp, w in weights.items():
        print(f"  {comp}: {w*100:.2f}%")
    print(f"  Total: {sum(weights.values())*100:.2f}%")
