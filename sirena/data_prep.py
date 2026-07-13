"""
Подготовка данных для BVAR Rate модели
======================================

Объединяет 47 субкомпонентов CPI с Ki/Ruonia для сценарного анализа.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_subcomponents(filepath: str = None) -> pd.DataFrame:
    """
    Загрузка MoM по 47 субкомпонентам.

    Args:
        filepath: Путь к sub_mom.csv

    Returns:
        DataFrame с индексом Date и колонками - кодами субкомпонентов
    """
    if filepath is None:
        filepath = Path(__file__).parent.parent / 'data' / 'raw' / 'sub_mom.csv'

    df = pd.read_csv(filepath, sep=';', decimal=',', encoding='utf-8-sig')
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
    df = df.set_index('Date')

    # Преобразуем все колонки в float
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_macro_data(filepath: str = None) -> pd.DataFrame:
    """
    Загрузка макро-данных (Ki, Ruonia, USD).

    Args:
        filepath: Путь к inflation_data.csv

    Returns:
        DataFrame с Ki, Ruonia, USD и производными
    """
    if filepath is None:
        filepath = Path(__file__).parent.parent / 'data' / 'inflation_data.csv'

    df = pd.read_csv(filepath, sep=';', decimal=',', encoding='utf-8-sig')
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
    df = df.set_index('Date')

    # Выбираем нужные колонки
    macro_cols = ['Ki_i', 'Ruonia', 'usd_nom_i', 'mom']
    result = df[[c for c in macro_cols if c in df.columns]].copy()

    # Добавляем производные
    if 'Ki_i' in result.columns:
        result['Ki_pct'] = (result['Ki_i'] - 100)  # Изменение Ki в п.п.
        result['Ki_diff'] = result['Ki_i'].diff()  # Месячное изменение

    if 'Ruonia' in result.columns:
        result['Ruonia_diff'] = result['Ruonia'].diff()

    if 'usd_nom_i' in result.columns:
        result['USD_pct'] = (result['usd_nom_i'] - 100)  # Изменение USD в %

    return result


def prepare_bvar_data(
    sub_mom_path: str = None,
    macro_path: str = None,
    min_date: str = '2013-01-01'
) -> pd.DataFrame:
    """
    Подготовка объединённого датасета для BVAR Rate.

    Args:
        sub_mom_path: Путь к sub_mom.csv
        macro_path: Путь к inflation_data.csv
        min_date: Минимальная дата (для избежания NaN в начале)

    Returns:
        DataFrame с субкомпонентами + Ki + Ruonia
    """
    # Загрузка данных
    sub_df = load_subcomponents(sub_mom_path)
    macro_df = load_macro_data(macro_path)

    # Объединение по дате
    combined = sub_df.join(macro_df, how='inner')

    # Фильтрация по дате
    combined = combined.loc[combined.index >= min_date]

    # Удаление строк с слишком большим количеством NaN
    threshold = len(combined.columns) * 0.3  # допускаем до 30% пропусков
    combined = combined.dropna(thresh=int(len(combined.columns) - threshold))

    return combined


def get_subcomponent_names() -> dict:
    """
    Справочник названий субкомпонентов по кодам.

    Returns:
        dict: {код: название}
    """
    # Основные группы из справочника Росстата
    names = {
        '11': 'Хлеб и хлебобулочные изделия',
        '12': 'Крупа и бобовые',
        '13': 'Макаронные изделия',
        '14': 'Мясо и птица',
        '15': 'Колбасные изделия',
        '16': 'Рыба и морепродукты',
        '17': 'Молоко и молочная продукция',
        '18': 'Масло и жиры',
        '19': 'Яйца',
        '20': 'Сахар',
        '21': 'Кондитерские изделия',
        '22': 'Чай, кофе, какао',
        '23': 'Плодоовощная продукция',
        '24': 'Алкогольные напитки',
        '25': 'Безалкогольные напитки',
        '26': 'Общественное питание',
        '27': 'Консервы',
        '28': 'Соль и специи',
        '29': 'Готовые блюда',
        '30': 'Детское питание',
        '31': 'Корма для животных',
        '32': 'Прочие продовольственные товары',
        '33': 'Одежда',
        '34': 'Обувь',
        '35': 'Ткани',
        '36': 'Галантерея',
        '37': 'Моющие средства',
        '38': 'Парфюмерия и косметика',
        '39': 'Табачные изделия',
        '40': 'Строительные материалы',
        '41': 'Автомобили',
        '42': 'ГСМ (бензин, дизель)',
        '43': 'Мебель',
        '44': 'Бытовая техника',
        '46': 'Медикаменты',
        '47': 'Медицинские товары',
        '48': 'Товары для отдыха',
        '49': 'Книги и печатные издания',
        '50': 'Электроника',
        '51': 'Услуги связи',
        '52': 'Услуги ЖКХ',
        '53': 'Транспортные услуги',
        '54': 'Бытовые услуги',
        '55': 'Образование',
        '67': 'Туризм и гостиницы',
    }
    return names


def get_rate_sensitivity() -> dict:
    """
    Чувствительность субкомпонентов к ключевой ставке (гипотеза).

    Returns:
        dict: {код: {sensitivity, lag, channel}}
    """
    sensitivity = {
        # Высокая чувствительность (кредитозависимые)
        '41': {'sensitivity': -0.20, 'lag': 4, 'channel': 'credit'},  # Автомобили
        '43': {'sensitivity': -0.15, 'lag': 3, 'channel': 'credit'},  # Мебель
        '44': {'sensitivity': -0.12, 'lag': 3, 'channel': 'credit'},  # Бытовая техника
        '50': {'sensitivity': -0.10, 'lag': 3, 'channel': 'credit'},  # Электроника
        '40': {'sensitivity': -0.10, 'lag': 4, 'channel': 'credit'},  # Стройматериалы

        # Средняя чувствительность (импортозависимые)
        '33': {'sensitivity': -0.10, 'lag': 3, 'channel': 'fx'},  # Одежда
        '34': {'sensitivity': -0.08, 'lag': 3, 'channel': 'fx'},  # Обувь
        '46': {'sensitivity': -0.08, 'lag': 2, 'channel': 'fx'},  # Медикаменты
        '47': {'sensitivity': -0.08, 'lag': 2, 'channel': 'fx'},  # Медицинские товары
        '38': {'sensitivity': -0.06, 'lag': 2, 'channel': 'fx'},  # Парфюмерия

        # Низкая чувствительность (базовые товары)
        '11': {'sensitivity': -0.02, 'lag': 6, 'channel': 'basic'},  # Хлеб
        '12': {'sensitivity': -0.02, 'lag': 6, 'channel': 'basic'},  # Крупы
        '14': {'sensitivity': -0.03, 'lag': 6, 'channel': 'basic'},  # Мясо
        '17': {'sensitivity': -0.03, 'lag': 6, 'channel': 'basic'},  # Молоко
        '19': {'sensitivity': -0.02, 'lag': 6, 'channel': 'basic'},  # Яйца
        '20': {'sensitivity': -0.02, 'lag': 6, 'channel': 'basic'},  # Сахар
        '23': {'sensitivity': -0.01, 'lag': 6, 'channel': 'basic'},  # Овощи/фрукты (сезонность)

        # Услуги
        '52': {'sensitivity': -0.01, 'lag': 12, 'channel': 'regulated'},  # ЖКХ (регулируемые)
        '53': {'sensitivity': -0.05, 'lag': 6, 'channel': 'demand'},  # Транспорт
        '55': {'sensitivity': -0.03, 'lag': 12, 'channel': 'demand'},  # Образование
        '67': {'sensitivity': -0.08, 'lag': 6, 'channel': 'demand'},  # Туризм
        '54': {'sensitivity': -0.05, 'lag': 4, 'channel': 'demand'},  # Бытовые услуги
    }
    return sensitivity


def get_subcomponent_groups() -> dict:
    """
    Группировка субкомпонентов по чувствительности к ставке.

    Returns:
        dict: {группа: [коды]}
    """
    groups = {
        'credit_dependent': ['41', '43', '44', '50', '40'],  # Кредитозависимые
        'import_dependent': ['33', '34', '46', '47', '38'],  # Импортозависимые
        'basic_food': ['11', '12', '14', '17', '19', '20', '23'],  # Базовые продукты
        'services': ['52', '53', '55', '67', '54'],  # Услуги
        'volatile': ['42', '23'],  # Волатильные (ГСМ, овощи/фрукты)
    }
    return groups


if __name__ == '__main__':
    # Тест загрузки данных
    print("=== Тест подготовки данных для BVAR Rate ===\n")

    # Загрузка субкомпонентов
    sub_df = load_subcomponents()
    print(f"Субкомпоненты: {sub_df.shape[0]} месяцев, {sub_df.shape[1]} колонок")
    print(f"Период: {sub_df.index.min()} — {sub_df.index.max()}")
    print(f"Колонки: {list(sub_df.columns[:10])}...")

    # Загрузка макро
    macro_df = load_macro_data()
    print(f"\nМакро-данные: {macro_df.shape[0]} месяцев, {macro_df.shape[1]} колонок")
    print(f"Колонки: {list(macro_df.columns)}")

    # Объединённый датасет
    combined = prepare_bvar_data()
    print(f"\nОбъединённый датасет: {combined.shape[0]} месяцев, {combined.shape[1]} колонок")
    print(f"Период: {combined.index.min()} — {combined.index.max()}")

    # Проверка пропусков
    missing = combined.isnull().sum()
    if missing.sum() > 0:
        print(f"\nПропуски: {missing[missing > 0].to_dict()}")
    else:
        print("\nПропусков нет!")

    # Статистика по Ki и Ruonia
    if 'Ki_i' in combined.columns:
        print(f"\nKi: min={combined['Ki_i'].min():.1f}, max={combined['Ki_i'].max():.1f}")
    if 'Ruonia' in combined.columns:
        print(f"Ruonia: min={combined['Ruonia'].min():.2f}, max={combined['Ruonia'].max():.2f}")
