#!/usr/bin/env python3
"""
Извлечение пропущенных данных из ОПР_статистика:
1. Бюджеты КБР
2. Жильё (цены, ипотека)
3. Инфляционные ожидания
4. Опережающий индикатор ВРП
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path("edge_lab/assets/charts/ОПР_статистика")
OUTPUT_PATH = Path("edge_lab/data")
OUTPUT_PATH.mkdir(exist_ok=True)

# КБР коды для поиска
KBR_PATTERNS = [
    'Кабардино-Балкарская',
    'Кабардино-Балкарск',
    'КБР',
    '81 -',  # Код региона
    '81-',
    '81 '
]

SKFO_PATTERNS = [
    'Северо-Кавказский',
    'СКФО',
    'Северо-Кавказ'
]


def find_region_row(df, patterns):
    """Найти строку с регионом по паттернам"""
    for idx, row in df.iterrows():
        row_str = ' '.join([str(x) for x in row.values])
        for pattern in patterns:
            if pattern.lower() in row_str.lower():
                return idx
    return None


def extract_budget_kbr():
    """Извлечь бюджетные данные КБР"""
    print("\n=== БЮДЖЕТЫ КБР ===")
    budget_file = BASE_PATH / "Бюджеты" / "Консолидированные бюджеты субъектов РФ.xlsx"

    if not budget_file.exists():
        print(f"Файл не найден: {budget_file}")
        return None

    xl = pd.ExcelFile(budget_file)
    print(f"Листы: {xl.sheet_names}")

    all_data = []

    for sheet in xl.sheet_names:
        if sheet == 'Контакты':
            continue

        print(f"\nОбработка листа: {sheet}")
        df = pd.read_excel(budget_file, sheet_name=sheet, header=None)

        # Найти КБР
        kbr_row = find_region_row(df, KBR_PATTERNS)
        skfo_row = find_region_row(df, SKFO_PATTERNS)
        rf_row = find_region_row(df, ['Российская Федерация', '0 -'])

        print(f"  RF row: {rf_row}, SKFO row: {skfo_row}, KBR row: {kbr_row}")

        if kbr_row is None:
            print(f"  КБР не найден в листе {sheet}")
            continue

        # Найти строку с датами (обычно строка 1)
        date_row = 1
        dates = df.iloc[date_row, 2:].values

        # Извлечь данные КБР
        kbr_values = df.iloc[kbr_row, 2:].values

        # Создать DataFrame
        for i, (date, value) in enumerate(zip(dates, kbr_values)):
            if pd.notna(date) and pd.notna(value):
                all_data.append({
                    'date': pd.to_datetime(date),
                    'indicator': sheet,
                    'region': 'KBR',
                    'value': value
                })

        # Также извлечь СКФО и РФ для сравнения
        if skfo_row is not None:
            skfo_values = df.iloc[skfo_row, 2:].values
            for i, (date, value) in enumerate(zip(dates, skfo_values)):
                if pd.notna(date) and pd.notna(value):
                    all_data.append({
                        'date': pd.to_datetime(date),
                        'indicator': sheet,
                        'region': 'SKFO',
                        'value': value
                    })

        if rf_row is not None:
            rf_values = df.iloc[rf_row, 2:].values
            for i, (date, value) in enumerate(zip(dates, rf_values)):
                if pd.notna(date) and pd.notna(value):
                    all_data.append({
                        'date': pd.to_datetime(date),
                        'indicator': sheet,
                        'region': 'RF',
                        'value': value
                    })

    if all_data:
        result = pd.DataFrame(all_data)
        result = result.sort_values(['indicator', 'region', 'date'])

        # Pivot для удобства
        pivot = result.pivot_table(
            index='date',
            columns=['indicator', 'region'],
            values='value'
        )
        pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
        pivot = pivot.reset_index()

        output_file = OUTPUT_PATH / "kbr_budget_data.csv"
        pivot.to_csv(output_file, index=False)
        print(f"\n✅ Сохранено: {output_file}")
        print(f"   Строк: {len(pivot)}, Колонок: {len(pivot.columns)}")
        print(f"   Период: {pivot['date'].min()} — {pivot['date'].max()}")

        return pivot

    return None


def extract_housing_prices():
    """Извлечь данные по ценам на жильё"""
    print("\n=== ЦЕНЫ НА ЖИЛЬЁ (Домклик) ===")

    housing_files = [
        ("Цены на жилье (Домклик, факт.сделки).xlsx", "housing_prices_deals"),
        ("Цены на жилье (Домклик, обьявления).xlsx", "housing_prices_listings"),
        ("Льготная ипотека.xlsx", "mortgage_subsidized"),
        ("Проектное финансирование.xlsx", "project_financing"),
        ("Строительство МКД.xlsx", "construction_mkd"),
    ]

    all_results = {}

    for filename, output_name in housing_files:
        filepath = BASE_PATH / "жилье" / filename
        if not filepath.exists():
            print(f"❌ Не найден: {filename}")
            continue

        print(f"\nОбработка: {filename}")
        xl = pd.ExcelFile(filepath)
        print(f"  Листы: {xl.sheet_names[:5]}...")

        # Ищем лист с регионами
        for sheet in ['регионы', 'Регионы', 'Свод (ФО)', 'данные', 'Данные']:
            if sheet in xl.sheet_names:
                df = pd.read_excel(filepath, sheet_name=sheet, header=None)

                # Поиск КБР
                kbr_row = find_region_row(df, KBR_PATTERNS)
                skfo_row = find_region_row(df, SKFO_PATTERNS)

                if kbr_row is not None:
                    print(f"  ✅ КБР найден в листе '{sheet}', строка {kbr_row}")

                    # Найти строку с датами
                    for date_row in range(5):
                        dates = df.iloc[date_row, 1:].values
                        if any(isinstance(d, (pd.Timestamp, str)) and '20' in str(d) for d in dates[:10] if pd.notna(d)):
                            break

                    # Извлечь данные
                    dates = pd.to_datetime(df.iloc[date_row, 1:].values, errors='coerce')
                    kbr_values = df.iloc[kbr_row, 1:].values

                    data = []
                    for date, value in zip(dates, kbr_values):
                        if pd.notna(date) and pd.notna(value):
                            data.append({'date': date, 'KBR': value})

                    if skfo_row is not None:
                        skfo_values = df.iloc[skfo_row, 1:].values
                        for i, (date, value) in enumerate(zip(dates, skfo_values)):
                            if pd.notna(date) and pd.notna(value) and i < len(data):
                                data[i]['SKFO'] = value

                    if data:
                        result_df = pd.DataFrame(data)
                        output_file = OUTPUT_PATH / f"{output_name}_kbr.csv"
                        result_df.to_csv(output_file, index=False)
                        print(f"  ✅ Сохранено: {output_file}")
                        print(f"     Строк: {len(result_df)}, Период: {result_df['date'].min()} — {result_df['date'].max()}")
                        all_results[output_name] = result_df
                    break
                else:
                    print(f"  ❌ КБР не найден в листе '{sheet}'")

    return all_results


def extract_inflation_expectations():
    """Извлечь инфляционные ожидания"""
    print("\n=== ИНФЛЯЦИОННЫЕ ОЖИДАНИЯ ===")

    ie_files = [
        "Инфляционные ожидания населения (РФ и ФО).xlsx",
        "Инфляционные ожидания и потребительские настроения.xlsx"
    ]

    ie_path = BASE_PATH / "инфляционные ожидания"

    for filename in ie_files:
        filepath = ie_path / filename
        if not filepath.exists():
            print(f"❌ Не найден: {filename}")
            continue

        print(f"\nОбработка: {filename}")
        xl = pd.ExcelFile(filepath)
        print(f"  Листы: {xl.sheet_names}")

        # Читаем первый лист данных
        for sheet in xl.sheet_names:
            if 'описание' in sheet.lower() or 'info' in sheet.lower():
                continue

            df = pd.read_excel(filepath, sheet_name=sheet, header=None, nrows=50)

            # Ищем СКФО (КБР может не быть отдельно)
            skfo_row = find_region_row(df, SKFO_PATTERNS)
            rf_row = find_region_row(df, ['Российская Федерация', 'РФ', 'Россия'])

            if skfo_row is not None or rf_row is not None:
                print(f"  Лист '{sheet}': RF row={rf_row}, SKFO row={skfo_row}")

                # Найти даты
                for date_row in range(5):
                    dates = df.iloc[date_row, 1:].values
                    valid_dates = [d for d in dates if pd.notna(d)]
                    if valid_dates:
                        try:
                            pd.to_datetime(valid_dates[0])
                            break
                        except:
                            continue

                dates = pd.to_datetime(df.iloc[date_row, 1:].values, errors='coerce')

                data = []
                for i, date in enumerate(dates):
                    if pd.notna(date):
                        row_data = {'date': date}
                        if rf_row is not None:
                            row_data['RF'] = df.iloc[rf_row, i+1]
                        if skfo_row is not None:
                            row_data['SKFO'] = df.iloc[skfo_row, i+1]
                        data.append(row_data)

                if data:
                    result_df = pd.DataFrame(data)
                    output_name = filename.replace('.xlsx', '').replace(' ', '_')
                    output_file = OUTPUT_PATH / f"inflation_expectations_{sheet}.csv"
                    result_df.to_csv(output_file, index=False)
                    print(f"  ✅ Сохранено: {output_file}")
                    print(f"     Строк: {len(result_df)}")
                break


def extract_grp_leading_indicator():
    """Извлечь опережающий индикатор ВРП"""
    print("\n=== ОПЕРЕЖАЮЩИЙ ИНДИКАТОР ВРП ===")

    vrp_file = BASE_PATH / "ВРП" / "Опережающий индикатор ВРП (с прогнозом).xlsm"

    if not vrp_file.exists():
        print(f"❌ Файл не найден: {vrp_file}")
        return None

    xl = pd.ExcelFile(vrp_file)
    print(f"Листы: {xl.sheet_names}")

    # Ключевые листы
    key_sheets = ['Данные_прогнозы', 'Данные_Регионы', 'Сводные_декомп']

    all_data = []

    for sheet in key_sheets:
        if sheet not in xl.sheet_names:
            continue

        print(f"\nЛист: {sheet}")
        df = pd.read_excel(vrp_file, sheet_name=sheet, header=None)
        print(f"  Размер: {df.shape}")
        print(f"  Первые строки:\n{df.iloc[:5, :5].to_string()}")

        # Поиск СКФО
        skfo_row = find_region_row(df, ['СКФО', 'Северо-Кавказское'])
        if skfo_row:
            print(f"  СКФО найден в строке {skfo_row}")

    # Лист с прогнозами
    if 'Данные_прогнозы' in xl.sheet_names:
        df = pd.read_excel(vrp_file, sheet_name='Данные_прогнозы', header=None)

        # Структура: колонка 0 = регион, остальные = кварталы
        # Найти строку с датами/кварталами

        output_file = OUTPUT_PATH / "grp_leading_indicator_raw.csv"
        df.to_csv(output_file, index=False, header=False)
        print(f"\n✅ Raw данные сохранены: {output_file}")

    return df


def main():
    print("=" * 60)
    print("ИЗВЛЕЧЕНИЕ ПРОПУЩЕННЫХ ДАННЫХ ИЗ ОПР_СТАТИСТИКА")
    print("=" * 60)

    # 1. Бюджеты
    budget_df = extract_budget_kbr()

    # 2. Жильё
    housing_data = extract_housing_prices()

    # 3. Инфляционные ожидания
    extract_inflation_expectations()

    # 4. ВРП
    grp_df = extract_grp_leading_indicator()

    print("\n" + "=" * 60)
    print("ИТОГО ИЗВЛЕЧЕНО:")
    print("=" * 60)

    # Список созданных файлов
    created_files = list(OUTPUT_PATH.glob("*.csv"))
    new_files = [f for f in created_files if f.stat().st_mtime > pd.Timestamp.now().timestamp() - 60]

    for f in sorted(OUTPUT_PATH.glob("*kbr*.csv")) + sorted(OUTPUT_PATH.glob("*inflation*.csv")) + sorted(OUTPUT_PATH.glob("*grp*.csv")):
        print(f"  ✅ {f.name}")


if __name__ == "__main__":
    main()
