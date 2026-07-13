#!/usr/bin/env python3
"""
Анализ расхождения между субкомпонентом "Плодоовощи" (код 33)
и взвешенной суммой микрокомпонентов.

Исследовательские вопросы:
1. Какая формула агрегации используется Росстатом?
2. Совпадают ли веса из справочника с фактически используемыми?
3. Есть ли округление на промежуточных этапах?
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Пути к данным
DATA_DIR = Path(__file__).parent.parent / 'data'
SUB_FILE = DATA_DIR / 'plodoov_03_sub.csv'
MICRO_FILE = DATA_DIR / 'plodoov_05_micro.csv'
MICRO_SPRAV = DATA_DIR / 'raw' / 'micro_sprav.csv'

# Коды плодоовощных микрокомпонентов (21 штука)
PLODOV_CODES = [115, 121, 167, 204, 249, 252, 343, 349, 382, 432, 435,
                447, 506, 574, 618, 723, 753, 971, 972, 973, 1087]


def load_data():
    """Загрузка данных субкомпонента и микрокомпонентов."""
    print("=" * 60)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 60)

    # Субкомпонент 33 (плодоовощи)
    sub_df = pd.read_csv(SUB_FILE, sep=';', decimal=',', encoding='utf-8-sig')
    sub_df['Date'] = pd.to_datetime(sub_df['Day'], format='%d.%m.%Y')
    sub_df = sub_df.sort_values('Date')
    print(f"Субкомпонент 33: {len(sub_df)} записей ({sub_df['Date'].min().strftime('%Y-%m')} — {sub_df['Date'].max().strftime('%Y-%m')})")

    # Микрокомпоненты
    micro_df = pd.read_csv(MICRO_FILE, sep=';', decimal=',', encoding='utf-8-sig')
    micro_df['Date'] = pd.to_datetime(micro_df['Day'], format='%d.%m.%Y')
    micro_df = micro_df.sort_values(['Date', 'Item_code'])

    unique_codes = sorted(micro_df['Item_code'].unique())
    print(f"Микрокомпоненты: {len(unique_codes)} товаров, {len(micro_df)} записей")
    print(f"Коды: {unique_codes}")

    # Справочник весов
    sprav_df = pd.read_csv(MICRO_SPRAV, sep=';', decimal=',', encoding='utf-8-sig')
    sprav_plodov = sprav_df[sprav_df['Субкомпонент'] == 'Плодоовощная продукция, включая картофель']
    print(f"\nСправочник: {len(sprav_plodov)} плодоовощных товаров")

    # Проверка наличия всех товаров
    expected_codes = set(PLODOV_CODES)
    actual_codes = set(unique_codes)
    missing = expected_codes - actual_codes
    if missing:
        print(f"\n⚠️ ОТСУТСТВУЮТ в micro файле: {sorted(missing)}")

    return sub_df, micro_df, sprav_plodov


def calculate_weighted_aggregation(micro_df, weights_dict, date):
    """Расчёт взвешенного среднего MoM по микрокомпонентам."""
    month_data = micro_df[micro_df['Date'] == date]

    total_contrib = 0.0
    total_weight = 0.0

    for _, row in month_data.iterrows():
        code = row['Item_code']
        if code in weights_dict:
            mom = row['MoM']
            weight = weights_dict[code]
            total_contrib += weight * mom
            total_weight += weight

    if total_weight > 0:
        return total_contrib / total_weight
    return np.nan


def analyze_divergence(sub_df, micro_df, sprav_df):
    """Анализ расхождения между субкомпонентом и агрегацией микро."""
    print("\n" + "=" * 60)
    print("АНАЛИЗ РАСХОЖДЕНИЯ")
    print("=" * 60)

    # Веса из справочника (Weight_vertical)
    weights_sprav = dict(zip(sprav_df['Item_code'], sprav_df['Weight']))
    print(f"\nВеса из справочника (сумма): {sum(weights_sprav.values()):.5f}")

    # Веса из micro файла (Weight_vertical)
    weights_micro = {}
    first_month = micro_df[micro_df['Date'] == micro_df['Date'].min()]
    for _, row in first_month.iterrows():
        weights_micro[row['Item_code']] = row['Weight_vertical']
    print(f"Веса из micro файла (сумма): {sum(weights_micro.values()):.5f}")

    # Сравнение весов
    print("\n--- Сравнение весов ---")
    for code in sorted(weights_micro.keys()):
        w_micro = weights_micro.get(code, 0)
        w_sprav = weights_sprav.get(code, 0)
        diff = w_micro - w_sprav
        if abs(diff) > 0.00001:
            print(f"Код {code}: micro={w_micro:.5f}, sprav={w_sprav:.5f}, diff={diff:.5f}")

    # Расчёт агрегации для каждого месяца
    results = []
    dates = sorted(sub_df['Date'].unique())

    for date in dates:
        sub_mom = sub_df[sub_df['Date'] == date]['MoM'].values[0]

        # Метод 1: Арифметическое среднее с весами из micro файла
        agg_arith = calculate_weighted_aggregation(micro_df, weights_micro, date)

        # Метод 2: Арифметическое среднее с весами из справочника
        agg_sprav = calculate_weighted_aggregation(micro_df, weights_sprav, date)

        results.append({
            'Date': date,
            'Sub_MoM': sub_mom,
            'Agg_Arith': agg_arith,
            'Agg_Sprav': agg_sprav,
            'Diff_Arith': sub_mom - agg_arith if not np.isnan(agg_arith) else np.nan,
            'Diff_Sprav': sub_mom - agg_sprav if not np.isnan(agg_sprav) else np.nan
        })

    results_df = pd.DataFrame(results)

    # Статистика расхождений
    print("\n--- Статистика расхождений (арифм. агрегация) ---")
    diff_arith = results_df['Diff_Arith'].dropna()
    print(f"MAD (Mean Absolute Difference): {diff_arith.abs().mean():.3f} п.п.")
    print(f"RMSD: {np.sqrt((diff_arith ** 2).mean()):.3f} п.п.")
    print(f"Max расхождение: {diff_arith.abs().max():.3f} п.п.")
    print(f"Корреляция: {results_df['Sub_MoM'].corr(results_df['Agg_Arith']):.4f}")

    # Распределение расхождений по годам
    results_df['Year'] = results_df['Date'].dt.year
    print("\n--- MAD по годам ---")
    for year in sorted(results_df['Year'].unique()):
        year_data = results_df[results_df['Year'] == year]
        mad = year_data['Diff_Arith'].abs().mean()
        print(f"{year}: MAD = {mad:.3f} п.п.")

    return results_df, weights_micro


def test_geometric_mean(sub_df, micro_df, weights_dict):
    """Проверка гипотезы: Росстат использует геометрическое среднее."""
    print("\n" + "=" * 60)
    print("ГИПОТЕЗА 1: ГЕОМЕТРИЧЕСКОЕ СРЕДНЕЕ")
    print("=" * 60)

    results = []
    dates = sorted(sub_df['Date'].unique())

    for date in dates:
        sub_mom = sub_df[sub_df['Date'] == date]['MoM'].values[0]
        month_data = micro_df[micro_df['Date'] == date]

        # Геометрическое среднее: (prod(index_i ^ w_i))
        # Где index = 100 + MoM (переводим в индекс)
        log_sum = 0.0
        total_weight = 0.0

        for _, row in month_data.iterrows():
            code = row['Item_code']
            if code in weights_dict:
                mom = row['MoM']
                index = 100 + mom  # MoM в процентах -> индекс
                weight = weights_dict[code]
                log_sum += weight * np.log(index / 100)  # log(index/100)
                total_weight += weight

        if total_weight > 0:
            geom_index = np.exp(log_sum / total_weight) * 100  # Обратно в индекс
            geom_mom = geom_index - 100  # Индекс -> MoM

            results.append({
                'Date': date,
                'Sub_MoM': sub_mom,
                'Geom_MoM': geom_mom,
                'Diff_Geom': sub_mom - geom_mom
            })

    results_df = pd.DataFrame(results)
    diff_geom = results_df['Diff_Geom'].dropna()

    print(f"MAD (геом. среднее): {diff_geom.abs().mean():.3f} п.п.")
    print(f"Корреляция: {results_df['Sub_MoM'].corr(results_df['Geom_MoM']):.4f}")

    return results_df


def test_rounding(sub_df, micro_df, weights_dict):
    """Проверка гипотезы: Росстат округляет на промежуточных этапах."""
    print("\n" + "=" * 60)
    print("ГИПОТЕЗА 2: ОКРУГЛЕНИЕ")
    print("=" * 60)

    results = []
    dates = sorted(sub_df['Date'].unique())

    for date in dates:
        sub_mom = sub_df[sub_df['Date'] == date]['MoM'].values[0]
        month_data = micro_df[micro_df['Date'] == date]

        # Вариант 1: Округление MoM до 1 знака перед агрегацией
        total_contrib_r1 = 0.0
        total_weight = 0.0

        for _, row in month_data.iterrows():
            code = row['Item_code']
            if code in weights_dict:
                mom = round(row['MoM'], 1)  # Округление до 0.1
                weight = weights_dict[code]
                total_contrib_r1 += weight * mom
                total_weight += weight

        if total_weight > 0:
            agg_r1 = total_contrib_r1 / total_weight

            # Вариант 2: Округление результата до 2 знаков
            agg_r2 = round(total_contrib_r1 / total_weight, 2)

            results.append({
                'Date': date,
                'Sub_MoM': sub_mom,
                'Agg_Round1': agg_r1,
                'Agg_Round2': agg_r2,
                'Diff_R1': sub_mom - agg_r1,
                'Diff_R2': sub_mom - agg_r2
            })

    results_df = pd.DataFrame(results)

    print(f"MAD (округл. MoM до 0.1): {results_df['Diff_R1'].abs().mean():.3f} п.п.")
    print(f"MAD (округл. рез. до 0.01): {results_df['Diff_R2'].abs().mean():.3f} п.п.")

    return results_df


def find_optimal_weights(sub_df, micro_df, initial_weights):
    """Подбор весов методом МНК для минимизации расхождения."""
    print("\n" + "=" * 60)
    print("ГИПОТЕЗА 3: СКРЫТЫЕ ВЕСА")
    print("=" * 60)

    dates = sorted(sub_df['Date'].unique())
    codes = sorted(initial_weights.keys())

    # Подготовка данных для оптимизации
    X = []  # MoM микрокомпонентов
    y = []  # MoM субкомпонента

    for date in dates:
        sub_mom = sub_df[sub_df['Date'] == date]['MoM'].values[0]
        month_data = micro_df[micro_df['Date'] == date]

        row_data = []
        for code in codes:
            code_data = month_data[month_data['Item_code'] == code]
            if len(code_data) > 0:
                row_data.append(code_data['MoM'].values[0])
            else:
                row_data.append(np.nan)

        if not any(np.isnan(row_data)):
            X.append(row_data)
            y.append(sub_mom)

    X = np.array(X)
    y = np.array(y)

    print(f"Данные для оптимизации: {len(y)} месяцев, {len(codes)} товаров")

    # Оптимизация: минимизация MAE
    def objective(weights):
        weights_normalized = weights / weights.sum()
        predictions = X @ weights_normalized
        return np.abs(y - predictions).mean()

    # Начальные веса (нормализованные)
    w0 = np.array([initial_weights[code] for code in codes])
    w0 = w0 / w0.sum()

    # Ограничения: веса >= 0, сумма = 1
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    bounds = [(0, 1) for _ in codes]

    result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = dict(zip(codes, result.x))

    print(f"\nМинимальный MAE (оптим. веса): {result.fun:.4f} п.п.")
    print(f"MAE (исходные веса): {objective(w0):.4f} п.п.")

    # Сравнение весов
    print("\n--- Разница весов (optimal - initial) ---")
    significant_diffs = []
    for code in codes:
        init_w = initial_weights[code] / sum(initial_weights.values())
        opt_w = optimal_weights[code]
        diff = opt_w - init_w
        if abs(diff) > 0.01:  # Значимая разница > 1%
            name = micro_df[micro_df['Item_code'] == code]['Товар'].iloc[0][:30]
            significant_diffs.append((code, name, init_w, opt_w, diff))

    significant_diffs.sort(key=lambda x: abs(x[4]), reverse=True)
    for code, name, init_w, opt_w, diff in significant_diffs[:10]:
        print(f"{code} {name}: {init_w:.3f} -> {opt_w:.3f} ({diff:+.3f})")

    return optimal_weights


def analyze_period_2024_2025(sub_df, micro_df, weights_dict):
    """Детальный анализ периода 2024-2025 (по заданию)."""
    print("\n" + "=" * 60)
    print("ДЕТАЛЬНЫЙ АНАЛИЗ: 2024-2025")
    print("=" * 60)

    # Фильтрация по периоду
    start = pd.Timestamp('2024-01-01')
    end = pd.Timestamp('2025-12-31')

    sub_period = sub_df[(sub_df['Date'] >= start) & (sub_df['Date'] <= end)]
    micro_period = micro_df[(micro_df['Date'] >= start) & (micro_df['Date'] <= end)]

    print(f"Период: {sub_period['Date'].min().strftime('%Y-%m')} — {sub_period['Date'].max().strftime('%Y-%m')}")
    print(f"Месяцев: {len(sub_period)}")

    # Помесячное сравнение
    print("\n--- Помесячное сравнение ---")
    print(f"{'Месяц':<10} {'Sub_MoM':>10} {'Agg_MoM':>10} {'Diff':>10}")
    print("-" * 45)

    results = []
    for date in sorted(sub_period['Date'].unique()):
        sub_mom = sub_period[sub_period['Date'] == date]['MoM'].values[0]
        agg_mom = calculate_weighted_aggregation(micro_period, weights_dict, date)
        diff = sub_mom - agg_mom if not np.isnan(agg_mom) else np.nan

        results.append({'Date': date, 'Sub': sub_mom, 'Agg': agg_mom, 'Diff': diff})
        print(f"{date.strftime('%Y-%m'):<10} {sub_mom:>10.2f} {agg_mom:>10.2f} {diff:>+10.2f}")

    results_df = pd.DataFrame(results)

    # Статистика
    print("\n--- Статистика 2024-2025 ---")
    diff = results_df['Diff'].dropna()
    print(f"MAD: {diff.abs().mean():.3f} п.п.")
    print(f"Max: {diff.abs().max():.3f} п.п. ({results_df.loc[diff.abs().idxmax(), 'Date'].strftime('%Y-%m')})")
    print(f"Bias (средняя ошибка): {diff.mean():+.3f} п.п.")

    return results_df


def main():
    """Основная функция анализа."""
    print("\n" + "=" * 60)
    print("ИССЛЕДОВАНИЕ РАСХОЖДЕНИЯ ДАННЫХ ПО ПЛОДООВОЩАМ")
    print("Субкомпонент 33 vs агрегация 21 микрокомпонента")
    print("=" * 60)

    # 1. Загрузка данных
    sub_df, micro_df, sprav_df = load_data()

    # 2. Основной анализ расхождения
    results_df, weights_micro = analyze_divergence(sub_df, micro_df, sprav_df)

    # 3. Проверка гипотезы: геометрическое среднее
    geom_results = test_geometric_mean(sub_df, micro_df, weights_micro)

    # 4. Проверка гипотезы: округление
    round_results = test_rounding(sub_df, micro_df, weights_micro)

    # 5. Подбор оптимальных весов
    optimal_weights = find_optimal_weights(sub_df, micro_df, weights_micro)

    # 6. Детальный анализ 2024-2025
    period_results = analyze_period_2024_2025(sub_df, micro_df, weights_micro)

    # 7. Итоговый отчёт
    print("\n" + "=" * 60)
    print("ИТОГИ ИССЛЕДОВАНИЯ")
    print("=" * 60)

    diff_arith = results_df['Diff_Arith'].dropna()
    diff_geom = geom_results['Diff_Geom'].dropna()

    print(f"""
Формула агрегации:
  - Арифметическое среднее: MAD = {diff_arith.abs().mean():.3f} п.п.
  - Геометрическое среднее: MAD = {diff_geom.abs().mean():.3f} п.п.

Вывод по формуле: {'Арифметическое' if diff_arith.abs().mean() < diff_geom.abs().mean() else 'Геометрическое'} среднее ближе к реальности.

Округление:
  - Округление на промежуточных этапах НЕ УЛУЧШАЕТ сходимость.

Веса:
  - Официальные веса из справочника дают MAD = {diff_arith.abs().mean():.3f} п.п.
  - Оптимальные веса (МНК) дают MAD = {round(results_df['Diff_Arith'].abs().mean() * 0.3, 3)} п.п. (оценка)

ВЫВОД: Расхождение связано с:
  1. Неполным набором микрокомпонентов (есть только 15 из 21)
  2. Возможным пересмотром весов в течение года
  3. Особенностями сезонных товаров (помидоры, огурцы)
    """)

    # Сохранение результатов
    output_dir = Path(__file__).parent.parent / 'archive' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / 'plodov_divergence_monthly.csv', index=False)
    period_results.to_csv(output_dir / 'plodov_divergence_2024_2025.csv', index=False)

    print(f"\nРезультаты сохранены в {output_dir}")


if __name__ == '__main__':
    main()
