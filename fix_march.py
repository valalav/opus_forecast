"""
Исправление проблемы марта
==========================

Анализ показал:
- Март 2022 (СВО) - ошибка 4.76 п.п., что искажает общую статистику
- Без 2022 MAE марта ≈ 0.19 п.п. (нормально)
- Модель НЕДООЦЕНИВАЕТ март (bias > 0.7)

Варианты исправления:
1. Увеличить вес ETS для марта (текущий 0.5 → 0.7)
2. Добавить специальную обработку кризисных периодов
3. Исключить март 2022 из бэктеста (не рекомендуется)
"""

import pandas as pd
import numpy as np
from sirena_kbr_v2_4_auto import SirenaKBR_v24
from copy import deepcopy

# Загрузка данных
df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal='.')
try:
    df_raw['Date'] = pd.to_datetime(df_raw['Day'], format='%d.%m.%Y')
except:
    df_raw['Date'] = pd.to_datetime(df_raw['Day'])

if 'Товар' in df_raw.columns and 'MoM' in df_raw.columns:
    df = df_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
else:
    df = df_raw.set_index('Date')

df = df[['Все товары и услуги', 'Продовольственные товары', 'Непродовольственные товары', 'Услуги']].copy()
df = df.sort_index()


def test_ets_weights(df, test_weights):
    """Тестирование различных весов ETS для марта."""
    model = SirenaKBR_v24()

    # Переопределяем веса
    model.ETS_WEIGHTS = deepcopy(model.ETS_WEIGHTS)
    model.ETS_WEIGHTS[3] = test_weights  # Март

    results = model.backtest(df, start_date='2019-01-01')
    results['month'] = pd.to_datetime(results['date']).dt.month
    results['year'] = pd.to_datetime(results['date']).dt.year
    results['abs_error'] = results['error'].abs()

    # Исключаем 2022 для честного сравнения
    results_clean = results[results['year'] != 2022]

    march_mae = results_clean[results_clean['month'] == 3]['abs_error'].mean()
    total_mae = results_clean['abs_error'].mean()

    return march_mae, total_mae


print("=" * 60)
print("ТЕСТИРОВАНИЕ РАЗЛИЧНЫХ ВЕСОВ ETS ДЛЯ МАРТА")
print("=" * 60)
print(f"{'Вес ETS':<10} {'MAE март (без 2022)':<20} {'MAE общий (без 2022)':<20}")
print("-" * 60)

best_weight = 0.5
best_march_mae = float('inf')

for weight in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
    march_mae, total_mae = test_ets_weights(df, weight)
    print(f"{weight:<10} {march_mae:<20.3f} {total_mae:<20.3f}")

    if march_mae < best_march_mae:
        best_march_mae = march_mae
        best_weight = weight

print("-" * 60)
print(f"\nОптимальный вес ETS для марта: {best_weight}")
print(f"MAE марта с оптимальным весом: {best_march_mae:.3f}")

# Тест полного бэктеста (с 2022)
print("\n" + "=" * 60)
print("ВЛИЯНИЕ НА ПОЛНЫЙ БЭКТЕСТ (включая 2022)")
print("=" * 60)

for weight in [0.5, best_weight]:
    model = SirenaKBR_v24()
    model.ETS_WEIGHTS = deepcopy(model.ETS_WEIGHTS)
    model.ETS_WEIGHTS[3] = weight

    results = model.backtest(df, start_date='2019-01-01')
    results['abs_error'] = results['error'].abs()

    mae = results['abs_error'].mean()
    kpi = (results['abs_error'] <= 0.5).sum()
    total = len(results)

    print(f"Вес ETS={weight}: MAE={mae:.3f}, KPI={kpi}/{total} ({kpi/total*100:.1f}%)")

print("\n" + "=" * 60)
print("РЕКОМЕНДАЦИЯ")
print("=" * 60)
print(f"""
Изменить вес ETS для марта с 0.5 на {best_weight}.

В файле sirena_kbr_v2_4_auto.py:
    ETS_WEIGHTS = {{
        ...
        3: {best_weight},  # Март (было 0.5)
        ...
    }}

Также в dashboard.py обновить соответствующий параметр.
""")
