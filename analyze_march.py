"""
Анализ проблемы марта в модели СИРЕНА-КБР
=========================================

Исследование причин высокой ошибки прогнозирования в марте.
"""

import pandas as pd
import numpy as np
from sirena_kbr_v2_4_auto import SirenaKBR_v24

# Загрузка данных
df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal='.')

# Парсинг
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

print("=" * 60)
print("АНАЛИЗ ПРОБЛЕМЫ МАРТА")
print("=" * 60)

# Запуск бэктеста
model = SirenaKBR_v24()
results = model.backtest(df, start_date='2019-01-01')

if results.empty:
    print("Нет результатов бэктеста")
    exit()

# Добавляем месяц и год
results['month'] = pd.to_datetime(results['date']).dt.month
results['year'] = pd.to_datetime(results['date']).dt.year
results['abs_error'] = results['error'].abs()

print("\n1. СТАТИСТИКА ОШИБОК ПО МЕСЯЦАМ")
print("-" * 40)
monthly_stats = results.groupby('month').agg({
    'abs_error': ['mean', 'std', 'count', 'max'],
    'error': 'mean'  # Bias
}).round(3)
monthly_stats.columns = ['MAE', 'Std', 'Count', 'Max_Error', 'Bias']
print(monthly_stats.to_string())

# Выделяем март
march_data = results[results['month'] == 3]
print(f"\n2. ДЕТАЛЬНЫЙ АНАЛИЗ МАРТА (n={len(march_data)})")
print("-" * 40)
print(f"MAE марта:  {march_data['abs_error'].mean():.3f}")
print(f"Bias марта: {march_data['error'].mean():.3f}")
print(f"Max ошибка: {march_data['abs_error'].max():.3f}")

print("\nОшибки по годам (март):")
for _, row in march_data.iterrows():
    print(f"  {row['year']}: факт={row['actual']-100:.2f}%, "
          f"прогноз={row['prediction']-100:.2f}%, "
          f"ошибка={row['error']:.2f}")

# Анализ сезонности в марте
print("\n3. СЕЗОННОСТЬ В МАРТЕ (исторические данные)")
print("-" * 40)
df['month'] = df.index.month
df['year'] = df.index.year

march_history = df[df['month'] == 3]['Все товары и услуги']
print(f"Среднее (март): {march_history.mean()-100:.3f}%")
print(f"Std (март):     {march_history.std():.3f}")
print(f"Min:            {march_history.min()-100:.3f}%")
print(f"Max:            {march_history.max()-100:.3f}%")

print("\nМартовская инфляция по годам:")
for year in sorted(df['year'].unique()):
    val = df[(df['year'] == year) & (df['month'] == 3)]['Все товары и услуги']
    if not val.empty:
        print(f"  {year}: {val.values[0]-100:.2f}%")

# Компоненты в марте
print("\n4. КОМПОНЕНТЫ В МАРТЕ")
print("-" * 40)
march_all = df[df['month'] == 3]
for col in ['Продовольственные товары', 'Непродовольственные товары', 'Услуги']:
    if col in march_all.columns:
        vals = march_all[col]
        print(f"{col[:15]:15} | Mean: {vals.mean()-100:.2f}% | Std: {vals.std():.2f}")

# Связь с февралём (лаг-1)
print("\n5. СВЯЗЬ МАРТ - ФЕВРАЛЬ")
print("-" * 40)
df['prev_month'] = df['Все товары и услуги'].shift(1)
feb_march = df[df['month'] == 3][['Все товары и услуги', 'prev_month']].dropna()
if len(feb_march) > 2:
    corr = feb_march['Все товары и услуги'].corr(feb_march['prev_month'])
    print(f"Корреляция март vs февраль: {corr:.3f}")

# Рекомендации
print("\n6. ВЫВОДЫ И РЕКОМЕНДАЦИИ")
print("-" * 40)

march_mae = march_data['abs_error'].mean()
other_mae = results[results['month'] != 3]['abs_error'].mean()

print(f"MAE март:         {march_mae:.3f}")
print(f"MAE остальные:    {other_mae:.3f}")
print(f"Разница:          {march_mae - other_mae:.3f} п.п.")

if march_mae > other_mae * 1.5:
    print("\n⚠️ Март значительно хуже остальных месяцев!")

    if march_data['error'].mean() > 0.2:
        print("   → Модель НЕДООЦЕНИВАЕТ март (bias > 0)")
        print("   → Рекомендация: увеличить вес ETS для марта")
    elif march_data['error'].mean() < -0.2:
        print("   → Модель ПЕРЕОЦЕНИВАЕТ март (bias < 0)")
        print("   → Рекомендация: уменьшить вес ETS для марта")
    else:
        print("   → Bias близок к нулю, но высокая дисперсия")
        print("   → Рекомендация: добавить специфические признаки для марта")
