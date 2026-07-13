"""
ГЛУБОКОЕ ИССЛЕДОВАНИЕ АНОМАЛИЙ КБР в 2025
Анализ всех 933 товарных позиций по всем месяцам
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("ЗАГРУЗКА ДАННЫХ...")
print("="*100)

# Загрузка детальных данных
df = pd.read_csv('/home/valalav/_projects/sirena-kbr/data/data_indices_dump.csv')
items = pd.read_csv('/home/valalav/_projects/sirena-kbr/data/items_names.csv')

print(f"Загружено {len(df):,} записей")
print(f"Справочник: {len(items)} товарных позиций")

# Преобразование дат - формат MM/DD/YY HH:MM:SS
df['Date'] = pd.to_datetime(df['Day'], format='%m/%d/%y %H:%M:%S', errors='coerce')
# Исправляем года после 2000: если год < 100, добавляем 2000
df['Year'] = df['Date'].dt.year
df.loc[df['Year'] < 100, 'Year'] = df.loc[df['Year'] < 100, 'Year'] + 2000
df['Month'] = df['Date'].dt.month
df['Month_name'] = df['Date'].dt.strftime('%B')

# Фильтр 2025 год
df_2025 = df[df['Year'] == 2025].copy()
print(f"Записей за 2025: {len(df_2025):,}")

# Константы
RUSSIA = 0
KBR = 7

print("\n" + "="*100)
print("АНАЛИЗ 1: ТОП-100 ТОВАРОВ С МАКСИМАЛЬНЫМИ ОТКЛОНЕНИЯМИ КБР ОТ РФ (по всем месяцам 2025)")
print("="*100)

# Для каждого месяца и товара найти отклонение
anomalies = []

for month in sorted(df_2025['Month'].unique()):
    month_data = df_2025[df_2025['Month'] == month]
    month_name = month_data.iloc[0]['Month_name'] if len(month_data) > 0 else str(month)

    # Группируем по товару
    for item_code in month_data['Item_code'].unique():
        item_data = month_data[month_data['Item_code'] == item_code]

        rf_data = item_data[item_data['Region_code'] == RUSSIA]
        kbr_data = item_data[item_data['Region_code'] == KBR]

        if len(rf_data) > 0 and len(kbr_data) > 0:
            rf_mom = rf_data.iloc[0]['MoM']
            kbr_mom = kbr_data.iloc[0]['MoM']

            # Пропускаем NaN
            if pd.notna(rf_mom) and pd.notna(kbr_mom):
                diff = kbr_mom - rf_mom
                abs_diff = abs(diff)

                # Получаем название товара
                item_name = items[items['Item_code'] == item_code]['Item_name'].values
                item_name = item_name[0] if len(item_name) > 0 else f"Item {item_code}"

                anomalies.append({
                    'Month': month,
                    'Month_name': month_name,
                    'Item_code': item_code,
                    'Item_name': item_name,
                    'RF_MoM': rf_mom,
                    'KBR_MoM': kbr_mom,
                    'Difference': diff,
                    'Abs_Difference': abs_diff,
                    'RF_change': rf_mom - 100,
                    'KBR_change': kbr_mom - 100
                })

anomalies_df = pd.DataFrame(anomalies)
print(f"\nНайдено {len(anomalies_df)} пар (месяц, товар) для сравнения")

# Топ-100 по абсолютному отклонению
top100 = anomalies_df.nlargest(100, 'Abs_Difference')

print("\nТОП-50 САМЫХ АНОМАЛЬНЫХ РАСХОЖДЕНИЙ:")
print("-"*100)
for i, row in top100.head(50).iterrows():
    print(f"{row['Month_name']:10} | {row['Item_name'][:50]:50} | "
          f"РФ: {row['RF_change']:+7.2f}% | КБР: {row['KBR_change']:+7.2f}% | "
          f"Разрыв: {row['Difference']:+7.2f}")

print("\n" + "="*100)
print("АНАЛИЗ 2: ГРУППИРОВКА ПО КАТЕГОРИЯМ ТОВАРОВ")
print("="*100)

# Анализируем по месяцам - где самые большие отклонения
monthly_stats = anomalies_df.groupby('Month_name').agg({
    'Abs_Difference': ['mean', 'max', 'count'],
    'Difference': 'mean'
}).round(2)
monthly_stats.columns = ['Средн_откл', 'Макс_откл', 'Кол-во_товаров', 'Сред_разрыв']
monthly_stats = monthly_stats.sort_values('Макс_откл', ascending=False)

print("\nМЕСЯЦЫ С НАИБОЛЬШИМИ РАСХОЖДЕНИЯМИ:")
print(monthly_stats)

print("\n" + "="*100)
print("АНАЛИЗ 3: ТОВАРЫ С УСТОЙЧИВО ВЫСОКИМИ ОТКЛОНЕНИЯМИ (по нескольким месяцам)")
print("="*100)

# Товары которые аномальны в нескольких месяцах
item_anomalies = anomalies_df.groupby('Item_name').agg({
    'Abs_Difference': ['mean', 'max', 'count'],
    'Difference': 'mean'
}).round(2)
item_anomalies.columns = ['Средн_откл', 'Макс_откл', 'Месяцев', 'Направл']
item_anomalies = item_anomalies[item_anomalies['Месяцев'] >= 3]  # Минимум 3 месяца
item_anomalies = item_anomalies.sort_values('Средн_откл', ascending=False)

print(f"\nТОП-30 ТОВАРОВ С УСТОЙЧИВЫМИ АНОМАЛИЯМИ (минимум 3 месяца):")
print(item_anomalies.head(30))

print("\n" + "="*100)
print("АНАЛИЗ 4: КОНКРЕТНЫЕ ПРИМЕРЫ - ФЕВРАЛЬ, ИЮЛЬ, СЕНТЯБРЬ, НОЯБРЬ 2025")
print("="*100)

key_months = {2: 'Февраль', 7: 'Июль', 9: 'Сентябрь', 11: 'Ноябрь'}

for month_num, month_name in key_months.items():
    month_anomalies = anomalies_df[anomalies_df['Month'] == month_num]
    if len(month_anomalies) > 0:
        top10 = month_anomalies.nlargest(10, 'Abs_Difference')
        print(f"\n{month_name.upper()} 2025 - Топ-10 расхождений:")
        print("-"*100)
        for i, row in top10.iterrows():
            print(f"{row['Item_name'][:50]:50} | РФ: {row['RF_change']:+7.2f}% | "
                  f"КБР: {row['KBR_change']:+7.2f}% | Разрыв: {row['Difference']:+7.2f}")

print("\n" + "="*100)
print("АНАЛИЗ 5: ИСТОРИЧЕСКИЙ КОНТЕКСТ (сравнение с 2020-2024)")
print("="*100)

# Загружаем данные за все годы для сравнения
df_hist = df[(df['Year'] >= 2020) & (df['Year'] <= 2024)].copy()

# Для КБР и РФ считаем среднее отклонение по месяцам
hist_anomalies = []

for year in range(2020, 2025):
    year_data = df_hist[df_hist['Year'] == year]

    for month in range(1, 13):
        month_data = year_data[year_data['Month'] == month]

        rf_month = month_data[month_data['Region_code'] == RUSSIA]['MoM']
        kbr_month = month_data[month_data['Region_code'] == KBR]['MoM']

        if len(rf_month) > 0 and len(kbr_month) > 0:
            # Среднее отклонение по всем товарам
            rf_mean = rf_month.mean()
            kbr_mean = kbr_month.mean()

            hist_anomalies.append({
                'Year': year,
                'Month': month,
                'Avg_Difference': abs(kbr_mean - rf_mean)
            })

hist_df = pd.DataFrame(hist_anomalies)
yearly_avg = hist_df.groupby('Year')['Avg_Difference'].mean()

print("\nСРЕДНЕЕ ОТКЛОНЕНИЕ КБР ОТ РФ ПО ГОДАМ:")
for year, avg_diff in yearly_avg.items():
    print(f"  {year}: {avg_diff:.3f}")

# Сравним с 2025
anomalies_2025_avg = anomalies_df.groupby('Month')['Abs_Difference'].mean().mean()
print(f"\n  2025: {anomalies_2025_avg:.3f} <-- АНОМАЛИЯ!")

if anomalies_2025_avg > yearly_avg.mean():
    increase = ((anomalies_2025_avg / yearly_avg.mean() - 1) * 100)
    print(f"\nРост отклонений в 2025 относительно 2020-2024: +{increase:.1f}%")

print("\n" + "="*100)
print("АНАЛИЗ 6: СПЕЦИФИЧЕСКИЕ КАТЕГОРИИ С ЭКСТРЕМАЛЬНЫМИ РАСХОЖДЕНИЯМИ")
print("="*100)

# Ищем категории с экстремумами
extreme_positive = top100[top100['Difference'] > 10].sort_values('Difference', ascending=False)
extreme_negative = top100[top100['Difference'] < -10].sort_values('Difference')

print("\nТОВАРЫ ГДЕ КБР РЕЗКО ВЫШЕ РФ (разрыв > 10 п.п.):")
print(f"Найдено {len(extreme_positive)} случаев\n")
for i, row in extreme_positive.head(20).iterrows():
    print(f"{row['Month_name']:10} | {row['Item_name'][:50]:50} | "
          f"РФ: {row['RF_change']:+6.2f}% | КБР: {row['KBR_change']:+6.2f}% | "
          f"Разрыв: {row['Difference']:+6.2f}")

print("\n\nТОВАРЫ ГДЕ КБР РЕЗКО НИЖЕ РФ (разрыв < -10 п.п.):")
print(f"Найдено {len(extreme_negative)} случаев\n")
for i, row in extreme_negative.head(20).iterrows():
    print(f"{row['Month_name']:10} | {row['Item_name'][:50]:50} | "
          f"РФ: {row['RF_change']:+6.2f}% | КБР: {row['KBR_change']:+6.2f}% | "
          f"Разрыв: {row['Difference']:+6.2f}")

print("\n" + "="*100)
print("ЭКСПОРТ РЕЗУЛЬТАТОВ")
print("="*100)

# Сохраняем топ-100 в CSV
top100.to_csv('/home/valalav/_projects/sirena-kbr/data/top100_anomalies_2025.csv',
              index=False, encoding='utf-8-sig')
print("✓ Топ-100 аномалий сохранен: data/top100_anomalies_2025.csv")

# Сохраняем полный список аномалий
anomalies_df.to_csv('/home/valalav/_projects/sirena-kbr/data/all_anomalies_2025.csv',
                    index=False, encoding='utf-8-sig')
print("✓ Все аномалии сохранены: data/all_anomalies_2025.csv")

print("\n" + "="*100)
print("АНАЛИЗ ЗАВЕРШЕН!")
print("="*100)
