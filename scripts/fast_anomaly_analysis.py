"""
БЫСТРЫЙ АНАЛИЗ АНОМАЛИЙ КБР - используем меньшие файлы
"""

import pandas as pd
import numpy as np

print("="*100)
print("ЗАГРУЗКА ДАННЫХ (kbr_indices.csv + справочники)")
print("="*100)

# Загружаем данные КБР
df_kbr = pd.read_csv('/home/valalav/_projects/sirena-kbr/data/kbr_indices.csv')
print(f"КБР: {len(df_kbr):,} записей")

# Загружаем данные РФ из all_regions
df_all = pd.read_csv('/home/valalav/_projects/sirena-kbr/data/all_regions_indices.csv')
df_rf = df_all[df_all['Region_code'] == 0].copy()
print(f"РФ (из all_regions): {len(df_rf):,} записей")

# Справочник товаров
items = pd.read_csv('/home/valalav/_projects/sirena-kbr/data/items_names.csv')
print(f"Товары: {len(items)} позиций")

# Парсим даты
df_kbr['Date'] = pd.to_datetime(df_kbr['Day'], format='%m/%d/%y %H:%M:%S', errors='coerce')
df_kbr['Year'] = df_kbr['Date'].dt.year
df_kbr['Month'] = df_kbr['Date'].dt.month

df_rf['Date'] = pd.to_datetime(df_rf['Day'], format='%d/%m/%y %H:%M:%S', errors='coerce')
df_rf['Year'] = df_rf['Date'].dt.year
df_rf['Month'] = df_rf['Date'].dt.month

# Фильтруем 2025
df_kbr_2025 = df_kbr[df_kbr['Year'] == 2025].copy()
df_rf_2025 = df_rf[df_rf['Year'] == 2025].copy()

print(f"\nКБР 2025: {len(df_kbr_2025):,} записей")
print(f"РФ 2025: {len(df_rf_2025):,} записей")

# Теперь загрузим детальные данные ТОЛЬКО для РФ и КБР из большого файла
print("\nЗагрузка ДЕТАЛЬНЫХ данных из data_indices_dump.csv...")
print("(только РФ и КБР за 2025, это займет минуту...)")

chunks = []
chunksize = 100000
for chunk in pd.read_csv('/home/valalav/_projects/sirena-kbr/data/data_indices_dump.csv', chunksize=chunksize):
    # Парсим даты
    chunk['Date'] = pd.to_datetime(chunk['Day'], format='%m/%d/%y %H:%M:%S', errors='coerce')
    chunk['Year'] = chunk['Date'].dt.year
    chunk['Month'] = chunk['Date'].dt.month

    # Фильтруем только 2025, только РФ и КБР
    chunk_filtered = chunk[(chunk['Year'] == 2025) & (chunk['Region_code'].isin([0, 7]))]

    if len(chunk_filtered) > 0:
        chunks.append(chunk_filtered)

    # Показываем прогресс
    if len(chunks) % 10 == 0 and len(chunks) > 0:
        print(f"  Обработано {len(chunks) * chunksize:,} строк...")

if len(chunks) > 0:
    df_detailed = pd.concat(chunks, ignore_index=True)
    print(f"\nЗагружено детальных данных: {len(df_detailed):,} записей")
else:
    print("\nНет детальных данных за 2025!")
    df_detailed = pd.DataFrame()

print("\n" + "="*100)
print("АНАЛИЗ: ДЕТАЛЬНЫЕ ТОВАРЫ С МАКСИМАЛЬНЫМИ РАСХОЖДЕНИЯМИ")
print("="*100)

if len(df_detailed) > 0:
    # Merge РФ и КБР для сравнения
    anomalies = []

    for month in sorted(df_detailed['Month'].unique()):
        month_data = df_detailed[df_detailed['Month'] == month]

        for item_code in month_data['Item_code'].unique():
            item_data = month_data[month_data['Item_code'] == item_code]

            rf = item_data[item_data['Region_code'] == 0]
            kbr = item_data[item_data['Region_code'] == 7]

            if len(rf) > 0 and len(kbr) > 0:
                rf_mom = rf.iloc[0]['MoM']
                kbr_mom = kbr.iloc[0]['MoM']

                if pd.notna(rf_mom) and pd.notna(kbr_mom):
                    diff = kbr_mom - rf_mom

                    # Название товара
                    item_name = items[items['Item_code'] == item_code]['Item_name'].values
                    item_name = item_name[0] if len(item_name) > 0 else f"Item {item_code}"

                    anomalies.append({
                        'Month': month,
                        'Item_code': item_code,
                        'Item_name': item_name,
                        'RF_MoM': rf_mom - 100,
                        'KBR_MoM': kbr_mom - 100,
                        'Difference': diff,
                        'Abs_Difference': abs(diff)
                    })

    anom_df = pd.DataFrame(anomalies)

    # ТОП-100 аномалий
    top100 = anom_df.nlargest(100, 'Abs_Difference')

    print(f"\nВсего найдено {len(anom_df)} пар для сравнения")
    print("\nТОП-50 САМЫХ АНОМАЛЬНЫХ РАСХОЖДЕНИЙ:")
    print("-"*100)

    month_names = {1:'Январь', 2:'Февраль', 3:'Март', 4:'Апрель', 5:'Май', 6:'Июнь',
                   7:'Июль', 8:'Август', 9:'Сентябрь', 10:'Октябрь', 11:'Ноябрь', 12:'Декабрь'}

    for i, row in top100.head(50).iterrows():
        month_name = month_names.get(row['Month'], str(row['Month']))
        print(f"{month_name:10} | {row['Item_name'][:50]:50} | "
              f"РФ: {row['RF_MoM']:+7.2f}% | КБР: {row['KBR_MoM']:+7.2f}% | "
              f"Разрыв: {row['Difference']:+7.2f}")

    # Группировка по месяцам
    print("\n" + "="*100)
    print("СТАТИСТИКА ПО МЕСЯЦАМ")
    print("="*100)

    monthly_stats = anom_df.groupby('Month').agg({
        'Abs_Difference': ['mean', 'max', 'count'],
        'Difference': 'mean'
    }).round(2)
    monthly_stats.columns = ['Средн_откл', 'Макс_откл', 'Товаров', 'Напр_средн']

    for month, stats in monthly_stats.iterrows():
        month_name = month_names.get(month, str(month))
        print(f"{month_name:10} | Товаров: {int(stats['Товаров']):4} | "
              f"Средн откл: {stats['Средн_откл']:5.2f} | "
              f"Макс откл: {stats['Макс_откл']:6.2f} | "
              f"Направл: {stats['Напр_средн']:+6.2f}")

    # Сохраняем результаты
    top100.to_csv('/home/valalav/_projects/sirena-kbr/data/top100_anomalies_detailed.csv',
                  index=False, encoding='utf-8-sig')
    print("\n✓ Сохранено: data/top100_anomalies_detailed.csv")

print("\n" + "="*100)
print("ГОТОВО!")
print("="*100)
