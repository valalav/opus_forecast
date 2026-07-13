"""
Скрипт для анализа аномалий инфляции КБР в 2025 году
Сравнение с РФ и регионами ЮФО/СКФО
"""

import pandas as pd
import numpy as np

# Загрузка данных
print("Загрузка данных...")
df_agg = pd.read_csv('/home/valalav/_projects/sirena-kbr/data/all_regions_indices.csv')

# Преобразование дат
df_agg['Date'] = pd.to_datetime(df_agg['Date'])
df_agg['Year'] = df_agg['Date'].dt.year
df_agg['Month'] = df_agg['Date'].dt.month

# Фильтрация 2024-2025
df_2425 = df_agg[df_agg['Year'].isin([2024, 2025])].copy()

# Регионы интереса
RUSSIA = 0
KBR = 7
KRASNODAR = 23
ROSTOV = 61

# Категории
ITEM_ALL = 1  # Все товары и услуги
ITEM_NONPROD = 2  # Непродовольственные
ITEM_PROD = 3  # Продовольственные
ITEM_SERVICES = 4  # Услуги
ITEM_FRUITS = 33  # Плодоовощи

print("\n" + "="*80)
print("1. СРАВНЕНИЕ КБР vs РФ: ИЮЛЬ 2025 (пик расхождений)")
print("="*80)

july_2025 = df_2425[(df_2425['Year'] == 2025) & (df_2425['Month'] == 7)]

for item_code, item_name in [(ITEM_ALL, "Все товары и услуги"),
                             (ITEM_SERVICES, "Услуги"),
                             (ITEM_PROD, "Продовольствие"),
                             (ITEM_FRUITS, "Плодоовощи")]:
    rf = july_2025[(july_2025['Region_code'] == RUSSIA) & (july_2025['Item_code'] == item_code)]
    kbr = july_2025[(july_2025['Region_code'] == KBR) & (july_2025['Item_code'] == item_code)]

    if not rf.empty and not kbr.empty:
        rf_mom = rf.iloc[0]['MoM']
        kbr_mom = kbr.iloc[0]['MoM']
        diff = kbr_mom - rf_mom

        print(f"\n{item_name}:")
        print(f"  РФ:  {rf_mom:.2f}% ({rf_mom-100:+.2f} п.п.)")
        print(f"  КБР: {kbr_mom:.2f}% ({kbr_mom-100:+.2f} п.п.)")
        print(f"  Разрыв: {diff:+.2f} п.п.")

print("\n" + "="*80)
print("2. ВОЛАТИЛЬНОСТЬ 2024 vs 2025 (КБР)")
print("="*80)

# КБР, все товары
kbr_all = df_2425[(df_2425['Region_code'] == KBR) & (df_2425['Item_code'] == ITEM_ALL)].copy()
kbr_all['MoM_change'] = kbr_all['MoM'] - 100

kbr_2024 = kbr_all[kbr_all['Year'] == 2024]['MoM_change']
kbr_2025 = kbr_all[(kbr_all['Year'] == 2025) & (kbr_all['Month'] <= 11)]['MoM_change']

print(f"\n2024 год (янв-ноя):")
print(f"  Диапазон: {kbr_2024.min():.2f}% до {kbr_2024.max():.2f}%")
print(f"  Месяцев с дефляцией: {(kbr_2024 < 0).sum()}")
print(f"  Std Dev: {kbr_2024.std():.3f}")

print(f"\n2025 год (янв-ноя):")
print(f"  Диапазон: {kbr_2025.min():.2f}% до {kbr_2025.max():.2f}%")
print(f"  Месяцев с дефляцией: {(kbr_2025 < 0).sum()}")
print(f"  Std Dev: {kbr_2025.std():.3f}")
print(f"  Рост волатильности: {((kbr_2025.std() / kbr_2024.std() - 1) * 100):.1f}%")

print("\n" + "="*80)
print("3. РЕГИОНАЛЬНОЕ СРАВНЕНИЕ (янв-ноя 2025)")
print("="*80)

# Список регионов ЮФО/СКФО
regions_yfo_skfo = {
    0: "Россия",
    7: "Кабардино-Балкария",
    23: "Краснодарский край",
    26: "Ставропольский край",
    61: "Ростовская область",
    34: "Волгоградская область",
    30: "Астраханская область",
    1: "Республика Адыгея",
    8: "Карачаево-Черкесия",
    15: "Северная Осетия-Алания",
    5: "Республика Дагестан",
    20: "Чеченская Республика",
    6: "Республика Ингушетия",
    92: "Севастополь",
}

volatility_data = []
for region_code, region_name in regions_yfo_skfo.items():
    region_data = df_2425[(df_2425['Region_code'] == region_code) &
                          (df_2425['Item_code'] == ITEM_ALL) &
                          (df_2425['Year'] == 2025) &
                          (df_2425['Month'] <= 11)].copy()

    if len(region_data) > 0:
        region_data['MoM_change'] = region_data['MoM'] - 100
        volatility = region_data['MoM_change'].std()
        volatility_data.append({
            'Region': region_name,
            'Code': region_code,
            'Volatility': volatility
        })

vol_df = pd.DataFrame(volatility_data).sort_values('Volatility', ascending=False)
print("\nТоп-15 регионов по волатильности (Std Dev):")
print(vol_df.to_string(index=False))

# Найти место КБР
kbr_rank = vol_df[vol_df['Code'] == KBR].index[0] + 1
kbr_vol = vol_df[vol_df['Code'] == KBR].iloc[0]['Volatility']
krasnodar_vol = vol_df[vol_df['Code'] == KRASNODAR].iloc[0]['Volatility']

print(f"\nМесто КБР: {kbr_rank} из {len(vol_df)}")
print(f"КБР волатильность: {kbr_vol:.3f}")
print(f"Краснодар волатильность: {krasnodar_vol:.3f}")
print(f"КБР выше Краснодара на: {((kbr_vol / krasnodar_vol - 1) * 100):.1f}%")

print("\n" + "="*80)
print("4. СЕНТЯБРЬ 2025: КБР vs СОСЕДИ")
print("="*80)

sept_2025 = df_2425[(df_2425['Year'] == 2025) & (df_2425['Month'] == 9)]

comparison_regions = {
    7: "КБР",
    23: "Краснодарский край",
    61: "Ростовская область",
    26: "Ставропольский край"
}

print("\nМесячная инфляция (MoM):")
for code, name in comparison_regions.items():
    data = sept_2025[(sept_2025['Region_code'] == code) & (sept_2025['Item_code'] == ITEM_ALL)]
    if not data.empty:
        mom = data.iloc[0]['MoM']
        print(f"  {name}: {mom-100:+.2f}%")

print("\n" + "="*80)
print("ГОТОВО!")
print("="*80)
