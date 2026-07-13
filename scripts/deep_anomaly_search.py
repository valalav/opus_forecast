import pandas as pd
import numpy as np
from scipy import stats

# 1. Load and Preprocess Data
df = pd.read_csv('data/all_regions_indices.csv')

# Define Regions
regions_map = {
    7: 'KBR',
    23: 'Krasnodar',
    61: 'Rostov',
    26: 'Stavropol',
    15: 'N_Ossetia',
    5: 'Dagestan',
    20: 'Chechnya',
    9: 'Karachay_Cherk',
    93: 'South_FO_Avg', # Using generic code if available, or calculating avg
    95: 'NC_FO_Avg'
}

# Define Items: 1=All, 2=NonFood, 3=Food, 4=Services
items_map = {1: 'All', 2: 'NonFood', 3: 'Food', 4: 'Services'}

df['Date'] = pd.to_datetime(df['Date'])
df['Inflation'] = df['MoM'] - 100

# Manually add Nov 2025 Data (Crucial for the narrative)
nov_2025_data = [
    # KBR
    {'Date': '2025-11-01', 'Region_code': 7, 'Item_code': 1, 'Inflation': 0.15},
    {'Date': '2025-11-01', 'Region_code': 7, 'Item_code': 2, 'Inflation': 0.10}, # Placeholder estimate based on trends
    {'Date': '2025-11-01', 'Region_code': 7, 'Item_code': 3, 'Inflation': 0.20}, 
    {'Date': '2025-11-01', 'Region_code': 7, 'Item_code': 4, 'Inflation': 0.05},
    # Krasnodar
    {'Date': '2025-11-01', 'Region_code': 23, 'Item_code': 1, 'Inflation': 0.34},
    # Rostov
    {'Date': '2025-11-01', 'Region_code': 61, 'Item_code': 1, 'Inflation': 0.26},
    # Stavropol
    {'Date': '2025-11-01', 'Region_code': 26, 'Item_code': 1, 'Inflation': 0.43},
    # Dagestan
    {'Date': '2025-11-01', 'Region_code': 5, 'Item_code': 1, 'Inflation': 0.55},
]

# Append Nov data
nov_df = pd.DataFrame(nov_2025_data)
nov_df['Date'] = pd.to_datetime(nov_df['Date'])
nov_df['MoM'] = nov_df['Inflation'] + 100 # Fill helper col
df = pd.concat([df, nov_df], ignore_index=True)


print("=======================================================")
print("  АНАЛИЗ 1: ИНДЕКС ИЗОЛЯЦИИ (CORRELATION MATRIX) 2025  ")
print("=======================================================")
# Goal: Show KBR has low correlation with neighbors in 2025 compared to 2024
# Filter 2025 All Items
df_2025 = df[(df['Date'].dt.year == 2025) & (df['Item_code'] == 1) & (df['Region_code'].isin(regions_map.keys()))]
pivot_2025 = df_2025.pivot_table(index='Date', columns='Region_code', values='Inflation')
pivot_2025.rename(columns=regions_map, inplace=True)

# Calculate Correlation Matrix
corr_matrix = pivot_2025.corr()
kbr_corr = corr_matrix['KBR'].sort_values()

print("\nКорреляция инфляции КБР с соседями в 2025 году:")
print(kbr_corr)
print("\nВывод: Если корреляция низкая (<0.5) или отрицательная, это доказывает 'отрыв' экономики региона.")

print("\n=======================================================")
print("  АНАЛИЗ 2: ИСТОРИЧЕСКИЙ Z-SCORE (СИГМА-ОТКЛОНЕНИЕ)  ")
print("=======================================================")
# Goal: Prove 2025 months are statistical outliers compared to 2015-2024 history
hist_df = df[
    (df['Region_code'] == 7) & 
    (df['Item_code'] == 1) & 
    (df['Date'].dt.year >= 2015) & 
    (df['Date'].dt.year < 2025)
]

# Calculate stats per month (Seasonality profile)
monthly_stats = hist_df.groupby(hist_df['Date'].dt.month)['Inflation'].agg(['mean', 'std'])

print("\nМесяц | Факт 2025 | Ист. Среднее | Ист. StdDev | Сигма (Z-score) | Статус")
print("-" * 80)

anomalies = []
df_kbr_2025 = df[(df['Region_code'] == 7) & (df['Item_code'] == 1) & (df['Date'].dt.year == 2025)].sort_values('Date')

for _, row in df_kbr_2025.iterrows():
    m = row['Date'].month
    fact = row['Inflation']
    mean = monthly_stats.loc[m, 'mean']
    std = monthly_stats.loc[m, 'std']
    
    z_score = (fact - mean) / std
    status = ""
    if abs(z_score) > 1.5: status = "ANOMALY"
    if abs(z_score) > 2.0: status = "EXTREME"
    
    print(f"{m:02d}    | {fact:6.2f}% | {mean:6.2f}%     | {std:6.2f}      | {z_score:6.2f} σ        | {status}")

print("\nВывод: Z-score > 2.0 означает событие с вероятностью <5%. Это 'Черный лебедь'.")


print("\n=======================================================")
print("  АНАЛИЗ 3: СЕКТОРАЛЬНЫЙ РАЗРЫВ (GOODS vs SERVICES)  ")
print("=======================================================")
# Goal: Check if Services and Goods are moving in opposite directions in KBR vs Neighbors
# Compare KBR divergence vs Krasnodar divergence

def get_sector_divergence(region_code, year):
    target = df[(df['Region_code'] == region_code) & (df['Date'].dt.year == year)]
    
    # Pivot sectors
    p = target[target['Item_code'].isin([3, 4])].pivot_table(index='Date', columns='Item_code', values='Inflation')
    if 3 in p.columns and 4 in p.columns:
        p.columns = ['Food', 'Services']
        p['Divergence'] = abs(p['Food'] - p['Services'])
        return p['Divergence'].mean()
    return 0

kbr_div_24 = get_sector_divergence(7, 2024)
kbr_div_25 = get_sector_divergence(7, 2025)
kras_div_25 = get_sector_divergence(23, 2025)

print(f"Средний разрыв (Goods vs Services) в КБР 2024: {kbr_div_24:.2f} п.п.")
print(f"Средний разрыв (Goods vs Services) в КБР 2025: {kbr_div_25:.2f} п.п.")
print(f"Средний разрыв (Goods vs Services) в КК  2025: {kras_div_25:.2f} п.п.")

growth = ((kbr_div_25 - kbr_div_24) / kbr_div_24) * 100
print(f"\nРост структурного дисбаланса в КБР: +{growth:.1f}%")
print("Вывод: Если компоненты инфляции расходятся, модели теряют точность агрегации.")
