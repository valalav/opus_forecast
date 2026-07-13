import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/all_regions_indices.csv')

regions = {
    7: 'KBR',
    23: 'Krasnodar',
    61: 'Rostov',
    26: 'Stavropol',
    1: 'Adygea',
    5: 'Dagestan',
    6: 'Ingushetia',
    9: 'Karachay-Cherkessia',
    15: 'North Ossetia',
    20: 'Chechnya',
    92: 'Sevastopol',
    30: 'Astrakhan',
    34: 'Volgograd'
}

items = {
    1: 'All Goods',
    42: 'Fuel',
    33: 'Vegetables'
}

df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Date'].dt.year.isin([2024, 2025])]
df = df[df['Region_code'].isin(regions.keys())]
df = df[df['Item_code'].isin(items.keys())]
df['Inflation'] = df['MoM'] - 100

# Manually append November 2025 data from Task file
# KBR (7): 0.15
# Krasnodar (23): 0.34
# Rostov (61): 0.26
# Stavropol (26): 0.43
# Average South: 0.45 (Not adding average to region list, but noting)
nov_data = [
    {'Date': pd.Timestamp('2025-11-01'), 'Region_code': 7, 'Item_code': 1, 'Inflation': 0.15},
    {'Date': pd.Timestamp('2025-11-01'), 'Region_code': 23, 'Item_code': 1, 'Inflation': 0.34},
    {'Date': pd.Timestamp('2025-11-01'), 'Region_code': 61, 'Item_code': 1, 'Inflation': 0.26},
    {'Date': pd.Timestamp('2025-11-01'), 'Region_code': 26, 'Item_code': 1, 'Inflation': 0.43},
]
# Append to df
for row in nov_data:
    # Check if row exists, if not append. (It shouldn't exist based on previous check)
    if df[(df['Date'] == row['Date']) & (df['Region_code'] == row['Region_code']) & (df['Item_code'] == row['Item_code'])].empty:
        # Create a DataFrame for the new row and concatenate
        new_row_df = pd.DataFrame([row])
        # We need to fill other columns like MoM to avoid NaNs if important, but for 'Inflation' analysis it's fine.
        new_row_df['MoM'] = new_row_df['Inflation'] + 100
        df = pd.concat([df, new_row_df], ignore_index=True)

# 1. Main Analysis: KBR vs Krasnodar All Goods (2024-2025)
print("--- KBR vs Krasnodar All Goods (2024-2025) ---")
main_view = df[(df['Region_code'].isin([7, 23])) & (df['Item_code'] == 1)].pivot(index='Date', columns='Region_code', values='Inflation')
main_view.columns = ['KBR', 'Krasnodar']
main_view['Spread'] = main_view['KBR'] - main_view['Krasnodar']
print(main_view.tail(15))
main_view.to_csv('kbr_vs_krasnodar_inflation.csv')

# 2. Volatility Ranking (Jan-Nov 2025)
print("\n--- Volatility Ranking (SD) Jan-Nov 2025 ---")
stats_2025 = df[(df['Item_code'] == 1) & (df['Date'].dt.year == 2025) & (df['Date'].dt.month <= 11)]
volatility = stats_2025.groupby('Region_code')['Inflation'].std().sort_values(ascending=False)
vol_df = pd.DataFrame(volatility)
vol_df['Region_Name'] = vol_df.index.map(regions)
print(vol_df)
vol_df.to_csv('region_volatility_2025.csv')

# 3. Components Check for KBR
print("\n--- KBR Components Check (Fuel, Veg) 2025 ---")
# Check if we have data for components for KBR in 2025
comp_df = df[(df['Region_code'] == 7) & (df['Item_code'].isin([42, 33])) & (df['Date'].dt.year == 2025)]
if not comp_df.empty:
    kbr_components = comp_df.pivot(index='Date', columns='Item_code', values='Inflation')
    # Map column names if they exist
    col_map = {33: 'Vegetables', 42: 'Fuel'}
    kbr_components.columns = [col_map.get(c, c) for c in kbr_components.columns]
    print(kbr_components)
    kbr_components.to_csv('kbr_components_2025.csv')
else:
    print("No component data found for KBR 2025")
