import pandas as pd
import numpy as np

# Load Data
df = pd.read_csv('data/all_regions_indices.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Inflation'] = df['MoM'] - 100

skfo_codes = [5, 6, 7, 9, 15, 20, 26]

# Calculate SKFO Avg History
skfo_df = df[(df['Region_code'].isin(skfo_codes)) & (df['Item_code'] == 1)].copy()
skfo_hist = skfo_df.groupby('Date')['Inflation'].mean().reset_index()
skfo_hist['Month'] = skfo_hist['Date'].dt.month
skfo_hist['Year'] = skfo_hist['Date'].dt.year

# Check Nov 2025 (we need to assume the 0.45 avg from task or calculate if we added it? 
# I didn't save the "added" nov data to the CSV, only to the dataframe in memory in previous script.
# So I need to manually check Nov history excluding 2025 to see where 0.45 would fall.

print("Исторические минимумы СКФО для НОЯБРЯ (2010-2024):")
nov_skfo = skfo_hist[skfo_hist['Month'] == 11].sort_values('Inflation')
print(nov_skfo[['Year', 'Inflation']].head(5))

# Check Aug 2025 (-0.52 approx from previous analysis) vs History
print("\nИсторические минимумы СКФО для АВГУСТА (2010-2024):")
aug_skfo = skfo_hist[skfo_hist['Month'] == 8].sort_values('Inflation')
print(aug_skfo[['Year', 'Inflation']].head(5))
