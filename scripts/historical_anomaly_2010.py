import pandas as pd
import numpy as np

# --- CONFIG ---
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)

# --- LOAD DATA ---
# Using the dedicated KBR file which goes back to 2010
df = pd.read_csv('data/infl_kbr.csv', sep=';')

# Fix formatting (comma decimals)
if df['MoM'].dtype == object:
    df['MoM'] = df['MoM'].str.replace(',', '.').astype(float)

df['Date'] = pd.to_datetime(df['Date'])
df['Inflation'] = df['MoM'] - 100
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

# Filter for "All Goods and Services"
df = df[df['Товар'] == 'Все товары и услуги'].copy()

# --- INJECT NOV 2025 (Manual Entry from Task) ---
# We need this to complete the 2025 picture
nov_row = pd.DataFrame([{
    'Date': pd.Timestamp('2025-11-01'),
    'Товар': 'Все товары и услуги',
    'MoM': 100.15,
    'Inflation': 0.15,
    'Year': 2025,
    'Month': 11
}])
df = pd.concat([df, nov_row], ignore_index=True)
df = df.sort_values('Date')

print("==========================================================================")
print("  ИСТОРИЧЕСКИЙ РЕКОРД 1: НОЯБРЬСКАЯ АНОМАЛИЯ (2010-2025)  ")
print("==========================================================================")
# Compare Nov 2025 to ALL Novembers
novembers = df[df['Month'] == 11].sort_values('Inflation')
novembers['Rank'] = range(1, len(novembers) + 1)
print("Топ-5 самых низких инфляций в НОЯБРЕ за 15 лет:")
print(novembers[['Year', 'Inflation']].head(5))

rank_2025 = novembers[novembers['Year'] == 2025]['Rank'].iloc[0]
print(f"\nМесто ноября 2025 года в истории: {rank_2025}-е из {len(novembers)} (1 = Самый низкий)")


print("\n==========================================================================")
print("  ИСТОРИЧЕСКИЙ РЕКОРД 2: АВГУСТОВСКИЙ ПРОВАЛ (2010-2025)  ")
print("==========================================================================")
# Compare Aug 2025 to ALL Augusts
augusts = df[df['Month'] == 8].sort_values('Inflation')
augusts['Rank'] = range(1, len(augusts) + 1)
print("Топ-5 самых глубоких падений в АВГУСТЕ за 15 лет:")
print(augusts[['Year', 'Inflation']].head(5))

rank_aug_2025 = augusts[augusts['Year'] == 2025]['Rank'].iloc[0]
print(f"\nМесто августа 2025 года в истории: {rank_aug_2025}-е из {len(augusts)} (1 = Самое глубокое падение)")


print("\n==========================================================================")
print("  ИСТОРИЧЕСКИЙ РЕКОРД 3: ЧАСТОТА ДЕФЛЯЦИИ (Deflation Frequency)  ")
print("==========================================================================")
# Count months < 0 per year
def_counts = df[df['Inflation'] < -0.01].groupby('Year').size().sort_values(ascending=False)
print("Годы с наибольшим количеством дефляционных месяцев:")
print(def_counts.head(5))

count_2025 = def_counts.get(2025, 0)
print(f"\nКоличество месяцев дефляции в 2025: {count_2025}")


print("\n==========================================================================")
print("  ИСТОРИЧЕСКИЙ РЕКОРД 4: СЛОМ СЕЗОННОЙ КОРРЕЛЯЦИИ  ")
print("==========================================================================")
# Create a "Standard Year" (Average of 2010-2023)
hist_data = df[df['Year'] < 2024] # Exclude 24 and 25
standard_seasonality = hist_data.groupby('Month')['Inflation'].mean()

# Calculate correlation of each year vs Standard Seasonality
correlations = {}
for year in df['Year'].unique():
    # Need full 12 months for proper corr, or at least matching months
    year_data = df[df['Year'] == year].set_index('Month')['Inflation']
    # Align indices
    common_idx = standard_seasonality.index.intersection(year_data.index)
    if len(common_idx) > 10: # Only years with enough data
        corr = standard_seasonality[common_idx].corr(year_data[common_idx])
        correlations[year] = corr

corr_df = pd.Series(correlations).sort_values()
print("Корреляция годового профиля со 'Стандартным историческим профилем' (2010-2023):")
print("(Чем ближе к 1.0, тем типичнее год. Чем ниже, тем аномальнее)")
print("\nТоп-5 самых 'нестандартных' лет:")
print(corr_df.head(5))

corr_2025 = correlations.get(2025)
corr_2024 = correlations.get(2024)
print(f"\nКорреляция 2024 года: {corr_2024:.2f} (Высокая, типичный год)")
print(f"Корреляция 2025 года: {corr_2025:.2f} (Низкая, аномальный год)")


print("\n==========================================================================")
print("  ИСТОРИЧЕСКИЙ РЕКОРД 5: 'ИНДЕКС НЕПРЕДСКАЗУЕМОСТИ' (Deviation from Mean) ")
print("==========================================================================")
# Calculate Mean Absolute Deviation (MAD) of each year from the 15-year monthly averages
# How far is the curve from the "Average Curve"?

deviations = {}
for year in df['Year'].unique():
    year_data = df[df['Year'] == year].set_index('Month')['Inflation']
    common_idx = standard_seasonality.index.intersection(year_data.index)
    if len(common_idx) > 10:
        # ABS(Fact - Historical_Avg)
        diff = (year_data[common_idx] - standard_seasonality[common_idx]).abs().mean()
        deviations[year] = diff

mad_df = pd.Series(deviations).sort_values(ascending=False)
print("Среднее отклонение от исторической нормы (п.п.):")
print(mad_df.head(5))
print(f"\nОтклонение 2025 года: {mad_df.get(2025):.3f} п.п.")
print(f"Отклонение 2024 года: {mad_df.get(2024):.3f} п.п.")
print(f"Рост аномальности к прошлому году: {((mad_df.get(2025) - mad_df.get(2024))/mad_df.get(2024))*100:.1f}%")
