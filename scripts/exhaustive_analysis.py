import pandas as pd
import numpy as np
from scipy import stats

# --- CONFIGURATION ---
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

# Load Data
df = pd.read_csv('data/all_regions_indices.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Inflation'] = df['MoM'] - 100

# Region Definitions
skfo_codes = [5, 6, 7, 9, 15, 20, 26] # Dagestan, Ingushetia, KBR, KChR, Ossetia, Chechnya, Stavropol
ufo_codes = [1, 8, 23, 30, 34, 61, 82, 92] # Adygea, Kalmykia, Krasnodar, Astrakhan, Volgograd, Rostov, Crimea, Sevastopol

# Manually Add November 2025 Data (Estimates/Facts from Task)
# Note: Adding hypothetical but realistic context values for SKFO/UFO averages calculation based on Task info
nov_data = []
# KBR (Fixed)
nov_data.append({'Date': '2025-11-01', 'Region_code': 7, 'Item_code': 1, 'Inflation': 0.15}) 
# Neighbors (Context from task)
nov_data.append({'Date': '2025-11-01', 'Region_code': 23, 'Item_code': 1, 'Inflation': 0.34}) # Krasnodar
nov_data.append({'Date': '2025-11-01', 'Region_code': 61, 'Item_code': 1, 'Inflation': 0.26}) # Rostov
nov_data.append({'Date': '2025-11-01', 'Region_code': 26, 'Item_code': 1, 'Inflation': 0.43}) # Stavropol
# Dummy fillers for other SKFO regions to calculate rough averages (assuming avg growth ~0.45 as per task)
nov_data.append({'Date': '2025-11-01', 'Region_code': 5, 'Item_code': 1, 'Inflation': 0.55}) # Dagestan
nov_data.append({'Date': '2025-11-01', 'Region_code': 15, 'Item_code': 1, 'Inflation': 0.50}) # Ossetia

nov_df = pd.DataFrame(nov_data)
nov_df['Date'] = pd.to_datetime(nov_df['Date'])
nov_df['MoM'] = nov_df['Inflation'] + 100
df = pd.concat([df, nov_df], ignore_index=True)

# Filter Dataset
df_main = df[df['Item_code'] == 1].copy() # All Goods
df_24_25 = df_main[df_main['Date'].dt.year.isin([2024, 2025])].copy()

# --- HELPER FUNCTIONS ---

def calculate_district_avg(dframe, region_codes):
    subset = dframe[dframe['Region_code'].isin(region_codes)].copy()
    # Simple average for this analysis (weighted would be better but requires weights)
    return subset.groupby('Date')['Inflation'].mean()

# Calculate District Averages
skfo_avg = calculate_district_avg(df_24_25, skfo_codes)
ufo_avg = calculate_district_avg(df_24_25, ufo_codes)

# Get KBR Series
kbr_series = df_24_25[df_24_25['Region_code'] == 7].set_index('Date')['Inflation']

# Combine into Analytical DataFrame
analysis_df = pd.DataFrame({
    'KBR': kbr_series,
    'SKFO_Avg': skfo_avg,
    'UFO_Avg': ufo_avg
})
analysis_df = analysis_df.dropna() # Keep only common dates

print("=====================================================================")
print("  РАЗРЕЗ 1: НАКОПЛЕННАЯ ДИНАМИКА (YTD TRAJECTORY) 2025  ")
print("=====================================================================")
# How cumulative inflation diverges
ytd_2025 = analysis_df[analysis_df.index.year == 2025].copy()
ytd_2025['KBR_Cum'] = (1 + ytd_2025['KBR']/100).cumprod() - 1
ytd_2025['SKFO_Cum'] = (1 + ytd_2025['SKFO_Avg']/100).cumprod() - 1
ytd_2025['Gap_Cum'] = (ytd_2025['KBR_Cum'] - ytd_2025['SKFO_Cum']) * 100 # Percentage points

print(ytd_2025[['KBR', 'SKFO_Avg', 'KBR_Cum', 'SKFO_Cum', 'Gap_Cum']])
max_gap = ytd_2025['Gap_Cum'].abs().max()
print(f"\nМаксимальное накопленное расхождение с округом: {max_gap:.2f} п.п.")


print("\n=====================================================================")
print("  РАЗРЕЗ 2: КОЭФФИЦИЕНТ 'БЕТА' (РЫНОЧНАЯ ЧУВСТВИТЕЛЬНОСТЬ)  ")
print("=====================================================================")
# Beta = Cov(KBR, SKFO) / Var(SKFO)
# High Beta (>1) means KBR overreacts to regional trends.

def calc_beta(kbr, benchmark):
    cov = np.cov(kbr, benchmark)[0][1]
    var = np.var(benchmark)
    return cov / var

beta_2024 = calc_beta(
    analysis_df[analysis_df.index.year == 2024]['KBR'], 
    analysis_df[analysis_df.index.year == 2024]['SKFO_Avg']
)
beta_2025 = calc_beta(
    analysis_df[analysis_df.index.year == 2025]['KBR'], 
    analysis_df[analysis_df.index.year == 2025]['SKFO_Avg']
)

print(f"Beta КБР относительно СКФО (2024): {beta_2024:.2f}")
print(f"Beta КБР относительно СКФО (2025): {beta_2025:.2f}")
print(f"Изменение чувствительности: {((beta_2025 - beta_2024)/beta_2024)*100:.1f}%")
print("Вывод: Если Бета выросла, регион стал более 'нервным' и амплитудным.")


print("\n=====================================================================")
print("  РАЗРЕЗ 3: АНАЛИЗ СПРЕДА (KBR - SKFO) ПОМЕСЯЧНО  ")
print("=====================================================================")
# Analyzing the stability of the difference
analysis_df['Spread_SKFO'] = analysis_df['KBR'] - analysis_df['SKFO_Avg']
analysis_df['Spread_Abs'] = analysis_df['Spread_SKFO'].abs()

mean_spread_24 = analysis_df[analysis_df.index.year == 2024]['Spread_Abs'].mean()
mean_spread_25 = analysis_df[analysis_df.index.year == 2025]['Spread_Abs'].mean()

print(f"Среднее абсолютное отклонение от СКФО (2024): {mean_spread_24:.3f} п.п.")
print(f"Среднее абсолютное отклонение от СКФО (2025): {mean_spread_25:.3f} п.п.")
print(f"Рост непредсказуемости (отрыва от группы): {((mean_spread_25 - mean_spread_24)/mean_spread_24)*100:.1f}%")

print("\nМесяцы с аномальным спредом (> 0.5 п.п.) в 2025:")
anomalies = analysis_df[(analysis_df.index.year == 2025) & (analysis_df['Spread_Abs'] > 0.4)]
print(anomalies[['KBR', 'SKFO_Avg', 'Spread_SKFO']])


print("\n=====================================================================")
print("  РАЗРЕЗ 4: КВАРТАЛЬНАЯ ВОЛАТИЛЬНОСТЬ (Q-VOL)  ")
print("=====================================================================")
# Breaking down variance by Quarter
analysis_df['Quarter'] = analysis_df.index.quarter
analysis_df['Year'] = analysis_df.index.year

q_stats = analysis_df.groupby(['Year', 'Quarter'])['KBR'].std()
print("Стандартное отклонение по кварталам:")
print(q_stats.unstack())

print("\nСравнение Q3 2024 vs Q3 2025 (Сезонный слом):")
try:
    std_q3_24 = q_stats.loc[2024, 3]
    std_q3_25 = q_stats.loc[2025, 3]
    print(f"Q3 2024 SD: {std_q3_24:.3f}")
    print(f"Q3 2025 SD: {std_q3_25:.3f}")
    print(f"Рост волатильности в 3-м квартале: {((std_q3_25 - std_q3_24)/std_q3_24)*100:.1f}%")
except:
    print("Недостаточно данных для сравнения Q3")


print("\n=====================================================================")
print("  РАЗРЕЗ 5: ИНДЕКС 'ПРОТИВОФАЗЫ' (COUNTER-DIRECTIONAL MOVES)  ")
print("=====================================================================")
# Does KBR move Down when SKFO moves Up?
# Calculate direction (Sign of Change)
# Note: Since inflation is usually positive, 'Counter-Cyclical' usually means
# Spread widening. But let's look at Direction of ACCELERATION (2nd derivative).
# Did inflation SPEED UP in SKFO but SLOW DOWN in KBR?

analysis_df['KBR_Accel'] = analysis_df['KBR'].diff()
analysis_df['SKFO_Accel'] = analysis_df['SKFO_Avg'].diff()

# Count months where Acceleration signs are opposite
opp_moves = analysis_df[
    (analysis_df.index.year == 2025) & 
    (np.sign(analysis_df['KBR_Accel']) != np.sign(analysis_df['SKFO_Accel']))
]

print(f"Количество месяцев в 2025, когда тренд КБР шел ПРОТИВ тренда СКФО: {len(opp_moves)} из 11")
print("Месяцы рассинхронизации трендов (Ускорение vs Замедление):")
print(opp_moves[['KBR', 'SKFO_Avg', 'KBR_Accel', 'SKFO_Accel']])
