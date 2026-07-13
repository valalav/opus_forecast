import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load
df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
if df['Date'].isna().any():
    df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date').sort_index()

# Fix types
if df['usd_nom_i'].dtype == object:
    df['usd_nom_i'] = df['usd_nom_i'].astype(str).str.replace(',', '.').astype(float)

usd_mom = df['usd_nom_i'] - 100
print(usd_mom.describe())

# ADF Test
result = adfuller(usd_mom.dropna())
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

# Save plot
plt.figure(figsize=(10, 6))
plt.plot(usd_mom)
plt.title("USD MoM Change (%)")
plt.savefig("usd_analysis.png")
