import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
res_df = pd.read_csv('docs/rolling_backtest_2025.csv')
w_df = pd.read_csv('docs/model_weights_2025.csv')

# --- PLOT 1: Forecast vs Fact ---
plt.figure(figsize=(12, 6))
plt.plot(res_df['Date'], res_df['Fact'], 'o-', label='Факт (Fact)', color='black', linewidth=3)
plt.plot(res_df['Date'], res_df['Ensemble_Pred'], 's--', label='Прогноз Ансамбля (Ensemble)', color='red', linewidth=2)

# Highlight anomalies
anomalies = res_df[abs(res_df['Ensemble_Error']) > 0.5]
plt.scatter(anomalies['Date'], anomalies['Fact'], s=200, facecolors='none', edgecolors='red', linewidth=2, label='Аномалия (>0.5 п.п.)')

plt.title('Роллинг-бэктест 2025: Факт vs Прогноз (Все модели)', fontsize=14)
plt.ylabel('Инфляция MoM (%)')
plt.grid(True, alpha=0.3)
plt.legend()

# Annotate Best Model
for idx, row in res_df.iterrows():
    plt.annotate(
        f"{row['Best_Model']}", 
        (row['Date'], row['Fact']), 
        xytext=(0, 15), 
        textcoords='offset points', 
        fontsize=9, 
        alpha=0.8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

plt.tight_layout()
plt.savefig('docs/backtest_forecast_vs_fact.png')
print("Plot 1 saved to docs/backtest_forecast_vs_fact.png")
plt.close()

# --- PLOT 2: Weights Evolution ---
plt.figure(figsize=(12, 6))

# Melt weights
w_melt = w_df.melt('Date', var_name='Model', value_name='Weight')
w_melt['Model'] = w_melt['Model'].str.replace('_weight', '')

# Plot Area Chart (Stacked) or Line Chart? Line chart is clearer for crossovers.
# Stacked is good for composition. Let's do Stacked Area.
# Pivot for stackplot
w_pivot = w_melt.pivot(index='Date', columns='Model', values='Weight')
dates = pd.to_datetime(w_pivot.index)

plt.stackplot(dates, w_pivot.T, labels=w_pivot.columns, alpha=0.8)

plt.title('Динамика весов моделей (Адаптация к ошибкам)', fontsize=14)
plt.ylabel('Вес в ансамбле')
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('docs/backtest_weights_evolution.png')
print("Plot 2 saved to docs/backtest_weights_evolution.png")
plt.close()
