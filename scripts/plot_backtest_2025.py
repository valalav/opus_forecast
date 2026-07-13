import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
res_df = pd.read_csv('docs/rolling_backtest_2025.csv')
w_df = pd.read_csv('docs/model_weights_2025.csv')

# Setup Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'height_ratios': [1, 1]})

# Plot 1: Forecast vs Fact
ax1.plot(res_df['Date'], res_df['Fact'], 'o-', label='Факт (Fact)', color='black', linewidth=2)
ax1.plot(res_df['Date'], res_df['Ensemble_Pred'], 's--', label='Прогноз Ансамбля (Ensemble)', color='red', linewidth=2)

# Highlight anomalies
anomalies = res_df[abs(res_df['Ensemble_Error']) > 0.5]
ax1.scatter(anomalies['Date'], anomalies['Fact'], s=200, facecolors='none', edgecolors='red', linewidth=2, label='Аномалия (>0.5 п.п.)')

ax1.set_title('Роллинг-бэктест 2025: Факт vs Прогноз (Все модели)', fontsize=14)
ax1.set_ylabel('Инфляция MoM (%)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Annotate Best Model
for idx, row in res_df.iterrows():
    ax1.annotate(row['Best_Model'], (row['Date'], row['Fact']), xytext=(0, 10), textcoords='offset points', fontsize=8, alpha=0.7)

# Plot 2: Weights Evolution
# Melt weights
w_melt = w_df.melt('Date', var_name='Model', value_name='Weight')
# Remove _weight suffix
w_melt['Model'] = w_melt['Model'].str.replace('_weight', '')

sns.lineplot(data=w_melt, x='Date', y='Weight', hue='Model', ax=ax2, linewidth=2)
ax2.set_title('Динамика весов моделей (Адаптация к ошибкам)', fontsize=14)
ax2.set_ylabel('Вес в ансамбле')
ax2.set_ylim(0, 1.0)
ax2.grid(True, alpha=0.3)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('docs/rolling_backtest_2025_plot.png')
print("Plot saved to docs/rolling_backtest_2025_plot.png")
