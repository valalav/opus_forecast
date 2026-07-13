"""
Подробный анализ бэктеста моделей v4.3 с графиками.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sirena import DataLoader
from sirena.models import ModelRegistry

# Настройки графиков
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True

print("=" * 60)
print("ПОДРОБНЫЙ АНАЛИЗ БЭКТЕСТА v4.3")
print("=" * 60)

# Загружаем данные
loader = DataLoader()
df = loader.load_monthly_kbr()
print(f"\nДанные: {len(df)} месяцев")
print(f"Период: {df.index.min().strftime('%Y-%m')} — {df.index.max().strftime('%Y-%m')}")

# Модели для сравнения
models_to_test = [
    ('Ridge baseline', 'ridge'),
    ('Ridge Extended v2', 'ridge_extended'),
    ('Bayesian Ridge', 'bayesian_ridge'),
    ('ElasticNet', 'elasticnet'),
    ('Huber', 'huber'),
    ('ETS', 'ets'),
]

# Собираем результаты бэктеста
all_results = {}
start_date = '2023-01-01'

print(f"\nБэктест с {start_date}...")
print("-" * 40)

for name, model_id in models_to_test:
    try:
        print(f"{name}...", end=" ", flush=True)
        model = ModelRegistry.get(model_id)
        bt = model.backtest(df, start_date=start_date)

        if len(bt) > 0:
            all_results[name] = bt
            mae = bt['error'].abs().mean()
            print(f"OK ({len(bt)} точек, MAE={mae:.4f})")
        else:
            print("нет данных")
    except Exception as e:
        print(f"ошибка: {e}")

# === ТАБЛИЦА МЕТРИК ===
print("\n" + "=" * 60)
print("МЕТРИКИ КАЧЕСТВА")
print("=" * 60)

metrics = []
baseline_mae = None

for name, bt in all_results.items():
    mae = bt['error'].abs().mean()
    rmse = np.sqrt((bt['error'] ** 2).mean())
    me = bt['error'].mean()  # Mean Error (bias)
    std = bt['error'].std()

    # % попаданий в ±0.3, ±0.5
    within_03 = (bt['error'].abs() <= 0.3).mean() * 100
    within_05 = (bt['error'].abs() <= 0.5).mean() * 100

    if name == 'Ridge baseline':
        baseline_mae = mae

    vs_baseline = ((mae / baseline_mae) - 1) * 100 if baseline_mae else 0

    metrics.append({
        'Модель': name,
        'MAE': mae,
        'RMSE': rmse,
        'ME (bias)': me,
        'Std': std,
        '±0.3': within_03,
        '±0.5': within_05,
        'vs Ridge': vs_baseline
    })

metrics_df = pd.DataFrame(metrics).sort_values('MAE')
print(metrics_df.to_string(index=False, float_format='%.4f'))

# === ГРАФИК 1: Прогноз vs Факт ===
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (name, bt) in enumerate(all_results.items()):
    if idx >= 6:
        break
    ax = axes[idx]

    bt_sorted = bt.sort_values('date')

    ax.plot(bt_sorted['date'], bt_sorted['actual'], 'b-', label='Факт', linewidth=2)
    ax.plot(bt_sorted['date'], bt_sorted['prediction'], 'r--', label='Прогноз', linewidth=1.5)

    mae = bt['error'].abs().mean()
    ax.set_title(f'{name} (MAE={mae:.3f})')
    ax.set_ylabel('MoM индекс')
    ax.legend(loc='upper right')
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Бэктест 2023-2025: Прогноз vs Факт', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('backtest_v43_forecast_vs_actual.png', dpi=150, bbox_inches='tight')
print("\nСохранено: backtest_v43_forecast_vs_actual.png")

# === ГРАФИК 2: Ошибки по времени ===
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (name, bt) in enumerate(all_results.items()):
    if idx >= 6:
        break
    ax = axes[idx]

    bt_sorted = bt.sort_values('date')

    colors = ['green' if e >= 0 else 'red' for e in bt_sorted['error']]
    ax.bar(bt_sorted['date'], bt_sorted['error'], color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5)
    ax.axhline(y=-0.3, color='orange', linestyle='--', alpha=0.5)

    mae = bt['error'].abs().mean()
    ax.set_title(f'{name} (MAE={mae:.3f})')
    ax.set_ylabel('Ошибка (факт - прогноз)')
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Бэктест 2023-2025: Ошибки по месяцам', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('backtest_v43_errors.png', dpi=150, bbox_inches='tight')
print("Сохранено: backtest_v43_errors.png")

# === ГРАФИК 3: Сравнение моделей ===
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 3.1 MAE по моделям
ax1 = axes[0, 0]
model_names = metrics_df['Модель'].values
mae_values = metrics_df['MAE'].values
colors = ['green' if v == min(mae_values) else 'steelblue' for v in mae_values]
bars = ax1.barh(model_names, mae_values, color=colors)
ax1.set_xlabel('MAE')
ax1.set_title('MAE по моделям (меньше = лучше)')
ax1.axvline(x=baseline_mae, color='red', linestyle='--', label=f'Ridge baseline ({baseline_mae:.3f})')
ax1.legend()
for bar, val in zip(bars, mae_values):
    ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')

# 3.2 % попаданий в ±0.5
ax2 = axes[0, 1]
within_05_values = metrics_df['±0.5'].values
colors = ['green' if v == max(within_05_values) else 'steelblue' for v in within_05_values]
bars = ax2.barh(model_names, within_05_values, color=colors)
ax2.set_xlabel('% попаданий')
ax2.set_title('Точность ±0.5 п.п. (больше = лучше)')
ax2.axvline(x=80, color='red', linestyle='--', label='Цель 80%')
ax2.legend()
for bar, val in zip(bars, within_05_values):
    ax2.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f}%', va='center')

# 3.3 Распределение ошибок (boxplot)
ax3 = axes[1, 0]
error_data = [all_results[name]['error'].values for name in model_names if name in all_results]
bp = ax3.boxplot(error_data, labels=[n[:12] for n in model_names], vert=True, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5)
ax3.axhline(y=-0.3, color='orange', linestyle='--', alpha=0.5)
ax3.set_ylabel('Ошибка')
ax3.set_title('Распределение ошибок')
ax3.tick_params(axis='x', rotation=45)

# 3.4 Bias (ME)
ax4 = axes[1, 1]
me_values = metrics_df['ME (bias)'].values
colors = ['green' if abs(v) < 0.05 else 'orange' if abs(v) < 0.1 else 'red' for v in me_values]
bars = ax4.barh(model_names, me_values, color=colors)
ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax4.set_xlabel('Mean Error (bias)')
ax4.set_title('Систематическая ошибка (bias)')
for bar, val in zip(bars, me_values):
    ax4.text(val + 0.01 if val >= 0 else val - 0.05, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center')

plt.suptitle('Сравнение моделей v4.3', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('backtest_v43_comparison.png', dpi=150, bbox_inches='tight')
print("Сохранено: backtest_v43_comparison.png")

# === ГРАФИК 4: Кумулятивная ошибка ===
fig, ax = plt.subplots(figsize=(14, 6))

for name, bt in all_results.items():
    bt_sorted = bt.sort_values('date')
    cumsum_error = bt_sorted['error'].abs().cumsum()
    ax.plot(bt_sorted['date'], cumsum_error, label=f'{name}', linewidth=2)

ax.set_xlabel('Дата')
ax.set_ylabel('Кумулятивная абс. ошибка')
ax.set_title('Накопленная ошибка по времени')
ax.legend(loc='upper left')
plt.tight_layout()
plt.savefig('backtest_v43_cumulative.png', dpi=150, bbox_inches='tight')
print("Сохранено: backtest_v43_cumulative.png")

# === ДЕТАЛЬНАЯ ТАБЛИЦА ПО МЕСЯЦАМ ===
print("\n" + "=" * 60)
print("ДЕТАЛИ ПО МЕСЯЦАМ (Ridge Extended v2)")
print("=" * 60)

if 'Ridge Extended v2' in all_results:
    bt = all_results['Ridge Extended v2'].sort_values('date')
    print(f"\n{'Дата':<12} {'Факт':>8} {'Прогноз':>8} {'Ошибка':>8} {'|Ошибка|':>8}")
    print("-" * 48)
    for _, row in bt.iterrows():
        date_str = row['date'].strftime('%Y-%m')
        print(f"{date_str:<12} {row['actual']:>8.2f} {row['prediction']:>8.2f} {row['error']:>8.3f} {abs(row['error']):>8.3f}")

    print("-" * 48)
    print(f"{'ИТОГО':<12} {'':>8} {'':>8} {bt['error'].mean():>8.3f} {bt['error'].abs().mean():>8.3f}")

# === АНАЛИЗ ПО МЕСЯЦАМ (СЕЗОННОСТЬ ОШИБОК) ===
print("\n" + "=" * 60)
print("СЕЗОННОСТЬ ОШИБОК (Ridge Extended v2)")
print("=" * 60)

if 'Ridge Extended v2' in all_results:
    bt = all_results['Ridge Extended v2'].copy()
    bt['month'] = bt['date'].dt.month

    monthly_stats = bt.groupby('month').agg({
        'error': ['mean', 'std', lambda x: x.abs().mean()]
    }).round(3)
    monthly_stats.columns = ['ME (bias)', 'Std', 'MAE']

    print("\n" + monthly_stats.to_string())

print("\n" + "=" * 60)
print("ГОТОВО!")
print("=" * 60)
print("\nГрафики сохранены:")
print("  - backtest_v43_forecast_vs_actual.png")
print("  - backtest_v43_errors.png")
print("  - backtest_v43_comparison.png")
print("  - backtest_v43_cumulative.png")
