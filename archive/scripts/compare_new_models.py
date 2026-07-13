"""
Сравнение новых моделей с baseline (Ridge).
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sirena import DataLoader
from sirena.models import ModelRegistry

print("=" * 60)
print("СРАВНЕНИЕ МОДЕЛЕЙ: Baseline vs Новые")
print("=" * 60)

# Загружаем данные
loader = DataLoader()
df = loader.load_monthly_kbr()
print(f"\nДанные: {len(df)} месяцев")

# Модели для сравнения
models_to_test = {
    'Ridge (baseline)': 'ridge',
    'Hierarchical': 'hierarchical',
    'Stacking (ridge+ets)': None,  # Особый случай
}

results = {}

# 1. Ridge baseline
print("\n1. Ridge (baseline)...")
ridge = ModelRegistry.get('ridge')
bt_ridge = ridge.backtest(df, start_date='2023-01-01')
results['Ridge'] = {
    'MAE': bt_ridge['error'].abs().mean(),
    'periods': len(bt_ridge)
}
print(f"   MAE: {results['Ridge']['MAE']:.4f}")

# 2. Hierarchical
print("\n2. Hierarchical (MinTrace)...")
hier = ModelRegistry.get('hierarchical')
bt_hier = hier.backtest(df, start_date='2023-01-01')
results['Hierarchical'] = {
    'MAE': bt_hier['error'].abs().mean(),
    'periods': len(bt_hier)
}
print(f"   MAE: {results['Hierarchical']['MAE']:.4f}")

# 3. Stacking (быстрая версия)
print("\n3. Stacking (ridge+ets)...")
from sirena.models import StackingForecaster
stacking = StackingForecaster(base_models=['ridge', 'ets'], oof_start='2022-01-01')
bt_stack = stacking.backtest(df, start_date='2023-01-01')
results['Stacking'] = {
    'MAE': bt_stack['error'].abs().mean(),
    'periods': len(bt_stack)
}
print(f"   MAE: {results['Stacking']['MAE']:.4f}")

# Сводка
print("\n" + "=" * 60)
print("СВОДНАЯ ТАБЛИЦА")
print("=" * 60)

baseline_mae = results['Ridge']['MAE']
print(f"\n{'Модель':<20} {'MAE':>8} {'vs Ridge':>10}")
print("-" * 40)
for name, data in results.items():
    diff = ((data['MAE'] / baseline_mae) - 1) * 100
    sign = '+' if diff > 0 else ''
    print(f"{name:<20} {data['MAE']:>8.4f} {sign}{diff:>9.1f}%")

# Лучшая модель
best = min(results.items(), key=lambda x: x[1]['MAE'])
print(f"\nЛучшая модель: {best[0]} (MAE: {best[1]['MAE']:.4f})")
