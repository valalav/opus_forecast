"""
Сравнение моделей v4.2 с baseline Ridge.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sirena import DataLoader
from sirena.models import ModelRegistry

print("=" * 60)
print("СРАВНЕНИЕ МОДЕЛЕЙ v4.2")
print("=" * 60)

# Загружаем данные
loader = DataLoader()
df = loader.load_monthly_kbr()
print(f"\nДанные: {len(df)} месяцев")

# Модели для сравнения
models_to_test = [
    ('Ridge (baseline)', 'ridge'),
    ('Ridge Extended', 'ridge_extended'),
    ('Bayesian Ridge', 'bayesian_ridge'),
    ('ETS', 'ets'),
    ('BVAR', 'bvar'),
]

# Добавляем CatBoost если доступен
try:
    from sirena.models import CATBOOST_AVAILABLE
    if CATBOOST_AVAILABLE:
        models_to_test.append(('CatBoost', 'catboost'))
except:
    pass

results = {}

for name, model_id in models_to_test:
    try:
        print(f"\n{name}...", end=" ", flush=True)
        model = ModelRegistry.get(model_id)
        bt = model.backtest(df, start_date='2023-01-01')
        mae = bt['error'].abs().mean()
        results[name] = {
            'MAE': mae,
            'periods': len(bt)
        }
        print(f"MAE: {mae:.4f}")
    except Exception as e:
        print(f"Ошибка: {e}")

# Сводка
if results:
    print("\n" + "=" * 60)
    print("СВОДНАЯ ТАБЛИЦА")
    print("=" * 60)

    baseline_mae = results.get('Ridge (baseline)', {}).get('MAE', 1.0)
    print(f"\n{'Модель':<20} {'MAE':>8} {'vs Ridge':>10}")
    print("-" * 40)
    for name, data in sorted(results.items(), key=lambda x: x[1]['MAE']):
        diff = ((data['MAE'] / baseline_mae) - 1) * 100
        sign = '+' if diff > 0 else ''
        print(f"{name:<20} {data['MAE']:>8.4f} {sign}{diff:>9.1f}%")

    # Лучшая модель
    best = min(results.items(), key=lambda x: x[1]['MAE'])
    print(f"\nЛучшая модель: {best[0]} (MAE: {best[1]['MAE']:.4f})")
