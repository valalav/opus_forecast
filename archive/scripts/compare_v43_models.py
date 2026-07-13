"""
Сравнение моделей v4.3 с baseline Ridge.

v4.3 новые модели:
- Ridge Extended v2 (с sample weights и расширенными календарными признаками)
- ElasticNet (L1+L2 с автоматическим feature selection)
- Huber (робастная к выбросам без исключения лет)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sirena import DataLoader
from sirena.models import ModelRegistry

print("=" * 60)
print("СРАВНЕНИЕ МОДЕЛЕЙ v4.3")
print("=" * 60)

# Загружаем данные
loader = DataLoader()
df = loader.load_monthly_kbr()
print(f"\nДанные: {len(df)} месяцев")
print(f"Период: {df.index.min().strftime('%Y-%m')} — {df.index.max().strftime('%Y-%m')}")

# Модели для сравнения
models_to_test = [
    ('Ridge (baseline)', 'ridge'),
    ('Ridge Extended v2', 'ridge_extended'),
    ('Bayesian Ridge', 'bayesian_ridge'),
    ('ElasticNet', 'elasticnet'),
    ('Huber', 'huber'),
    ('ETS', 'ets'),
]

results = {}

print("\n" + "-" * 40)
print("Бэктест на 2023-2025...")
print("-" * 40)

for name, model_id in models_to_test:
    try:
        print(f"\n{name}...", end=" ", flush=True)
        model = ModelRegistry.get(model_id)
        bt = model.backtest(df, start_date='2023-01-01')

        if len(bt) == 0:
            print("Нет данных")
            continue

        mae = bt['error'].abs().mean()
        rmse = np.sqrt((bt['error'] ** 2).mean())

        results[name] = {
            'MAE': mae,
            'RMSE': rmse,
            'periods': len(bt)
        }
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    except Exception as e:
        print(f"Ошибка: {e}")

# Сводка
if results:
    print("\n" + "=" * 60)
    print("СВОДНАЯ ТАБЛИЦА (отсортировано по MAE)")
    print("=" * 60)

    baseline_mae = results.get('Ridge (baseline)', {}).get('MAE', 1.0)

    print(f"\n{'Модель':<25} {'MAE':>8} {'RMSE':>8} {'vs Ridge':>10}")
    print("-" * 55)

    for name, data in sorted(results.items(), key=lambda x: x[1]['MAE']):
        diff = ((data['MAE'] / baseline_mae) - 1) * 100
        sign = '+' if diff > 0 else ''
        print(f"{name:<25} {data['MAE']:>8.4f} {data['RMSE']:>8.4f} {sign}{diff:>9.1f}%")

    # Лучшая модель
    best = min(results.items(), key=lambda x: x[1]['MAE'])
    print(f"\n✓ Лучшая модель: {best[0]} (MAE: {best[1]['MAE']:.4f})")

    # Улучшение vs baseline
    improvement = (baseline_mae - best[1]['MAE']) / baseline_mae * 100
    if improvement > 0:
        print(f"✓ Улучшение vs Ridge baseline: {improvement:.1f}%")

# Дополнительная информация о новых моделях
print("\n" + "=" * 60)
print("ИНФОРМАЦИЯ О НОВЫХ МОДЕЛЯХ v4.3")
print("=" * 60)

try:
    # ElasticNet: отобранные признаки
    print("\n[ElasticNet] Автоматический feature selection:")
    en = ModelRegistry.get('elasticnet')
    en.fit(df)
    selected = en.get_selected_features()
    print(f"  Отобрано {len(selected)} из {len(en._features)} признаков")
    print(f"  Best alpha: {en._best_alpha:.4f}")
    print(f"  Best l1_ratio: {en._best_l1_ratio:.2f}")
    # Топ-5 признаков
    importance = en.get_feature_importance()
    top5 = importance[importance['is_selected']].head(5)
    print("  Топ-5 признаков:")
    for _, row in top5.iterrows():
        print(f"    - {row['feature']}: {row['coefficient']:.4f}")
except Exception as e:
    print(f"  Ошибка: {e}")

try:
    # Huber: информация о выбросах
    print("\n[Huber] Информация о выбросах:")
    hub = ModelRegistry.get('huber')
    hub.fit(df)
    info = hub.get_model_info()
    print(f"  Выбросов обнаружено: {info['outliers_detected']}")
    print(f"  Scale: {info['scale']:.4f}")
    print(f"  Epsilon: {info['epsilon']}")
except Exception as e:
    print(f"  Ошибка: {e}")

try:
    # Ridge Extended v2: информация о новых признаках
    print("\n[Ridge Extended v2] Новые признаки:")
    rex = ModelRegistry.get('ridge_extended')
    print(f"  Исключаемые годы: {rex.OUTLIER_YEARS}")
    print(f"  Новые календарные признаки: is_tariff_month, is_q1, is_summer")
    rex.fit(df)
    importance = rex.get_feature_importance()
    new_features = importance[importance['is_new']]
    print(f"  Топ новых признаков:")
    for _, row in new_features.head(5).iterrows():
        print(f"    - {row['feature']}: {row['coefficient']:.4f}")
except Exception as e:
    print(f"  Ошибка: {e}")

print("\n" + "=" * 60)
print("Готово!")
