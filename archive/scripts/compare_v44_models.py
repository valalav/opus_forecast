"""
Сравнение всех моделей v4.4 (включая новые улучшения)
=====================================================

Модели для сравнения:
1. Ridge baseline (v2.4)
2. Ridge Extended v2 (v4.3) - лучшая текущая
3. Ridge v3 (dummy для выбросов вместо исключения)
4. ElasticNet (L1+L2)
5. Huber (робастная)
6. Quantile Ridge (медиана + CI)
7. Constrained Components (компонентный прогноз)
8. NGBoost (probabilistic boosting)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sirena import DataLoader
from sirena.models import ModelRegistry

print("=" * 70)
print("СРАВНЕНИЕ МОДЕЛЕЙ v4.4")
print("=" * 70)

# Загрузка данных
loader = DataLoader()
df = loader.load_monthly_kbr()
print(f"\nДанные: {len(df)} месяцев")
print(f"Период: {df.index.min().strftime('%Y-%m')} — {df.index.max().strftime('%Y-%m')}")

# Модели для сравнения
models_to_test = [
    ('Ridge baseline', 'ridge'),
    ('Ridge Extended v2', 'ridge_extended'),
    ('Ridge v3 (dummy)', 'ridge_v3'),
    ('ElasticNet', 'elasticnet'),
    ('Huber', 'huber'),
    ('Quantile Ridge', 'quantile_ridge'),
    ('Constrained Comp', 'constrained_components'),
    ('NGBoost', 'ngboost'),
    ('ETS', 'ets'),
]

# Бэктест
all_results = {}
start_date = '2023-01-01'

print(f"\nБэктест с {start_date}...")
print("-" * 50)

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

# Метрики
print("\n" + "=" * 70)
print("МЕТРИКИ КАЧЕСТВА")
print("=" * 70)

metrics = []
baseline_mae = None

for name, bt in all_results.items():
    mae = bt['error'].abs().mean()
    rmse = np.sqrt((bt['error'] ** 2).mean())
    me = bt['error'].mean()
    std = bt['error'].std()
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
        '±0.3 %': within_03,
        '±0.5 %': within_05,
        'vs Ridge %': vs_baseline
    })

metrics_df = pd.DataFrame(metrics).sort_values('MAE')
print("\n" + metrics_df.to_string(index=False, float_format='%.4f'))

# Доверительные интервалы (для моделей с CI)
print("\n" + "=" * 70)
print("КАЧЕСТВО ДОВЕРИТЕЛЬНЫХ ИНТЕРВАЛОВ")
print("=" * 70)

ci_models = ['Quantile Ridge', 'NGBoost']
for name in ci_models:
    if name in all_results:
        bt = all_results[name]
        if 'in_ci' in bt.columns:
            coverage = bt['in_ci'].mean() * 100
            avg_width = bt['ci_width'].mean()
            print(f"{name}: покрытие 90% CI = {coverage:.1f}%, средняя ширина = {avg_width:.3f}")

# Сохраняем результаты
metrics_df.to_csv('model_comparison_v44.csv', index=False)
print("\nРезультаты сохранены: model_comparison_v44.csv")

# Лучшая модель
best_model = metrics_df.iloc[0]['Модель']
best_mae = metrics_df.iloc[0]['MAE']
print(f"\n🏆 Лучшая модель: {best_model} (MAE={best_mae:.4f})")

print("\n" + "=" * 70)
print("ГОТОВО!")
print("=" * 70)
