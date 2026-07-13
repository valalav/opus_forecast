"""
Сравнение трёх вариантов Ridge модели:
1. Ridge Raw - на сырых данных (baseline)
2. Ridge SA - на SA по 3 компонентам
3. Ridge SA Sub - на SA по 47 субкомпонентам
"""

import pandas as pd
import numpy as np
from sirena.models import RidgeForecaster, RidgeSAForecaster, RidgeSASubForecaster
from sirena import DataLoader

print("=" * 60)
print("СРАВНЕНИЕ ВАРИАНТОВ RIDGE МОДЕЛИ")
print("=" * 60)

# Загружаем сырые данные для Ridge Raw
print("\n1. Бэктест Ridge Raw (baseline)...")
loader = DataLoader()
raw_data = loader.load_monthly_kbr()

ridge_raw = RidgeForecaster()
bt_raw = ridge_raw.backtest(raw_data, start_date='2020-01-01')
mae_raw = bt_raw['error'].abs().mean()
print(f"   Ridge Raw MAE: {mae_raw:.4f}")
print(f"   Периодов в бэктесте: {len(bt_raw)}")

# Ridge SA по 3 компонентам
print("\n2. Бэктест Ridge SA (3 компонента)...")
ridge_sa = RidgeSAForecaster()
bt_sa = ridge_sa.backtest(start_date='2020-01-01')
mae_sa = bt_sa['error'].abs().mean()
print(f"   Ridge SA MAE: {mae_sa:.4f}")
print(f"   Периодов в бэктесте: {len(bt_sa)}")

# Ridge SA по 47 субкомпонентам
print("\n3. Бэктест Ridge SA Sub (47 субкомпонентов)...")
ridge_sa_sub = RidgeSASubForecaster()
bt_sa_sub = ridge_sa_sub.backtest(start_date='2020-01-01')
mae_sa_sub = bt_sa_sub['error'].abs().mean()
print(f"   Ridge SA Sub MAE: {mae_sa_sub:.4f}")
print(f"   Периодов в бэктесте: {len(bt_sa_sub)}")

# Сводная таблица
print("\n" + "=" * 60)
print("СВОДНАЯ ТАБЛИЦА")
print("=" * 60)

results = pd.DataFrame({
    'Модель': ['Ridge Raw', 'Ridge SA (3 comp)', 'Ridge SA Sub (47 comp)'],
    'MAE': [mae_raw, mae_sa, mae_sa_sub],
    'Периодов': [len(bt_raw), len(bt_sa), len(bt_sa_sub)]
})
results['vs Raw %'] = ((results['MAE'] / mae_raw - 1) * 100).round(1)
results['MAE'] = results['MAE'].round(4)

print(results.to_string(index=False))

# Дополнительные метрики
print("\n" + "=" * 60)
print("ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ (по годам)")
print("=" * 60)

for name, bt in [('Ridge Raw', bt_raw), ('Ridge SA', bt_sa), ('Ridge SA Sub', bt_sa_sub)]:
    bt['year'] = bt['date'].dt.year
    yearly = bt.groupby('year')['error'].apply(lambda x: x.abs().mean())
    print(f"\n{name}:")
    for year, mae in yearly.items():
        print(f"   {year}: MAE = {mae:.4f}")

# Направленная точность
print("\n" + "=" * 60)
print("НАПРАВЛЕННАЯ ТОЧНОСТЬ")
print("=" * 60)

for name, bt in [('Ridge Raw', bt_raw), ('Ridge SA', bt_sa), ('Ridge SA Sub', bt_sa_sub)]:
    # Предсказываем направление изменения (выше/ниже 100)
    bt['pred_dir'] = (bt['prediction'] > 100).astype(int)
    bt['actual_dir'] = (bt['actual'] > 100).astype(int)
    accuracy = (bt['pred_dir'] == bt['actual_dir']).mean() * 100
    print(f"{name}: {accuracy:.1f}%")

print("\n" + "=" * 60)
print("ИТОГ")
print("=" * 60)

best_model = results.loc[results['MAE'].idxmin(), 'Модель']
best_mae = results['MAE'].min()
print(f"\nЛучшая модель: {best_model}")
print(f"MAE: {best_mae:.4f}")
