"""
Тестирование новых моделей СИРЕНА-КБР v4.2
==========================================

Новые модели:
1. RegimeSwitchingEnsemble — режимозависимые веса
2. CatBoost — gradient boosting для малых данных
3. RidgeExtended — Ridge с расширенными признаками
4. BayesianRidge — с доверительными интервалами
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sirena import DataLoader
from sirena.models import ModelRegistry

print("=" * 60)
print("ТЕСТИРОВАНИЕ МОДЕЛЕЙ СИРЕНА-КБР v4.2")
print("=" * 60)

# Загружаем данные
loader = DataLoader()
df = loader.load_monthly_kbr()
print(f"\nДанные: {len(df)} месяцев, {df.index.min().date()} - {df.index.max().date()}")

# ============================================================================
# 1. REGIME-SWITCHING ENSEMBLE
# ============================================================================
print("\n" + "=" * 60)
print("1. REGIME-SWITCHING ENSEMBLE")
print("=" * 60)

try:
    from sirena.models import RegimeSwitchingEnsemble, detect_regime

    print("\nОпределение текущего режима...")
    regime, diag = detect_regime(df)
    print(f"  Режим: {regime}")
    print(f"  ΔRuonia: {diag.get('ruonia_change', 'N/A')}")
    print(f"  ΔKi: {diag.get('ki_change', 'N/A')}")
    print(f"  Сигналы: {diag.get('signals', [])}")

    print("\nИнициализация модели...")
    rs = RegimeSwitchingEnsemble()
    rs.fit(df)
    print(f"  ✓ Модель обучена")
    print(f"  Текущий режим: {rs.current_regime}")
    print(f"  Текущие веса: {rs.current_weights}")

    print("\nПрогноз...")
    result = rs.forecast_with_regime(horizon=3)
    print(f"  Прогноз (3 мес): {result['ensemble']}")
    print(f"  ✓ Regime-Switching работает")

except Exception as e:
    print(f"  ✗ Ошибка: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 2. RIDGE EXTENDED
# ============================================================================
print("\n" + "=" * 60)
print("2. RIDGE EXTENDED (расширенные признаки)")
print("=" * 60)

try:
    from sirena.models import RidgeExtendedForecaster

    print("\nИнициализация модели...")
    rex = RidgeExtendedForecaster()
    rex.fit(df)
    print(f"  ✓ Модель обучена")
    print(f"  Признаков: {len(rex.FEATURES)}")

    print("\nВажность признаков (топ-10):")
    importance = rex.get_feature_importance()
    for _, row in importance.head(10).iterrows():
        new_tag = " [NEW]" if row.get('is_new', False) else ""
        print(f"  {row['feature']}: {row['coefficient']:.4f}{new_tag}")

    print("\n  ✓ Ridge Extended работает")

except Exception as e:
    print(f"  ✗ Ошибка: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 3. BAYESIAN RIDGE
# ============================================================================
print("\n" + "=" * 60)
print("3. BAYESIAN RIDGE (с доверительными интервалами)")
print("=" * 60)

try:
    from sirena.models import BayesianRidgeForecaster

    print("\nИнициализация модели...")
    br = BayesianRidgeForecaster()
    br.fit(df)
    print(f"  ✓ Модель обучена")

    params = br.get_model_params()
    print(f"  Оптимизированные параметры:")
    print(f"    alpha (noise precision): {params['alpha']:.4f}")
    print(f"    lambda (weight precision): {params['lambda']:.4f}")
    print(f"    sigma (noise std): {params['sigma']:.4f}")

    print("\nПрогноз с CI на последнюю дату...")
    target_date = df.index[-1]
    pred = br.predict_with_ci(df, target_date)
    print(f"  Дата: {target_date.date()}")
    print(f"  Прогноз: {pred['prediction']:.3f}")
    print(f"  Bayesian: {pred['pred_bayesian']:.3f}")
    print(f"  Std: {pred['std']:.3f}")
    print(f"  95% CI: [{pred['ci_lower']:.3f}, {pred['ci_upper']:.3f}]")

    print("\n  ✓ Bayesian Ridge работает")

except Exception as e:
    print(f"  ✗ Ошибка: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 4. CATBOOST (если установлен)
# ============================================================================
print("\n" + "=" * 60)
print("4. CATBOOST")
print("=" * 60)

try:
    from sirena.models import CatBoostForecaster, CATBOOST_AVAILABLE

    if not CATBOOST_AVAILABLE:
        print("  ⚠ CatBoost не установлен")
        print("  Установите: pip install catboost")
    else:
        print("\nИнициализация модели...")
        cb = CatBoostForecaster()
        cb.fit(df)
        print(f"  ✓ Модель обучена")

        print("\nПрогноз (3 мес)...")
        fc = cb.forecast(horizon=3)
        print(f"  Прогноз: {fc}")

        print("\nВажность признаков (топ-5):")
        importance = cb.get_feature_importance()
        for _, row in importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.2f}")

        print("\n  ✓ CatBoost работает")

except Exception as e:
    print(f"  ✗ Ошибка: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# СРАВНЕНИЕ МОДЕЛЕЙ (бэктест)
# ============================================================================
print("\n" + "=" * 60)
print("СРАВНЕНИЕ МОДЕЛЕЙ (MAE на бэктесте 2023-2025)")
print("=" * 60)

models_to_test = [
    ('Ridge (baseline)', 'ridge'),
    ('Ridge Extended', 'ridge_extended'),
    ('Bayesian Ridge', 'bayesian_ridge'),
    ('Regime-Switching', 'regime_switching'),
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
        print(f"\n{name}...", end=" ")
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
    print("\n" + "-" * 40)
    print("СВОДНАЯ ТАБЛИЦА")
    print("-" * 40)

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

# ============================================================================
# ИТОГ
# ============================================================================
print("\n" + "=" * 60)
print("ИТОГ")
print("=" * 60)

print("""
Реализовано в v4.2:

1. ✓ RegimeSwitchingEnsemble — режимозависимые веса
   - Автоматическое определение режима (shock/normal)
   - Разные веса моделей для разных режимов
   - forecast_with_regime() для диагностики

2. ✓ RidgeExtended — расширенные признаки
   - Дополнительные лаги (y_lag3, y_lag6)
   - Momentum (d_y_lag1, d_y_lag3)
   - Volatility (y_vol3, y_vol6)
   - Календарь (is_jan, is_dec, quarter)

3. ✓ BayesianRidge — доверительные интервалы
   - Автоматическая регуляризация
   - predict_with_ci() возвращает CI
   - Калиброванная неопределённость

4. ✓ CatBoost — gradient boosting для малых данных
   - Ordered boosting (меньше переобучение)
   - Категориальные признаки
   - Агрессивная регуляризация

Использование:
    from sirena.models import ModelRegistry

    # Режимозависимый ансамбль
    rs = ModelRegistry.get('regime_switching')
    rs.fit(df)
    fc = rs.forecast_with_regime(12)
    print(rs.current_regime)

    # Bayesian Ridge с CI
    br = ModelRegistry.get('bayesian_ridge')
    br.fit(df)
    pred = br.predict_with_ci(df, target_date)
    print(f"{pred['prediction']:.2f} ± {1.96*pred['std']:.2f}")
""")
