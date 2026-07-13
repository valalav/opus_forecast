"""
Сравнительный бэктест: Shock Dummies vs Original
=================================================

Тестируем модели с shock dummies против оригинальных версий:
1. NGBoost Shock vs NGBoost
2. XGBoost Shock vs XGBoost
3. Ridge Extended Shock vs Ridge Extended
4. LightGBM Shock vs LightGBM

Также включаем базовые Ridge и Ridge Shock Dummies для сравнения.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_data():
    """Загрузка данных из CSV."""
    df = pd.read_csv(
        'data/infl_kbr.csv',
        sep=';',
        decimal=','
    )
    df['Day'] = pd.to_datetime(df['Day'], format='%d.%m.%Y')

    pivot = df.pivot(index='Day', columns='Товар', values='MoM')
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()

    for col in pivot.columns:
        pivot[col] = pd.to_numeric(pivot[col], errors='coerce')

    return pivot


def get_metrics(results: pd.DataFrame) -> dict:
    """Расчёт метрик качества."""
    if results.empty:
        return {'MAE': float('inf'), 'RMSE': float('inf'), 'KPI': 0}

    errors = results['error'].abs()
    mae = errors.mean()
    rmse = np.sqrt((results['error'] ** 2).mean())
    kpi = (errors <= 0.5).sum() / len(results) * 100

    return {'MAE': mae, 'RMSE': rmse, 'KPI': kpi, 'n_points': len(results)}


def run_backtest():
    """Запуск сравнительного бэктеста."""
    print("=" * 70)
    print("SIRENA-KBR v4.6: Shock Dummies Comparison")
    print("=" * 70)

    df = load_data()
    print(f"\nДанные: {df.index.min().strftime('%Y-%m')} — {df.index.max().strftime('%Y-%m')}")
    print(f"Количество точек: {len(df)}")

    start_date = '2023-01-01'
    target_col = 'Все товары и услуги'

    results = {}

    # === 1. Ridge baseline ===
    print("\n[1/10] Ridge baseline...")
    try:
        from sirena.models import RidgeForecaster
        model = RidgeForecaster()
        bt = model.backtest(df, start_date=start_date, target_col=target_col)
        metrics = get_metrics(bt)
        results['Ridge'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")
    except Exception as e:
        print(f"   ОШИБКА: {e}")

    # === 2. Ridge Shock Dummies ===
    print("\n[2/10] Ridge Shock Dummies...")
    try:
        from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster
        model = RidgeShockDummiesForecaster(use_2022_dummy=True)
        bt = model.backtest(df, start_date=start_date, target_col=target_col)
        metrics = get_metrics(bt)
        results['Ridge Shock'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")
    except Exception as e:
        print(f"   ОШИБКА: {e}")

    # === 3. NGBoost original ===
    print("\n[3/10] NGBoost original...")
    try:
        from sirena.models.ngboost_model import NGBoostForecaster
        model = NGBoostForecaster()
        bt = model.backtest(df, start_date=start_date, target_col=target_col)
        metrics = get_metrics(bt)
        results['NGBoost'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")
    except Exception as e:
        print(f"   ОШИБКА: {e}")

    # === 4. NGBoost Shock ===
    print("\n[4/10] NGBoost Shock...")
    try:
        from sirena.models.ngboost_shock import NGBoostShockForecaster
        model = NGBoostShockForecaster()
        bt = model.backtest(df, start_date=start_date, target_col=target_col)
        metrics = get_metrics(bt)
        results['NGBoost Shock'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")
    except Exception as e:
        print(f"   ОШИБКА: {e}")

    # === 5. XGBoost original ===
    print("\n[5/10] XGBoost original...")
    try:
        from sirena.models.xgboost_model import XGBoostForecaster
        model = XGBoostForecaster()
        bt = model.backtest(df, start_date=start_date, target_col=target_col)
        metrics = get_metrics(bt)
        results['XGBoost'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")
    except Exception as e:
        print(f"   ОШИБКА: {e}")

    # === 6. XGBoost Shock ===
    print("\n[6/10] XGBoost Shock...")
    try:
        from sirena.models.xgboost_shock import XGBoostShockForecaster
        model = XGBoostShockForecaster()
        bt = model.backtest(df, start_date=start_date, target_col=target_col)
        metrics = get_metrics(bt)
        results['XGBoost Shock'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")
    except Exception as e:
        print(f"   ОШИБКА: {e}")

    # === 7. Ridge Extended original ===
    print("\n[7/10] Ridge Extended original...")
    try:
        from sirena.models.ridge_extended import RidgeExtendedForecaster
        model = RidgeExtendedForecaster()
        bt = model.backtest(df, start_date=start_date, target_col=target_col)
        metrics = get_metrics(bt)
        results['Ridge Extended'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")
    except Exception as e:
        print(f"   ОШИБКА: {e}")

    # === 8. Ridge Extended Shock ===
    print("\n[8/10] Ridge Extended Shock...")
    try:
        from sirena.models.ridge_extended_shock import RidgeExtendedShockForecaster
        model = RidgeExtendedShockForecaster()
        bt = model.backtest(df, start_date=start_date, target_col=target_col)
        metrics = get_metrics(bt)
        results['Ridge Ext Shock'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")
    except Exception as e:
        print(f"   ОШИБКА: {e}")

    # === 9. LightGBM original ===
    print("\n[9/10] LightGBM original...")
    try:
        from sirena.models.lightgbm import LightGBMForecaster
        model = LightGBMForecaster()
        bt = model.backtest(df, start_date=start_date, target_col=target_col)
        metrics = get_metrics(bt)
        results['LightGBM'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")
    except Exception as e:
        print(f"   ОШИБКА: {e}")

    # === 10. LightGBM Shock ===
    print("\n[10/10] LightGBM Shock...")
    try:
        from sirena.models.lightgbm_shock import LightGBMShockForecaster
        model = LightGBMShockForecaster()
        bt = model.backtest(df, start_date=start_date, target_col=target_col)
        metrics = get_metrics(bt)
        results['LightGBM Shock'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")
    except Exception as e:
        print(f"   ОШИБКА: {e}")

    # === РЕЗУЛЬТАТЫ ===
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("=" * 70)

    if not results:
        print("Нет результатов для отображения")
        return

    # Создаём DataFrame
    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values('MAE')

    baseline_mae = results.get('Ridge', {}).get('MAE', df_results['MAE'].iloc[0])

    print("\n{:<20} {:>8} {:>8} {:>8} {:>12}".format(
        'Model', 'MAE', 'RMSE', 'KPI%', 'vs Ridge%'))
    print("-" * 70)

    for model_name, row in df_results.iterrows():
        vs = (row['MAE'] - baseline_mae) / baseline_mae * 100 if baseline_mae > 0 else 0
        print("{:<20} {:>8.4f} {:>8.4f} {:>8.1f} {:>+12.2f}%".format(
            model_name, row['MAE'], row['RMSE'], row['KPI'], vs))

    # === СРАВНЕНИЕ SHOCK vs ORIGINAL ===
    print("\n" + "=" * 70)
    print("ЭФФЕКТ SHOCK DUMMIES (Shock vs Original)")
    print("=" * 70)

    comparisons = [
        ('Ridge Shock', 'Ridge'),
        ('NGBoost Shock', 'NGBoost'),
        ('XGBoost Shock', 'XGBoost'),
        ('Ridge Ext Shock', 'Ridge Extended'),
        ('LightGBM Shock', 'LightGBM'),
    ]

    print("\n{:<20} {:>12} {:>12} {:>12}".format(
        'Comparison', 'Original', 'Shock', 'Improvement'))
    print("-" * 70)

    improvements = []
    for shock_name, orig_name in comparisons:
        if shock_name in results and orig_name in results:
            orig_mae = results[orig_name]['MAE']
            shock_mae = results[shock_name]['MAE']
            improvement = (orig_mae - shock_mae) / orig_mae * 100
            improvements.append((shock_name.replace(' Shock', ''), improvement))
            marker = '+++' if improvement > 2 else ('++' if improvement > 0 else '--')
            print("{:<20} {:>12.4f} {:>12.4f} {:>+11.2f}% {}".format(
                orig_name, orig_mae, shock_mae, improvement, marker))

    # Сохранение результатов
    df_results.to_csv('shock_dummies_comparison.csv')
    print(f"\nРезультаты сохранены в shock_dummies_comparison.csv")

    # Лучшая модель
    best_model = df_results['MAE'].idxmin()
    best_mae = df_results['MAE'].min()
    best_vs = (best_mae - baseline_mae) / baseline_mae * 100

    print(f"\n{'='*70}")
    print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model}")
    print(f"MAE: {best_mae:.4f} ({best_vs:+.2f}% vs Ridge baseline)")
    print(f"{'='*70}")

    # Рекомендации
    print("\n" + "=" * 70)
    print("РЕКОМЕНДАЦИИ")
    print("=" * 70)

    positive_improvements = [(name, imp) for name, imp in improvements if imp > 0]
    if positive_improvements:
        print("\nМодели, которые УЛУЧШИЛИСЬ с shock dummies:")
        for name, imp in sorted(positive_improvements, key=lambda x: -x[1]):
            print(f"  - {name}: +{imp:.2f}%")
    else:
        print("\nShock dummies НЕ улучшили ни одну модель.")

    return df_results


if __name__ == '__main__':
    run_backtest()
