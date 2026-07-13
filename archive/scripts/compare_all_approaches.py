"""
Полное сравнение всех подходов из методик ЦБ
=============================================

Сравниваем 3 подхода для каждой модели:
1. Original — базовая версия (исключает 2022)
2. Shock Dummies — dummy переменные для шоков (включает 2022)
3. Base Index — обучение на базисных индексах

Модели: Ridge, NGBoost, XGBoost, Ridge Extended, LightGBM
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
        return {'MAE': float('inf'), 'RMSE': float('inf'), 'KPI': 0, 'n_points': 0}

    errors = results['error'].abs()
    mae = errors.mean()
    rmse = np.sqrt((results['error'] ** 2).mean())
    kpi = (errors <= 0.5).sum() / len(results) * 100

    return {'MAE': mae, 'RMSE': rmse, 'KPI': kpi, 'n_points': len(results)}


def run_model_backtest(model_class, df, start_date, target_col, **kwargs):
    """Запуск бэктеста для одной модели."""
    try:
        model = model_class(**kwargs)
        bt = model.backtest(df, start_date=start_date, target_col=target_col)
        return get_metrics(bt)
    except Exception as e:
        print(f"      ОШИБКА: {e}")
        return None


def run_backtest():
    """Запуск полного сравнения."""
    print("=" * 80)
    print("SIRENA-KBR v4.6: Полное сравнение подходов из методик ЦБ")
    print("=" * 80)

    df = load_data()
    print(f"\nДанные: {df.index.min().strftime('%Y-%m')} — {df.index.max().strftime('%Y-%m')}")
    print(f"Количество точек: {len(df)}")

    start_date = '2023-01-01'
    target_col = 'Все товары и услуги'

    results = {}

    # === RIDGE ===
    print("\n" + "=" * 40)
    print("RIDGE")
    print("=" * 40)

    print("\n[Ridge] Original...")
    from sirena.models import RidgeForecaster
    metrics = run_model_backtest(RidgeForecaster, df, start_date, target_col)
    if metrics:
        results['Ridge Original'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")

    print("[Ridge] Shock Dummies...")
    from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster
    metrics = run_model_backtest(RidgeShockDummiesForecaster, df, start_date, target_col, use_2022_dummy=True)
    if metrics:
        results['Ridge Shock'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")

    print("[Ridge] Base Index...")
    from sirena.models.ridge_base_index import RidgeBaseIndexForecaster
    metrics = run_model_backtest(RidgeBaseIndexForecaster, df, start_date, target_col)
    if metrics:
        results['Ridge Base'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")

    # === NGBOOST ===
    print("\n" + "=" * 40)
    print("NGBOOST")
    print("=" * 40)

    print("\n[NGBoost] Original...")
    from sirena.models.ngboost_model import NGBoostForecaster
    metrics = run_model_backtest(NGBoostForecaster, df, start_date, target_col)
    if metrics:
        results['NGBoost Original'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")

    print("[NGBoost] Shock Dummies...")
    from sirena.models.ngboost_shock import NGBoostShockForecaster
    metrics = run_model_backtest(NGBoostShockForecaster, df, start_date, target_col)
    if metrics:
        results['NGBoost Shock'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")

    print("[NGBoost] Base Index...")
    from sirena.models.ngboost_base import NGBoostBaseIndexForecaster
    metrics = run_model_backtest(NGBoostBaseIndexForecaster, df, start_date, target_col)
    if metrics:
        results['NGBoost Base'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")

    # === XGBOOST ===
    print("\n" + "=" * 40)
    print("XGBOOST")
    print("=" * 40)

    print("\n[XGBoost] Original...")
    from sirena.models.xgboost_model import XGBoostForecaster
    metrics = run_model_backtest(XGBoostForecaster, df, start_date, target_col)
    if metrics:
        results['XGBoost Original'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")

    print("[XGBoost] Shock Dummies...")
    from sirena.models.xgboost_shock import XGBoostShockForecaster
    metrics = run_model_backtest(XGBoostShockForecaster, df, start_date, target_col)
    if metrics:
        results['XGBoost Shock'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")

    print("[XGBoost] Base Index...")
    from sirena.models.xgboost_base import XGBoostBaseIndexForecaster
    metrics = run_model_backtest(XGBoostBaseIndexForecaster, df, start_date, target_col)
    if metrics:
        results['XGBoost Base'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")

    # === RIDGE EXTENDED ===
    print("\n" + "=" * 40)
    print("RIDGE EXTENDED")
    print("=" * 40)

    print("\n[Ridge Ext] Original...")
    from sirena.models.ridge_extended import RidgeExtendedForecaster
    metrics = run_model_backtest(RidgeExtendedForecaster, df, start_date, target_col)
    if metrics:
        results['RidgeExt Original'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")

    print("[Ridge Ext] Shock Dummies...")
    from sirena.models.ridge_extended_shock import RidgeExtendedShockForecaster
    metrics = run_model_backtest(RidgeExtendedShockForecaster, df, start_date, target_col)
    if metrics:
        results['RidgeExt Shock'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")

    print("[Ridge Ext] Base Index...")
    from sirena.models.ridge_extended_base import RidgeExtendedBaseIndexForecaster
    metrics = run_model_backtest(RidgeExtendedBaseIndexForecaster, df, start_date, target_col)
    if metrics:
        results['RidgeExt Base'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")

    # === LIGHTGBM ===
    print("\n" + "=" * 40)
    print("LIGHTGBM")
    print("=" * 40)

    print("\n[LightGBM] Original...")
    from sirena.models.lightgbm import LightGBMForecaster
    metrics = run_model_backtest(LightGBMForecaster, df, start_date, target_col)
    if metrics:
        results['LightGBM Original'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")

    print("[LightGBM] Shock Dummies...")
    from sirena.models.lightgbm_shock import LightGBMShockForecaster
    metrics = run_model_backtest(LightGBMShockForecaster, df, start_date, target_col)
    if metrics:
        results['LightGBM Shock'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")

    print("[LightGBM] Base Index...")
    from sirena.models.lightgbm_base import LightGBMBaseIndexForecaster
    metrics = run_model_backtest(LightGBMBaseIndexForecaster, df, start_date, target_col)
    if metrics:
        results['LightGBM Base'] = metrics
        print(f"   MAE: {metrics['MAE']:.4f}")

    # === РЕЗУЛЬТАТЫ ===
    print("\n" + "=" * 80)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 80)

    if not results:
        print("Нет результатов для отображения")
        return

    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values('MAE')

    baseline_mae = results.get('Ridge Original', {}).get('MAE', df_results['MAE'].iloc[0])

    print("\n{:<22} {:>8} {:>8} {:>8} {:>12}".format(
        'Model', 'MAE', 'RMSE', 'KPI%', 'vs Ridge%'))
    print("-" * 80)

    for model_name, row in df_results.iterrows():
        vs = (row['MAE'] - baseline_mae) / baseline_mae * 100 if baseline_mae > 0 else 0
        print("{:<22} {:>8.4f} {:>8.4f} {:>8.1f} {:>+12.2f}%".format(
            model_name, row['MAE'], row['RMSE'], row['KPI'], vs))

    # === СРАВНЕНИЕ ПО ПОДХОДАМ ===
    print("\n" + "=" * 80)
    print("СРАВНЕНИЕ ПО ПОДХОДАМ (лучший для каждой модели)")
    print("=" * 80)

    models = ['Ridge', 'NGBoost', 'XGBoost', 'RidgeExt', 'LightGBM']
    approaches = ['Original', 'Shock', 'Base']

    print("\n{:<12} {:>12} {:>12} {:>12} {:>15}".format(
        'Model', 'Original', 'Shock', 'Base', 'Best Approach'))
    print("-" * 80)

    best_approaches = []
    for model in models:
        row = {}
        for approach in approaches:
            key = f'{model} {approach}'
            if key in results:
                row[approach] = results[key]['MAE']
            else:
                row[approach] = float('inf')

        best = min(row.items(), key=lambda x: x[1])
        best_approaches.append((model, best[0], best[1]))

        orig = f"{row.get('Original', float('inf')):.4f}" if row.get('Original') != float('inf') else "N/A"
        shock = f"{row.get('Shock', float('inf')):.4f}" if row.get('Shock') != float('inf') else "N/A"
        base = f"{row.get('Base', float('inf')):.4f}" if row.get('Base') != float('inf') else "N/A"

        print("{:<12} {:>12} {:>12} {:>12} {:>15}".format(
            model, orig, shock, base, f"{best[0]} ({best[1]:.4f})"))

    # === РЕКОМЕНДАЦИИ ===
    print("\n" + "=" * 80)
    print("РЕКОМЕНДАЦИИ")
    print("=" * 80)

    # Подсчёт лучших подходов
    approach_wins = {}
    for model, approach, mae in best_approaches:
        approach_wins[approach] = approach_wins.get(approach, 0) + 1

    print("\nПобеды по подходам:")
    for approach, wins in sorted(approach_wins.items(), key=lambda x: -x[1]):
        print(f"  - {approach}: {wins} модел(ей/и)")

    # Лучшая модель
    best_model = df_results['MAE'].idxmin()
    best_mae = df_results['MAE'].min()
    best_vs = (best_mae - baseline_mae) / baseline_mae * 100

    print(f"\n{'='*80}")
    print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model}")
    print(f"MAE: {best_mae:.4f} ({best_vs:+.2f}% vs Ridge Original)")
    print(f"{'='*80}")

    # Сохранение
    df_results.to_csv('all_approaches_comparison.csv')
    print(f"\nРезультаты сохранены в all_approaches_comparison.csv")

    return df_results


if __name__ == '__main__':
    run_backtest()
