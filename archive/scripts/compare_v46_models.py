"""
Сравнительный бэктест моделей v4.6 (Эксперименты из методик ЦБ)
===============================================================

Тестируем:
1. Ridge на базисных индексах (Эксперимент 1)
2. Ridge с shock dummies (Эксперимент 3)
3. Ridge baseline для сравнения
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Загрузка данных
def load_data():
    """Загрузка данных из CSV."""
    df = pd.read_csv(
        'data/infl_kbr.csv',
        sep=';',
        decimal=','
    )
    df['Day'] = pd.to_datetime(df['Day'], format='%d.%m.%Y')

    # Pivot
    pivot = df.pivot(index='Day', columns='Товар', values='MoM')
    pivot.index = pd.to_datetime(pivot.index)
    pivot = pivot.sort_index()

    # Конвертируем в numeric (важно!)
    for col in pivot.columns:
        pivot[col] = pd.to_numeric(pivot[col], errors='coerce')

    return pivot


def run_backtest():
    """Запуск бэктеста для всех моделей."""
    print("=" * 60)
    print("SIRENA-KBR v4.6: Эксперименты из методик ЦБ")
    print("=" * 60)

    df = load_data()
    print(f"\nДанные: {df.index.min().strftime('%Y-%m')} — {df.index.max().strftime('%Y-%m')}")
    print(f"Количество точек: {len(df)}")

    # Импорт моделей
    from sirena.models import RidgeForecaster
    from sirena.models.ridge_base_index import RidgeBaseIndexForecaster
    from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

    # Период бэктеста: 2023-01 — последние данные (исключая 2022)
    start_date = '2023-01-01'
    target_col = 'Все товары и услуги'

    results = {}

    # 1. Ridge baseline
    print("\n[1/5] Ridge baseline...")
    try:
        model = RidgeForecaster()
        bt_results = model.backtest(df, start_date=start_date, target_col=target_col)
        metrics = model.get_metrics(bt_results)
        results['Ridge baseline'] = {
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'KPI': metrics['KPI'],
            'n_points': len(bt_results)
        }
        print(f"   MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")
    except Exception as e:
        print(f"   ОШИБКА: {e}")

    # 2. Ridge на базисных индексах
    print("\n[2/5] Ridge Base Index...")
    try:
        model = RidgeBaseIndexForecaster(use_log=False)
        bt_results = model.backtest(df, start_date=start_date, target_col=target_col)
        metrics = model.get_metrics(bt_results)
        results['Ridge Base Index'] = {
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'KPI': metrics['KPI'],
            'n_points': len(bt_results)
        }
        print(f"   MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")
    except Exception as e:
        print(f"   ОШИБКА: {e}")

    # 3. Ridge на базисных индексах с логарифмом (Эксперимент 4)
    print("\n[3/5] Ridge Base Index + Log...")
    try:
        model = RidgeBaseIndexForecaster(use_log=True)
        bt_results = model.backtest(df, start_date=start_date, target_col=target_col)
        metrics = model.get_metrics(bt_results)
        results['Ridge Base Log'] = {
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'KPI': metrics['KPI'],
            'n_points': len(bt_results)
        }
        print(f"   MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")
    except Exception as e:
        print(f"   ОШИБКА: {e}")

    # 4. Ridge с shock dummies (используем 2022 как dummy)
    print("\n[4/5] Ridge Shock Dummies (include 2022)...")
    try:
        model = RidgeShockDummiesForecaster(use_2022_dummy=True)
        bt_results = model.backtest(df, start_date=start_date, target_col=target_col)
        metrics = model.get_metrics(bt_results)
        results['Ridge Shock Dummies'] = {
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'KPI': metrics['KPI'],
            'n_points': len(bt_results)
        }
        print(f"   MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")
    except Exception as e:
        print(f"   ОШИБКА: {e}")

    # 5. Ridge с shock dummies (без 2022)
    print("\n[5/5] Ridge Shock Dummies (exclude 2022)...")
    try:
        model = RidgeShockDummiesForecaster(use_2022_dummy=False)
        bt_results = model.backtest(df, start_date=start_date, target_col=target_col)
        metrics = model.get_metrics(bt_results)
        results['Ridge Shock (no 2022)'] = {
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'KPI': metrics['KPI'],
            'n_points': len(bt_results)
        }
        print(f"   MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")
    except Exception as e:
        print(f"   ОШИБКА: {e}")

    # Сводная таблица
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("=" * 60)

    if not results:
        print("Нет результатов для отображения")
        return

    # Создаём DataFrame
    df_results = pd.DataFrame(results).T
    df_results['vs_baseline'] = ((df_results['MAE'] - results.get('Ridge baseline', {}).get('MAE', df_results['MAE'].iloc[0])) /
                                  results.get('Ridge baseline', {}).get('MAE', df_results['MAE'].iloc[0]) * 100)

    # Сортируем по MAE
    df_results = df_results.sort_values('MAE')

    print("\n{:<25} {:>8} {:>8} {:>8} {:>10}".format(
        'Model', 'MAE', 'RMSE', 'KPI%', 'vs Ridge%'))
    print("-" * 60)

    baseline_mae = results.get('Ridge baseline', {}).get('MAE', 0)
    for model_name, row in df_results.iterrows():
        vs = (row['MAE'] - baseline_mae) / baseline_mae * 100 if baseline_mae > 0 else 0
        print("{:<25} {:>8.4f} {:>8.4f} {:>8.1f} {:>+10.2f}%".format(
            model_name, row['MAE'], row['RMSE'], row['KPI'], vs))

    # Сохранение результатов
    df_results.to_csv('model_comparison_v46.csv')
    print(f"\nРезультаты сохранены в model_comparison_v46.csv")

    # Анализ лучшей модели
    best_model = df_results['MAE'].idxmin()
    best_mae = df_results['MAE'].min()
    best_vs = df_results.loc[best_model, 'vs_baseline']

    print(f"\n{'='*60}")
    print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model}")
    print(f"MAE: {best_mae:.4f} ({best_vs:+.2f}% vs Ridge baseline)")
    print(f"{'='*60}")

    return df_results


if __name__ == '__main__':
    run_backtest()
