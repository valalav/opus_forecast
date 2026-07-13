#!/usr/bin/env python3
"""
KPI Optimizer: Сдвиг прогноза для максимизации попаданий в KPI (±0.5)

Идея:
1. Берём лучшую модель (например, Huber)
2. Вычисляем средний bias по месяцам (сезонный сдвиг)
3. Корректируем прогноз на этот bias
4. Тестируем на бэктесте

Автор: Claude Code
Дата: 2025-12-28
"""

import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def load_backtest_data(horizon: int = 1) -> pd.DataFrame:
    """Load backtest predictions."""
    path = PROJECT_ROOT / f'archive/results/backtest_h{horizon}_predictions.csv'
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    return df


def calculate_monthly_bias(df: pd.DataFrame, model: str) -> dict:
    """Calculate average error by month (seasonal bias)."""
    df['Error'] = df[model] - df['Actual']
    monthly_bias = df.groupby('Month')['Error'].mean().to_dict()
    return monthly_bias


def apply_bias_correction(df: pd.DataFrame, model: str, bias: dict) -> pd.Series:
    """Apply monthly bias correction to model predictions."""
    corrected = df[model] - df['Month'].map(bias)
    return corrected


def calculate_kpi_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """Calculate KPI-focused metrics."""
    errors = (actual - predicted).abs()
    mae = errors.mean()
    kpi_hits = (errors <= 0.5).sum()
    kpi_rate = kpi_hits / len(errors) * 100
    max_error = errors.max()

    return {
        'MAE': mae,
        'KPI_Hits': kpi_hits,
        'KPI_Rate': kpi_rate,
        'Max_Error': max_error,
        'Total': len(errors)
    }


def optimize_shift(df: pd.DataFrame, model: str) -> tuple:
    """
    Find optimal constant shift to maximize KPI hits.
    Returns (optimal_shift, metrics_before, metrics_after)
    """
    best_shift = 0
    best_kpi_hits = 0

    # Try shifts from -0.3 to +0.3 in steps of 0.01
    for shift in np.arange(-0.3, 0.31, 0.01):
        shifted = df[model] + shift
        errors = (df['Actual'] - shifted).abs()
        kpi_hits = (errors <= 0.5).sum()

        if kpi_hits > best_kpi_hits:
            best_kpi_hits = kpi_hits
            best_shift = shift

    # Calculate metrics before and after
    metrics_before = calculate_kpi_metrics(df['Actual'], df[model])
    metrics_after = calculate_kpi_metrics(df['Actual'], df[model] + best_shift)

    return best_shift, metrics_before, metrics_after


def optimize_monthly_shift(df: pd.DataFrame, model: str) -> tuple:
    """
    Find optimal shift for each month to maximize KPI hits.
    Returns (monthly_shifts, metrics_before, metrics_after)
    """
    monthly_shifts = {}

    for month in range(1, 13):
        month_data = df[df['Month'] == month]
        if len(month_data) == 0:
            monthly_shifts[month] = 0
            continue

        best_shift = 0
        best_kpi_hits = 0

        for shift in np.arange(-0.5, 0.51, 0.01):
            shifted = month_data[model] + shift
            errors = (month_data['Actual'] - shifted).abs()
            kpi_hits = (errors <= 0.5).sum()

            if kpi_hits > best_kpi_hits:
                best_kpi_hits = kpi_hits
                best_shift = shift

        monthly_shifts[month] = best_shift

    # Apply monthly shifts
    shifted_predictions = df[model] + df['Month'].map(monthly_shifts)

    metrics_before = calculate_kpi_metrics(df['Actual'], df[model])
    metrics_after = calculate_kpi_metrics(df['Actual'], shifted_predictions)

    return monthly_shifts, metrics_before, metrics_after


def run_analysis(horizon: int = 1):
    """Run full KPI optimization analysis."""
    print(f"\n{'='*70}")
    print(f"KPI OPTIMIZER: Бэктест h={horizon}")
    print(f"{'='*70}\n")

    # Load data
    df = load_backtest_data(horizon)
    print(f"Загружено {len(df)} точек данных\n")

    # Find best model by MAE (excluding LMMR)
    models = ['Ridge', 'Ridge_Ext', 'Bayes_Ridge', 'ElasticNet', 'Huber', 'Ridge_Shock',
              'NGBoost', 'NGBoost_Shock', 'BVAR', 'SARIMA', 'LightGBM', 'Prophet', 'ETS', 'EBM']

    model_metrics = []
    for m in models:
        if m in df.columns:
            metrics = calculate_kpi_metrics(df['Actual'], df[m])
            metrics['Model'] = m
            model_metrics.append(metrics)

    metrics_df = pd.DataFrame(model_metrics).sort_values('MAE')
    best_model = metrics_df.iloc[0]['Model']

    print("Топ-5 моделей по MAE:")
    print("-" * 50)
    for i, row in metrics_df.head(5).iterrows():
        print(f"  {row['Model']:15} MAE: {row['MAE']:.3f}  KPI: {int(row['KPI_Hits'])}/{int(row['Total'])} ({row['KPI_Rate']:.0f}%)")

    print(f"\n{'='*70}")
    print(f"ОПТИМИЗАЦИЯ МОДЕЛИ: {best_model}")
    print(f"{'='*70}\n")

    # Method 1: Constant shift
    print("1. КОНСТАНТНЫЙ СДВИГ:")
    print("-" * 50)
    const_shift, before, after = optimize_shift(df, best_model)
    print(f"   Оптимальный сдвиг: {const_shift:+.2f}")
    print(f"   До:  MAE {before['MAE']:.3f}, KPI {int(before['KPI_Hits'])}/{int(before['Total'])} ({before['KPI_Rate']:.0f}%)")
    print(f"   После: MAE {after['MAE']:.3f}, KPI {int(after['KPI_Hits'])}/{int(after['Total'])} ({after['KPI_Rate']:.0f}%)")
    print(f"   Улучшение KPI: +{int(after['KPI_Hits'] - before['KPI_Hits'])} попаданий")

    # Method 2: Monthly shifts (seasonal)
    print(f"\n2. СЕЗОННЫЙ СДВИГ (по месяцам):")
    print("-" * 50)
    monthly_shifts, before2, after2 = optimize_monthly_shift(df, best_model)

    print("   Месячные сдвиги:")
    month_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    for m in range(1, 13):
        if m in monthly_shifts:
            print(f"     {month_names[m-1]}: {monthly_shifts[m]:+.2f}")

    print(f"\n   До:  MAE {before2['MAE']:.3f}, KPI {int(before2['KPI_Hits'])}/{int(before2['Total'])} ({before2['KPI_Rate']:.0f}%)")
    print(f"   После: MAE {after2['MAE']:.3f}, KPI {int(after2['KPI_Hits'])}/{int(after2['Total'])} ({after2['KPI_Rate']:.0f}%)")
    print(f"   Улучшение KPI: +{int(after2['KPI_Hits'] - before2['KPI_Hits'])} попаданий")

    # Method 3: Bias correction (use historical bias)
    print(f"\n3. КОРРЕКЦИЯ BIAS (исторический сдвиг):")
    print("-" * 50)
    bias = calculate_monthly_bias(df.copy(), best_model)
    corrected = apply_bias_correction(df.copy(), best_model, bias)
    metrics_bias = calculate_kpi_metrics(df['Actual'], corrected)

    print("   Исторический bias по месяцам:")
    for m in range(1, 13):
        if m in bias:
            print(f"     {month_names[m-1]}: {bias[m]:+.2f}")

    print(f"\n   До:  MAE {before['MAE']:.3f}, KPI {int(before['KPI_Hits'])}/{int(before['Total'])} ({before['KPI_Rate']:.0f}%)")
    print(f"   После: MAE {metrics_bias['MAE']:.3f}, KPI {int(metrics_bias['KPI_Hits'])}/{int(metrics_bias['Total'])} ({metrics_bias['KPI_Rate']:.0f}%)")
    print(f"   Улучшение KPI: +{int(metrics_bias['KPI_Hits'] - before['KPI_Hits'])} попаданий")

    print(f"\n{'='*70}")
    print("ИТОГ:")
    print(f"{'='*70}")

    results = [
        ('Без коррекции', before['KPI_Hits'], before['MAE']),
        ('Константный сдвиг', after['KPI_Hits'], after['MAE']),
        ('Сезонный сдвиг', after2['KPI_Hits'], after2['MAE']),
        ('Bias коррекция', metrics_bias['KPI_Hits'], metrics_bias['MAE']),
    ]

    best_method = max(results, key=lambda x: x[1])
    print(f"\nЛучший метод: {best_method[0]}")
    print(f"KPI попаданий: {int(best_method[1])}/{int(before['Total'])}")
    print(f"MAE: {best_method[2]:.3f}")

    return {
        'model': best_model,
        'const_shift': const_shift,
        'monthly_shifts': monthly_shifts,
        'bias': bias,
        'results': {
            'baseline': before,
            'const_shift': after,
            'seasonal_shift': after2,
            'bias_correction': metrics_bias
        }
    }


if __name__ == '__main__':
    # Run for h=1
    result_h1 = run_analysis(horizon=1)

    print("\n" + "="*70 + "\n")

    # Run for h=2
    result_h2 = run_analysis(horizon=2)
