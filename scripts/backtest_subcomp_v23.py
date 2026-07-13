#!/usr/bin/env python3
"""
Бэктест SubcomponentMulti v2.3 с rate-признаками

Сравнение с v2.2:
- v2.2 MAE: 0.274
- v2.3 добавляет: ruonia_diff_lag1, spread_lag4, ki_diff_lag6
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sirena.models.subcomponent_multi import SubcomponentMultiForecaster


def load_data():
    """Загрузка данных."""
    df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')

    # Fix numeric columns
    for col in df.columns:
        if col != 'Date' and df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
    df = df.set_index('Date').sort_index()
    df.index = df.index.to_period('M').to_timestamp()

    return df


def run_backtest(horizon=1, test_months=12):
    """Rolling backtest для SubcomponentMulti v2.3."""
    print(f"\n{'='*60}")
    print(f"БЭКТЕСТ SubcomponentMulti v2.3 (h={horizon})")
    print(f"{'='*60}\n")

    # Load data
    df = load_data()
    last_fact = df.index.max()

    # Test dates
    test_dates = pd.date_range(
        end=last_fact,
        periods=test_months,
        freq='MS'
    )

    print(f"Период теста: {test_dates[0].strftime('%Y-%m')} — {test_dates[-1].strftime('%Y-%m')}")
    print(f"Горизонт: {horizon} мес.\n")

    results = []

    print(f"{'Месяц':<10} | {'Факт':>6} | {'SubcompMulti':>12} | {'Error':>7}")
    print("-" * 45)

    for target_date in test_dates:
        # Train до (target_date - horizon)
        cutoff = target_date - pd.DateOffset(months=horizon)
        train = df[df.index <= cutoff].copy()

        # Actual
        if target_date not in df.index or 'mom' not in df.columns:
            continue
        actual = df.loc[target_date, 'mom'] - 100

        # Forecast
        try:
            model = SubcomponentMultiForecaster(horizon=horizon)
            model.fit(train, 'Все товары и услуги')
            result = model.predict(train, target_date)
            pred = result['prediction'] - 100 if result else np.nan
        except Exception as e:
            pred = np.nan

        error = actual - pred if not np.isnan(pred) else np.nan

        results.append({
            'Date': target_date,
            'Actual': actual,
            'SubcompMulti': pred,
            'Error': error
        })

        print(f"{target_date.strftime('%Y-%m'):<10} | {actual:6.2f} | {pred:12.2f} | {error:7.2f}")

    # Calculate metrics
    results_df = pd.DataFrame(results)
    valid = results_df.dropna()

    if len(valid) > 0:
        mae = valid['Error'].abs().mean()
        rmse = np.sqrt((valid['Error'] ** 2).mean())
        kpi_violations = (valid['Error'].abs() > 0.5).sum()

        print(f"\n{'='*60}")
        print("РЕЗУЛЬТАТЫ")
        print(f"{'='*60}")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"KPI violations: {kpi_violations}/{len(valid)}")
        print(f"Coverage 50%: {((valid['Error'].abs() <= 0.5).mean() * 100):.1f}%")

        return mae, results_df

    return None, results_df


if __name__ == '__main__':
    # Run h=1 backtest (main KPI)
    mae_h1, results_h1 = run_backtest(horizon=1, test_months=12)

    # Save results
    if results_h1 is not None:
        results_h1.to_csv('archive/results/backtest_subcomp_v23_h1.csv', index=False)
        print(f"\nРезультаты сохранены: archive/results/backtest_subcomp_v23_h1.csv")

    # Comparison with v2.2
    print(f"\n{'='*60}")
    print("СРАВНЕНИЕ С v2.2")
    print(f"{'='*60}")
    print(f"v2.2 MAE: 0.274")
    if mae_h1:
        improvement = (0.274 - mae_h1) / 0.274 * 100
        print(f"v2.3 MAE: {mae_h1:.3f}")
        print(f"Улучшение: {improvement:+.1f}%")
