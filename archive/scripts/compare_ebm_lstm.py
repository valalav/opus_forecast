"""
Сравнение метрик EBM vs LSTM
============================

Цель: Определить, стоит ли заменять LSTM на EBM в ансамбле.
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from sirena.models.ebm import EBMForecaster
from sirena.models.lstm import LSTMForecaster, create_lstm_model

def load_data():
    """Загрузка данных КБР."""
    df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')

    if 'MoM' in df_raw.columns:
        if df_raw['MoM'].dtype == object:
            df_raw['MoM'] = df_raw['MoM'].astype(str).str.replace(',', '.')
        df_raw['MoM'] = pd.to_numeric(df_raw['MoM'], errors='coerce')

    if 'Day' in df_raw.columns:
        df_raw['Date'] = pd.to_datetime(df_raw['Day'], format='%d.%m.%Y', errors='coerce')

    df = df_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
    df = df.sort_index()

    return df

def calculate_metrics(results):
    """Расчёт метрик."""
    if results.empty:
        return {'MAE': np.nan, 'RMSE': np.nan, 'KPI': np.nan, 'n': 0}

    errors = results['error'].abs()
    mae = errors.mean()
    rmse = np.sqrt((results['error'] ** 2).mean())
    kpi = (errors <= 0.5).sum() / len(results) * 100

    return {
        'MAE': mae,
        'RMSE': rmse,
        'KPI': kpi,
        'n': len(results)
    }

def main():
    print("=" * 60)
    print("СРАВНЕНИЕ МЕТРИК: EBM vs LSTM")
    print("=" * 60)

    # Загрузка данных
    print("\n1. Загрузка данных...")
    df = load_data()
    print(f"   Период: {df.index.min().strftime('%Y-%m')} - {df.index.max().strftime('%Y-%m')}")
    print(f"   Наблюдений: {len(df)}")

    # Бэктест EBM
    print("\n2. Бэктест EBM...")
    try:
        ebm = EBMForecaster()
        ebm_results = ebm.backtest(df, start_date='2019-01-01')
        ebm_metrics = calculate_metrics(ebm_results)
        print(f"   Успешно: {ebm_metrics['n']} прогнозов")
    except Exception as e:
        print(f"   ОШИБКА: {e}")
        ebm_metrics = {'MAE': np.nan, 'RMSE': np.nan, 'KPI': np.nan, 'n': 0}

    # Бэктест LSTM
    print("\n3. Бэктест LSTM...")
    try:
        lstm = create_lstm_model(epochs=50, patience=5)
        lstm_results = lstm.backtest(df, start_date='2019-01-01')
        lstm_metrics = calculate_metrics(lstm_results)
        print(f"   Успешно: {lstm_metrics['n']} прогнозов")
    except Exception as e:
        print(f"   ОШИБКА: {e}")
        lstm_metrics = {'MAE': np.nan, 'RMSE': np.nan, 'KPI': np.nan, 'n': 0}

    # Результаты
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 60)

    print(f"\n{'Метрика':<10} {'EBM':>12} {'LSTM':>12} {'Разница':>12}")
    print("-" * 50)

    for metric in ['MAE', 'RMSE', 'KPI']:
        ebm_val = ebm_metrics[metric]
        lstm_val = lstm_metrics[metric]

        if pd.notna(ebm_val) and pd.notna(lstm_val):
            diff = ebm_val - lstm_val
            better = "EBM лучше" if (diff < 0 and metric != 'KPI') or (diff > 0 and metric == 'KPI') else "LSTM лучше"
            print(f"{metric:<10} {ebm_val:>12.4f} {lstm_val:>12.4f} {diff:>+12.4f} ({better})")
        else:
            print(f"{metric:<10} {ebm_val if pd.notna(ebm_val) else 'N/A':>12} {lstm_val if pd.notna(lstm_val) else 'N/A':>12}")

    print(f"{'n':>10} {ebm_metrics['n']:>12} {lstm_metrics['n']:>12}")

    # Вывод
    print("\n" + "=" * 60)
    print("ВЫВОД")
    print("=" * 60)

    if pd.notna(ebm_metrics['MAE']) and pd.notna(lstm_metrics['MAE']):
        if ebm_metrics['MAE'] < lstm_metrics['MAE']:
            print("✓ EBM имеет МЕНЬШЕ MAE — рекомендуется замена LSTM на EBM")
        elif ebm_metrics['MAE'] > lstm_metrics['MAE']:
            print("✗ LSTM имеет МЕНЬШЕ MAE — замена НЕ рекомендуется")
        else:
            print("= MAE одинаковый — можно заменить ради интерпретируемости")

        if ebm_metrics['KPI'] > lstm_metrics['KPI']:
            print("✓ EBM имеет ВЫШЕ KPI")
        elif ebm_metrics['KPI'] < lstm_metrics['KPI']:
            print("✗ LSTM имеет ВЫШЕ KPI")
    else:
        print("Недостаточно данных для сравнения")

    # Сохранение результатов
    comparison = pd.DataFrame({
        'Model': ['EBM', 'LSTM'],
        'MAE': [ebm_metrics['MAE'], lstm_metrics['MAE']],
        'RMSE': [ebm_metrics['RMSE'], lstm_metrics['RMSE']],
        'KPI': [ebm_metrics['KPI'], lstm_metrics['KPI']],
        'n': [ebm_metrics['n'], lstm_metrics['n']]
    })
    comparison.to_csv('ebm_lstm_comparison.csv', index=False)
    print(f"\nРезультаты сохранены в ebm_lstm_comparison.csv")

    return ebm_metrics, lstm_metrics

if __name__ == '__main__':
    main()
