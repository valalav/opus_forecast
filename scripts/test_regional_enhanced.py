#!/usr/bin/env python3
"""
Тест: добавление региональных макропоказателей к существующим production моделям.

Гипотеза: региональные данные могут улучшить уже хорошие модели (Huber, Ridge Shock).

Автор: Claude Code
Дата: 2025-12-28
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'


def load_all_data():
    """Load and merge all data sources."""
    # Load inflation data
    infl = pd.read_csv(DATA_DIR / 'inflation_data.csv', sep=';', decimal=',', on_bad_lines='skip')
    infl['Date'] = pd.to_datetime(infl['Date'], format='%d.%m.%Y')
    infl['Date'] = infl['Date'].dt.to_period('M').dt.to_timestamp()
    infl = infl.set_index('Date')

    # Load regional monthly
    macro = pd.read_csv(DATA_DIR / 'month.csv', sep=';', decimal=',')
    macro['Date'] = pd.to_datetime(macro['Date'], format='%d.%m.%Y')
    macro = macro.set_index('Date')

    col_names = {
        '1': 'ind_prod', '3': 'construction', '6': 'retail',
        '7': 'services', '11': 'ppi', '12': 'agri_prices',
        '18': 'wage', '19': 'wage_agri',
    }
    existing = [c for c in col_names.keys() if c in macro.columns]
    macro = macro[existing].rename(columns={k: col_names[k] for k in existing})

    # Load quarterly
    quart = pd.read_csv(DATA_DIR / 'quart.csv', sep=';', decimal=',')
    quart['Date'] = pd.to_datetime(quart['Data'], format='%d.%m.%Y', errors='coerce')
    quart = quart.dropna(subset=['Date'])
    quart = quart.set_index('Date')
    quart = quart[['16', '17']].rename(columns={'16': 'income_nom', '17': 'income_real'})
    quart = quart[~quart.index.duplicated(keep='last')]
    quart = quart.resample('MS').ffill()

    return infl, macro, quart


def create_baseline_features(df):
    """Create baseline features (same as production models).

    For h=1 forecasting:
    - At time t, we predict y_{t+1}
    - Features available: y_t, y_{t-1}, y_{t-2}, ...
    """
    # Target: predict NEXT month (h=1 forecast)
    df['y'] = df['mom'].shift(-1)  # Target is next month's value

    # Lags of mom (available at time t for predicting t+1)
    df['y_lag1'] = df['mom']           # Current month (most recent available)
    df['y_lag2'] = df['mom'].shift(1)  # Previous month
    df['y_lag3'] = df['mom'].shift(2)
    df['y_lag6'] = df['mom'].shift(5)
    df['y_lag12'] = df['mom'].shift(11)

    # Momentum (changes in known values)
    df['d_y_lag1'] = df['mom'].diff(1)
    df['d_y_lag3'] = df['mom'].diff(3)

    # Seasonality
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Calendar dummies (from Ridge Shock)
    df['is_jan'] = (df['month'] == 1).astype(int)
    df['is_jul'] = (df['month'] == 7).astype(int)
    df['is_dec'] = (df['month'] == 12).astype(int)

    return df


def add_regional_features(df, macro, quart):
    """Add regional macro features."""
    # Join macro data
    df = df.join(macro, how='left')
    df = df.join(quart, how='left')

    # Forward fill quarterly data
    for col in ['income_nom', 'income_real']:
        if col in df.columns:
            df[col] = df[col].ffill()

    # Create lagged features for regional indicators
    regional_cols = ['ppi', 'agri_prices', 'wage', 'retail', 'income_real']

    for col in regional_cols:
        if col in df.columns:
            df[f'{col}_lag1'] = df[col].shift(1)
            df[f'{col}_d1'] = df[col].diff(1)

    return df


def run_backtest(df, feature_configs, start_date='2023-12-01', end_date='2025-11-01'):
    """Run rolling backtest."""
    results = []
    test_dates = pd.date_range(start_date, end_date, freq='MS')

    for test_date in test_dates:
        if test_date not in df.index:
            continue

        actual = df.loc[test_date, 'y']
        if pd.isna(actual):
            continue

        row = {'Date': test_date, 'Actual': actual}

        for name, (features, model_class) in feature_configs.items():
            avail = [f for f in features if f in df.columns]

            if len(avail) < 3:
                row[name] = np.nan
                continue

            train_df = df[df.index < test_date][['y'] + avail].dropna()

            if len(train_df) < 24:
                row[name] = np.nan
                continue

            X_train = train_df[avail].values
            y_train = train_df['y'].values

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            test_row = df[avail].ffill().loc[[test_date]]
            if test_row.isna().any().any():
                row[name] = np.nan
                continue

            X_test_scaled = scaler.transform(test_row.values)

            model = model_class()
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)[0]

            row[name] = pred

        results.append(row)

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("ТЕСТ: Региональные макро + Production модели")
    print("=" * 70)
    print()

    # Load data
    infl, macro, quart = load_all_data()

    print(f"Инфляция: {len(infl)} точек")
    print(f"Региональные месячные: {len(macro)} точек")
    print(f"Региональные квартальные: {len(quart)} точек")
    print()

    # Create features
    df = infl[['mom']].copy()
    df = create_baseline_features(df)
    df = add_regional_features(df, macro, quart)

    print(f"Признаков всего: {len(df.columns)}")
    print()

    # Define feature sets
    baseline_features = [
        'y_lag1', 'y_lag2', 'y_lag3', 'y_lag6', 'y_lag12',
        'd_y_lag1', 'd_y_lag3',
        'month_sin', 'month_cos',
        'is_jan', 'is_jul', 'is_dec',
    ]

    # Regional features to add
    regional_features = [
        'ppi_lag1', 'ppi_d1',           # Producer prices
        'agri_prices_lag1', 'agri_prices_d1',  # Agricultural prices
        'wage_lag1', 'wage_d1',         # Wages
    ]

    minimal_regional = [
        'ppi_lag1',                     # Only PPI
        'agri_prices_lag1',             # Agricultural prices
    ]

    feature_configs = {
        'Huber Baseline': (baseline_features, HuberRegressor),
        'Huber + Regional': (baseline_features + regional_features, HuberRegressor),
        'Huber + Minimal': (baseline_features + minimal_regional, HuberRegressor),
        'Ridge Baseline': (baseline_features, Ridge),
        'Ridge + Regional': (baseline_features + regional_features, Ridge),
    }

    # Run backtest
    print("Запуск бэктеста (дек 2023 — ноя 2025)...")
    print("-" * 70)

    results = run_backtest(df, feature_configs)

    # Calculate metrics
    print()
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 70)
    print(f"{'Модель':<25} {'MAE':>8} {'vs Huber':>10} {'KPI':>10}")
    print("-" * 70)

    baseline_mae = None
    metrics = []

    for name in feature_configs.keys():
        if name not in results.columns:
            continue

        valid = results[[name, 'Actual']].dropna()
        if len(valid) == 0:
            continue

        errors = (valid[name] - valid['Actual']).abs()
        mae = errors.mean()
        kpi = (errors <= 0.5).sum()
        total = len(valid)

        if baseline_mae is None:
            baseline_mae = mae

        vs = (mae - baseline_mae) / baseline_mae * 100

        print(f"{name:<25} {mae:>8.3f} {vs:>+9.1f}% {kpi:>5}/{total}")

        metrics.append({'Model': name, 'MAE': mae, 'KPI': kpi, 'Total': total})

    print("=" * 70)

    # Show monthly details for best regional model
    print()
    print("ДЕТАЛИ: Huber + Regional")
    print("-" * 70)

    if 'Huber + Regional' in results.columns and 'Huber Baseline' in results.columns:
        for _, row in results.tail(12).iterrows():
            if pd.notna(row.get('Huber + Regional')) and pd.notna(row.get('Huber Baseline')):
                actual = row['Actual']
                pred_base = row['Huber Baseline']
                pred_reg = row['Huber + Regional']
                err_base = abs(pred_base - actual)
                err_reg = abs(pred_reg - actual)
                better = "✓" if err_reg < err_base else "—"
                print(f"  {row['Date'].strftime('%Y-%m')}: факт {actual:.2f} | Base {pred_base:.2f} (err {err_base:.2f}) | Regional {pred_reg:.2f} (err {err_reg:.2f}) {better}")

    # Summary
    print()
    if metrics:
        best = min(metrics, key=lambda x: x['MAE'])
        worst = max(metrics, key=lambda x: x['MAE'])

        print(f"Лучшая модель: {best['Model']} (MAE {best['MAE']:.3f})")
        print(f"Худшая модель: {worst['Model']} (MAE {worst['MAE']:.3f})")

        # Check if regional features helped
        base_huber = next((m for m in metrics if m['Model'] == 'Huber Baseline'), None)
        reg_huber = next((m for m in metrics if m['Model'] == 'Huber + Regional'), None)

        if base_huber and reg_huber:
            diff = (reg_huber['MAE'] - base_huber['MAE']) / base_huber['MAE'] * 100
            if diff < 0:
                print(f"\n✓ Региональные данные УЛУЧШИЛИ Huber на {abs(diff):.1f}%")
            else:
                print(f"\n✗ Региональные данные НЕ ПОМОГЛИ Huber (+{diff:.1f}%)")

    return results, metrics


if __name__ == '__main__':
    results, metrics = main()
