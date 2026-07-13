#!/usr/bin/env python3
"""
Тест региональных макропоказателей КБР для прогнозирования инфляции.

Гипотеза: локальные показатели (цены производителей, зарплаты, розница)
могут улучшить прогноз инфляции КБР по сравнению с федеральными.

Автор: Claude Code
Дата: 2025-12-28
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'


def load_inflation_data():
    """Load KBR inflation data."""
    df = pd.read_csv(DATA_DIR / 'inflation_data.csv', sep=';', decimal=',', on_bad_lines='skip')
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
    # Normalize to start of month for joining
    df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
    df = df.set_index('Date')
    # mom column is the target (MoM inflation index, base 100)
    return df


def load_regional_monthly():
    """Load regional monthly macro data."""
    df = pd.read_csv(DATA_DIR / 'month.csv', sep=';', decimal=',')
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y')
    df = df.set_index('Date')

    # Rename columns based on spravka
    col_names = {
        '1': 'ind_prod',       # Индекс промпроизводства
        '2': 'shipped_goods',  # Отгружено товаров
        '3': 'construction',   # Строительство
        '6': 'retail',         # Розничный оборот
        '7': 'services',       # Платные услуги
        '8': 'profit',         # Прибыль организаций
        '9': 'payables',       # Кредиторская задолженность
        '10': 'receivables',   # Дебиторская задолженность
        '11': 'ppi',           # Цены производителей промтоваров
        '12': 'agri_prices',   # Цены с/х продукции
        '13': 'invest_prices', # Цены инвестиционных товаров
        '18': 'wage',          # Номинальная зарплата
        '19': 'wage_agri',     # Зарплата в с/х
    }

    # Select and rename only existing columns
    existing = [c for c in col_names.keys() if c in df.columns]
    df = df[existing].rename(columns={k: col_names[k] for k in existing})

    # Drop empty columns
    df = df.dropna(axis=1, how='all')

    return df


def load_regional_quarterly():
    """Load regional quarterly data."""
    df = pd.read_csv(DATA_DIR / 'quart.csv', sep=';', decimal=',')
    df['Date'] = pd.to_datetime(df['Data'], format='%d.%m.%Y', errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.set_index('Date')

    col_names = {
        '14': 'housing_primary',   # Цены первичного жилья
        '15': 'housing_secondary', # Цены вторичного жилья
        '16': 'income_nominal',    # Доходы населения (номинальные)
        '17': 'income_real',       # Доходы (реальные)
        '20': 'unknown_20',        # Неизвестный показатель
    }

    existing = [c for c in col_names.keys() if c in df.columns]
    df = df[existing].rename(columns={k: col_names[k] for k in existing})
    df = df.dropna(axis=1, how='all')

    # Remove duplicates (keep last)
    df = df[~df.index.duplicated(keep='last')]

    # Resample to monthly (forward fill quarterly values)
    df = df.resample('MS').ffill()

    return df


def prepare_features(infl_df, macro_df, quarterly_df=None):
    """Merge inflation and macro data, create features."""

    # Align dates
    merged = infl_df[['mom']].copy()
    merged = merged.join(macro_df, how='left')

    if quarterly_df is not None:
        merged = merged.join(quarterly_df, how='left')

    # Forward fill NaN (quarterly data)
    merged = merged.ffill()

    # Create lagged features
    feature_cols = [c for c in merged.columns if c != 'mom']

    for col in feature_cols:
        if col in merged.columns:
            merged[f'{col}_lag1'] = merged[col].shift(1)
            merged[f'{col}_lag2'] = merged[col].shift(2)
            merged[f'{col}_lag3'] = merged[col].shift(3)
            # Changes
            merged[f'{col}_d1'] = merged[col].diff(1)
            merged[f'{col}_d3'] = merged[col].diff(3)

    # Add inflation lags (autoregressive)
    merged['mom_lag1'] = merged['mom'].shift(1)
    merged['mom_lag2'] = merged['mom'].shift(2)
    merged['mom_lag3'] = merged['mom'].shift(3)
    merged['mom_lag6'] = merged['mom'].shift(6)
    merged['mom_lag12'] = merged['mom'].shift(12)

    # Seasonality
    merged['month'] = merged.index.month
    merged['month_sin'] = np.sin(2 * np.pi * merged['month'] / 12)
    merged['month_cos'] = np.cos(2 * np.pi * merged['month'] / 12)

    return merged


def run_backtest(df, feature_sets, start_date='2023-12-01', end_date='2025-11-01'):
    """Run rolling backtest comparing feature sets."""

    results = []
    test_dates = pd.date_range(start_date, end_date, freq='MS')

    for test_date in test_dates:
        # Check if test_date exists
        if test_date not in df.index:
            continue

        actual = df.loc[test_date, 'mom']
        if pd.isna(actual):
            continue

        row = {'Date': test_date, 'Actual': actual}

        for name, features in feature_sets.items():
            # Filter to existing features
            avail_features = [f for f in features if f in df.columns]

            if len(avail_features) < 3:
                row[name] = np.nan
                continue

            # Get training data (only rows with all features available)
            train_df = df[df.index < test_date][['mom'] + avail_features].dropna()

            if len(train_df) < 24:  # Minimum training size
                row[name] = np.nan
                continue

            X_train = train_df[avail_features].values
            y_train = train_df['mom'].values

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # Get test features
            test_row = df.loc[[test_date], avail_features]

            if test_row.isna().any().any():
                # Try forward filling
                test_row = df[avail_features].ffill().loc[[test_date]]

            if test_row.isna().any().any():
                row[name] = np.nan
                continue

            X_test = test_row.values
            X_test_scaled = scaler.transform(X_test)

            # Train Ridge model
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)[0]

            row[name] = pred

        results.append(row)

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("ТЕСТ РЕГИОНАЛЬНЫХ МАКРОПОКАЗАТЕЛЕЙ КБР")
    print("=" * 70)
    print()

    # Load data
    print("Загрузка данных...")
    infl_df = load_inflation_data()
    macro_df = load_regional_monthly()
    quarterly_df = load_regional_quarterly()

    print(f"  Инфляция: {len(infl_df)} точек ({infl_df.index.min()} — {infl_df.index.max()})")
    print(f"  Месячные макро: {len(macro_df)} точек, {len(macro_df.columns)} показателей")
    print(f"  Квартальные: {len(quarterly_df)} точек, {len(quarterly_df.columns)} показателей")
    print()

    # Show available indicators
    print("Доступные месячные показатели:")
    for col in macro_df.columns:
        print(f"  - {col}")
    print()

    print("Доступные квартальные показатели:")
    for col in quarterly_df.columns:
        print(f"  - {col}")
    print()

    # Prepare features
    df = prepare_features(infl_df, macro_df, quarterly_df)
    df = df.dropna(subset=['mom'])

    print(f"Объединенный датасет: {len(df)} точек, {len(df.columns)} признаков")
    print()

    # Define feature sets to compare
    baseline_features = [
        'mom_lag1', 'mom_lag2', 'mom_lag3', 'mom_lag6', 'mom_lag12',
        'month_sin', 'month_cos'
    ]

    # Regional macro features (current and lagged)
    regional_features = baseline_features + [
        # Producer prices (leading indicator)
        'ppi', 'ppi_lag1', 'ppi_lag2', 'ppi_d1', 'ppi_d3',
        # Agricultural prices
        'agri_prices', 'agri_prices_lag1', 'agri_prices_d1',
        # Wages (cost-push)
        'wage', 'wage_lag1', 'wage_d1',
        # Retail (demand)
        'retail', 'retail_lag1', 'retail_d1',
        # Services
        'services', 'services_lag1',
        # Industrial production
        'ind_prod', 'ind_prod_lag1',
    ]

    # Minimal regional (only most important)
    minimal_regional = baseline_features + [
        'ppi_lag1', 'ppi_d1',          # Producer prices (leading)
        'wage_lag1', 'wage_d1',         # Wages (costs)
        'agri_prices_lag1',             # Agricultural prices
    ]

    # Housing + income (quarterly)
    quarterly_features = baseline_features + [
        'housing_primary', 'housing_secondary',
        'income_nominal', 'income_real',
        'income_nominal_lag1', 'income_real_lag1',
    ]

    # Full model
    full_features = baseline_features + [
        'ppi_lag1', 'ppi_d1',
        'agri_prices_lag1', 'agri_prices_d1',
        'wage_lag1', 'wage_d1',
        'retail_lag1', 'retail_d1',
        'ind_prod_lag1',
        'income_real_lag1',
    ]

    feature_sets = {
        'Baseline (AR only)': baseline_features,
        'Regional Full': regional_features,
        'Regional Minimal': minimal_regional,
        'Quarterly Only': quarterly_features,
        'Full Combined': full_features,
    }

    # Run backtest
    print("Запуск бэктеста (дек 2023 — ноя 2025)...")
    print("-" * 70)

    results = run_backtest(df, feature_sets, start_date='2023-12-01', end_date='2025-11-01')

    # Calculate metrics
    print()
    print("РЕЗУЛЬТАТЫ:")
    print("=" * 70)
    print(f"{'Модель':<25} {'MAE':>8} {'vs Base':>10} {'KPI Hits':>10}")
    print("-" * 70)

    baseline_mae = None
    metrics = []

    for name in feature_sets.keys():
        if name not in results.columns:
            continue

        valid = results[[name, 'Actual']].dropna()
        if len(valid) == 0:
            continue

        errors = (valid[name] - valid['Actual']).abs()
        mae = errors.mean()
        kpi_hits = (errors <= 0.5).sum()
        total = len(valid)

        if baseline_mae is None:
            baseline_mae = mae

        vs_base = (mae - baseline_mae) / baseline_mae * 100 if baseline_mae else 0

        print(f"{name:<25} {mae:>8.3f} {vs_base:>+9.1f}% {kpi_hits:>5}/{total}")

        metrics.append({
            'Model': name,
            'MAE': mae,
            'vs_Baseline': vs_base,
            'KPI_Hits': kpi_hits,
            'Total': total
        })

    print("=" * 70)

    # Best model
    if metrics:
        best = min(metrics, key=lambda x: x['MAE'])
        print(f"\nЛучшая модель: {best['Model']}")
        print(f"MAE: {best['MAE']:.3f}, KPI: {best['KPI_Hits']}/{best['Total']}")

    # Show monthly predictions for best regional model
    print()
    print("ДЕТАЛИЗАЦИЯ ПО МЕСЯЦАМ (Regional Minimal):")
    print("-" * 70)

    if 'Regional Minimal' in results.columns:
        for _, row in results.tail(12).iterrows():
            if pd.notna(row.get('Regional Minimal')):
                actual = row['Actual']
                pred = row['Regional Minimal']
                error = abs(pred - actual)
                kpi = "✓" if error <= 0.5 else "✗"
                print(f"  {row['Date'].strftime('%Y-%m')}: факт {actual:.2f}, прогноз {pred:.2f}, ошибка {error:.2f} {kpi}")

    # Feature importance analysis
    print()
    print("КОРРЕЛЯЦИЯ ПРИЗНАКОВ С ИНФЛЯЦИЕЙ:")
    print("-" * 70)

    target = df['mom']
    correlations = []

    for col in df.columns:
        if col == 'mom' or col == 'month':
            continue
        valid = df[[col, 'mom']].dropna()
        if len(valid) > 20:
            corr = valid[col].corr(valid['mom'])
            if not pd.isna(corr):
                correlations.append((col, corr))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Топ-15 по абсолютной корреляции:")
    for col, corr in correlations[:15]:
        print(f"  {col:<30} {corr:>+.3f}")

    return results, metrics


if __name__ == '__main__':
    results, metrics = main()
