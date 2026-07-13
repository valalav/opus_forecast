#!/usr/bin/env python3
"""
СУБКОМПОНЕНТНАЯ МОДЕЛЬ — ПРАВИЛЬНОЕ СРАВНЕНИЕ
=============================================
Прогнозируем 45 субкомпонентов, агрегируем, сравниваем с официальным Total.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

def p(msg):
    print(msg, flush=True)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'archive' / 'results' / 'research'

TEST_START = '2022-01-01'
MIN_TRAIN = 24
RANDOM_STATE = 42


def load_data():
    """Load all data."""
    # Subcomponents (in % format: 2.08 means +2.08%)
    sub = pd.read_csv(DATA_DIR / 'raw' / 'sub_mom.csv', sep=';', decimal=',', encoding='utf-8-sig')
    sub['Date'] = pd.to_datetime(sub['Date'], format='%d.%m.%Y')
    sub = sub.set_index('Date').sort_index()
    sub.index = sub.index.to_period('M').to_timestamp()
    sub = sub[~sub.index.duplicated(keep='last')]

    # Weights
    weights_df = pd.read_csv(DATA_DIR / 'raw' / 'sub_weight.csv', sep=';', decimal=',')
    weights = dict(zip(weights_df['Item_code'].astype(str), weights_df['Weight']))

    # Filter valid columns
    valid_cols = [c for c in sub.columns if c in weights]
    sub = sub[valid_cols]

    # Official Total (convert from index to %)
    infl = pd.read_csv(DATA_DIR / 'inflation_data.csv', sep=';', decimal=',')
    infl['Date'] = pd.to_datetime(infl['Date'])
    infl = infl.set_index('Date').sort_index()
    infl.index = infl.index.to_period('M').to_timestamp()
    infl = infl[~infl.index.duplicated(keep='last')]
    infl['mom_pct'] = infl['mom'] - 100  # Convert to %

    return sub, weights, infl


def forecast_subcomponent(series, horizon, train_start=None):
    """Forecast a single subcomponent."""
    df = pd.DataFrame({'y': series})

    # Features
    for lag in [1, 2, 3, 6, 12]:
        df[f'L{lag}'] = df['y'].shift(lag)
    df['D1'] = df['y'].diff(1)
    df['MA3'] = df['y'].rolling(3).mean()
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    # Target
    df['target'] = df['y'].shift(-horizon)
    df = df.dropna()

    if train_start:
        df = df[df.index >= pd.to_datetime(train_start)]

    if len(df) < MIN_TRAIN + 10:
        return None, None

    # Expanding window backtest
    test_dt = pd.to_datetime(TEST_START)
    feature_cols = [c for c in df.columns if c not in ['target', 'y']]

    preds, dates = [], []

    for test_date in df.index[df.index >= test_dt]:
        train_mask = df.index < test_date
        if train_mask.sum() < MIN_TRAIN:
            continue

        X_tr = df.loc[train_mask, feature_cols].values
        y_tr = df.loc[train_mask, 'target'].values
        X_te = df.loc[[test_date], feature_cols].values

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = VotingRegressor([
            ('ridge', Ridge(alpha=100.0, random_state=RANDOM_STATE)),
            ('lasso', Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=5000)),
        ])
        model.fit(X_tr_s, y_tr)
        pred = model.predict(X_te_s)[0]

        preds.append(pred)
        # Target date is test_date + horizon months
        target_date = test_date + pd.DateOffset(months=horizon)
        dates.append(target_date)

    if len(preds) < 6:
        return None, None

    return pd.Series(preds, index=dates), dates


def run_subcomponent_forecast(horizon=1, train_start=None):
    """Forecast all subcomponents and aggregate."""
    p(f"\n{'='*70}")
    p(f"СУБКОМПОНЕНТНАЯ МОДЕЛЬ h={horizon}, train={train_start or 'Full'}")
    p("=" * 70)

    sub, weights, infl = load_data()

    p(f"  Субкомпонентов: {len(sub.columns)}")

    # Forecast each subcomponent
    forecasts = {}
    success = 0

    for col in sub.columns:
        pred_series, dates = forecast_subcomponent(sub[col], horizon, train_start)
        if pred_series is not None:
            forecasts[col] = pred_series
            success += 1

    p(f"  Успешно прогнозировано: {success}/{len(sub.columns)}")

    if success < 10:
        return None

    # Find common dates across all forecasts
    common_dates = forecasts[list(forecasts.keys())[0]].index
    for col in forecasts:
        common_dates = common_dates.intersection(forecasts[col].index)

    p(f"  Общих дат прогноза: {len(common_dates)}")

    # Aggregate forecasts with weights
    total_weight = sum(weights[c] for c in forecasts.keys())
    agg_forecast = pd.Series(0.0, index=common_dates)

    for col in forecasts.keys():
        w = weights[col] / total_weight
        agg_forecast += w * forecasts[col].loc[common_dates]

    # Get actual Total for these dates
    actual_dates = common_dates.intersection(infl.index)
    if len(actual_dates) < 6:
        p("  Недостаточно дат для сравнения")
        return None

    agg_forecast = agg_forecast.loc[actual_dates]
    actual_total = infl.loc[actual_dates, 'mom_pct']

    # Calculate metrics
    mae = mean_absolute_error(actual_total, agg_forecast)
    kpi = np.sum(np.abs(agg_forecast - actual_total) <= 0.5)

    p(f"\n  Агрегированный прогноз vs официальный Total:")
    p(f"  MAE: {mae:.3f}")
    p(f"  KPI: {kpi}/{len(actual_dates)} ({100*kpi/len(actual_dates):.0f}%)")

    # Show last predictions
    p(f"\n  Последние прогнозы:")
    comparison = pd.DataFrame({
        'Прогноз': agg_forecast.round(2),
        'Факт': actual_total.round(2),
        'Ошибка': (agg_forecast - actual_total).round(3)
    }).tail(6)
    p(comparison.to_string())

    return {
        'horizon': horizon,
        'train_start': train_start or 'Full',
        'mae': mae,
        'kpi': kpi,
        'total': len(actual_dates),
        'n_subcomp': success
    }


def run_direct_forecast(horizon=1, train_start=None):
    """Direct forecast of Total for comparison."""
    _, _, infl = load_data()

    df = pd.DataFrame({'y': infl['mom_pct']})

    for lag in [1, 2, 3, 6, 12]:
        df[f'L{lag}'] = df['y'].shift(lag)
    df['D1'] = df['y'].diff(1)
    df['MA3'] = df['y'].rolling(3).mean()
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    df['target'] = df['y'].shift(-horizon)
    df = df.dropna()

    if train_start:
        df = df[df.index >= pd.to_datetime(train_start)]

    test_dt = pd.to_datetime(TEST_START)
    feature_cols = [c for c in df.columns if c not in ['target', 'y']]

    preds, acts = [], []

    for test_date in df.index[df.index >= test_dt]:
        train_mask = df.index < test_date
        if train_mask.sum() < MIN_TRAIN:
            continue

        X_tr = df.loc[train_mask, feature_cols].values
        y_tr = df.loc[train_mask, 'target'].values
        X_te = df.loc[[test_date], feature_cols].values
        y_te = df.loc[test_date, 'target']

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = VotingRegressor([
            ('ridge', Ridge(alpha=100.0, random_state=RANDOM_STATE)),
            ('lasso', Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=5000)),
        ])
        model.fit(X_tr_s, y_tr)
        pred = model.predict(X_te_s)[0]

        preds.append(pred)
        acts.append(y_te)

    mae = mean_absolute_error(acts, preds)
    kpi = np.sum(np.abs(np.array(preds) - np.array(acts)) <= 0.5)

    return {'mae': mae, 'kpi': kpi, 'total': len(preds)}


def main():
    start_time = datetime.now()
    p(f"\nНачало: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    for horizon in [1, 2, 12]:
        for train_start in [None, '2016-01-01']:
            p(f"\n{'#'*70}")
            p(f"# ТЕСТ: h={horizon}, train_start={train_start or 'Full'}")
            p(f"{'#'*70}")

            # Subcomponent model
            sub_result = run_subcomponent_forecast(horizon, train_start)

            # Direct baseline
            p(f"\n  BASELINE (прямой прогноз Total):")
            direct = run_direct_forecast(horizon, train_start)
            p(f"  MAE: {direct['mae']:.3f}, KPI: {direct['kpi']}/{direct['total']}")

            if sub_result:
                improvement = (direct['mae'] - sub_result['mae']) / direct['mae'] * 100
                p(f"\n  >>> УЛУЧШЕНИЕ: {improvement:+.1f}%")

                results.append({
                    'Horizon': f'h={horizon}',
                    'Train': train_start or 'Full',
                    'Subcomp_MAE': round(sub_result['mae'], 3),
                    'Direct_MAE': round(direct['mae'], 3),
                    'Improvement': f"{improvement:+.1f}%",
                    'Subcomp_KPI': f"{sub_result['kpi']}/{sub_result['total']}",
                    'Direct_KPI': f"{direct['kpi']}/{direct['total']}"
                })

    # Summary
    p("\n" + "=" * 70)
    p("ИТОГОВАЯ СВОДКА")
    p("=" * 70)

    df = pd.DataFrame(results)
    p("\n" + df.to_string(index=False))

    # Save
    df.to_csv(RESULTS_DIR / 'subcomponent_correct_results.csv', index=False)
    p(f"\n  Сохранено: {RESULTS_DIR / 'subcomponent_correct_results.csv'}")

    p(f"\n  Время: {datetime.now() - start_time}")


if __name__ == '__main__':
    main()
