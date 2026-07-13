#!/usr/bin/env python3
"""
СУБКОМПОНЕНТНАЯ МОДЕЛЬ v2
=========================
Улучшенная версия с топ-субкомпонентами и проверкой согласованности.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone

def p(msg):
    print(msg, flush=True)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'archive' / 'results' / 'research'

TEST_START = '2022-01-01'
MIN_TRAIN = 36
RANDOM_STATE = 42


def load_all_data():
    """Load subcomponent and total data."""

    # Subcomponents MoM
    sub_mom = pd.read_csv(DATA_DIR / 'raw' / 'sub_mom.csv', sep=';', decimal=',', encoding='utf-8-sig')
    sub_mom['Date'] = pd.to_datetime(sub_mom['Date'], format='%d.%m.%Y')
    sub_mom = sub_mom.set_index('Date').sort_index()
    sub_mom.index = sub_mom.index.to_period('M').to_timestamp()
    sub_mom = sub_mom[~sub_mom.index.duplicated(keep='last')]

    # Weights
    weights_df = pd.read_csv(DATA_DIR / 'raw' / 'sub_weight.csv', sep=';', decimal=',')
    weights = dict(zip(weights_df['Item_code'].astype(str), weights_df['Weight']))

    # Filter valid columns
    valid_cols = [c for c in sub_mom.columns if c in weights]
    sub_mom = sub_mom[valid_cols]

    # Справочник
    sprav = pd.read_csv(DATA_DIR / 'raw' / 'subcomp_sprav.csv', sep=';', decimal=',', encoding='utf-8-sig')

    # Total inflation
    infl = pd.read_csv(DATA_DIR / 'inflation_data.csv', sep=';', decimal=',')
    infl['Date'] = pd.to_datetime(infl['Date'])
    infl = infl.set_index('Date').sort_index()
    infl.index = infl.index.to_period('M').to_timestamp()
    infl = infl[~infl.index.duplicated(keep='last')]

    return sub_mom, weights, sprav, infl


def aggregate_subcomponents(sub_mom, weights, infl):
    """Aggregate subcomponents to verify weights sum to total."""

    p("\n" + "=" * 70)
    p("ПРОВЕРКА СОГЛАСОВАННОСТИ")
    p("=" * 70)

    # Calculate weighted sum
    total_weight = sum(weights[c] for c in sub_mom.columns if c in weights)
    p(f"  Суммарный вес: {total_weight:.4f}")

    # Create aggregated series
    agg = pd.Series(0.0, index=sub_mom.index)
    for col in sub_mom.columns:
        if col in weights:
            agg += weights[col] * sub_mom[col]

    # Normalize
    agg = agg / total_weight

    # Compare with actual total
    common_idx = agg.index.intersection(infl.index)
    agg_common = agg.loc[common_idx]
    infl_common = infl.loc[common_idx, 'mom']

    # Calculate difference
    diff = agg_common - infl_common

    p(f"  Средняя разница (agg - actual): {diff.mean():.4f}")
    p(f"  Std разницы: {diff.std():.4f}")
    p(f"  Max разница: {diff.abs().max():.4f}")

    return agg


def forecast_with_lags(target_series, horizon, train_start=None):
    """Forecast a series using lagged features."""

    df = pd.DataFrame({'y': target_series})

    # Features
    for lag in [1, 2, 3, 6, 12]:
        df[f'L{lag}'] = df['y'].shift(lag)

    df['D1'] = df['y'].diff(1)
    df['D3'] = df['y'].diff(3)
    df['MA3'] = df['y'].rolling(3).mean()
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    # Target
    df['target'] = df['y'].shift(-horizon)

    # Clean
    df = df.dropna()

    if train_start:
        df = df[df.index >= pd.to_datetime(train_start)]

    # Split
    test_start_dt = pd.to_datetime(TEST_START)
    train_idx = df.index < test_start_dt
    test_idx = df.index >= test_start_dt

    if train_idx.sum() < MIN_TRAIN or test_idx.sum() < 6:
        return None, None, None

    # Backtest
    predictions = []
    actuals = []
    dates = []
    test_dates = df.index[test_idx]

    feature_cols = [c for c in df.columns if c not in ['target', 'y']]

    for test_date in test_dates:
        train_mask = df.index < test_date

        if train_mask.sum() < MIN_TRAIN:
            continue

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, 'target'].values

        X_test = df.loc[[test_date], feature_cols].values
        y_test = df.loc[test_date, 'target']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = VotingRegressor([
            ('ridge', Ridge(alpha=100.0, random_state=RANDOM_STATE)),
            ('lasso', Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=5000)),
        ])

        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)[0]
        predictions.append(pred)
        actuals.append(y_test)
        dates.append(test_date + pd.DateOffset(months=horizon))

    return np.array(predictions), np.array(actuals), dates


def run_top_subcomponents(n_top=10, horizon=1, train_start=None):
    """Run model using only top N subcomponents by weight."""

    p(f"\n{'='*70}")
    p(f"ТОП-{n_top} СУБКОМПОНЕНТОВ h={horizon}")
    p("=" * 70)

    sub_mom, weights, sprav, infl = load_all_data()

    # Sort by weight
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    top_cols = [c for c, w in sorted_weights[:n_top] if c in sub_mom.columns]

    p(f"  Топ субкомпоненты: {top_cols}")

    # Forecast each
    all_forecasts = {}
    all_actuals = {}

    for col in top_cols:
        preds, acts, dates = forecast_with_lags(sub_mom[col], horizon, train_start)
        if preds is not None:
            all_forecasts[col] = preds
            all_actuals[col] = acts

    p(f"  Успешно прогнозировано: {len(all_forecasts)}")

    if len(all_forecasts) < 3:
        return None

    # Aggregate
    min_len = min(len(v) for v in all_forecasts.values())
    total_weight = sum(weights[c] for c in all_forecasts.keys())

    agg_forecast = np.zeros(min_len)
    agg_actual = np.zeros(min_len)

    for col in all_forecasts.keys():
        w = weights[col] / total_weight
        agg_forecast += w * all_forecasts[col][-min_len:]
        agg_actual += w * all_actuals[col][-min_len:]

    mae = mean_absolute_error(agg_actual, agg_forecast)
    kpi = np.sum(np.abs(agg_forecast - agg_actual) <= 0.5)

    p(f"  MAE: {mae:.3f}, KPI: {kpi}/{min_len}")

    return {'mae': mae, 'kpi': kpi, 'total': min_len, 'n_top': n_top}


def run_component_level(horizon=1, train_start=None):
    """Run model at component level (3 components)."""

    p(f"\n{'='*70}")
    p(f"КОМПОНЕНТНЫЙ УРОВЕНЬ (3 компонента) h={horizon}")
    p("=" * 70)

    sub_mom, weights, sprav, infl = load_all_data()

    # Group by component
    component_map = dict(zip(sprav['Item_code'].astype(str), sprav['Компонент']))

    components = {}
    component_weights = {}

    for comp in ['Продовольственные товары', 'Непродовольственные товары', 'Услуги']:
        cols = [c for c in sub_mom.columns if c in component_map and component_map[c] == comp]
        if not cols:
            continue

        # Weighted average within component
        comp_series = pd.Series(0.0, index=sub_mom.index)
        comp_weight = 0

        for col in cols:
            if col in weights:
                comp_series += weights[col] * sub_mom[col]
                comp_weight += weights[col]

        comp_series = comp_series / comp_weight
        components[comp] = comp_series
        component_weights[comp] = comp_weight

    p(f"  Компонентов: {list(components.keys())}")
    p(f"  Веса: {component_weights}")

    # Forecast each component
    all_forecasts = {}
    all_actuals = {}

    for comp, series in components.items():
        preds, acts, dates = forecast_with_lags(series, horizon, train_start)
        if preds is not None:
            all_forecasts[comp] = preds
            all_actuals[comp] = acts
            p(f"    {comp[:20]}: прогноз OK")

    # Aggregate
    min_len = min(len(v) for v in all_forecasts.values())
    total_weight = sum(component_weights[c] for c in all_forecasts.keys())

    agg_forecast = np.zeros(min_len)
    agg_actual = np.zeros(min_len)

    for comp in all_forecasts.keys():
        w = component_weights[comp] / total_weight
        agg_forecast += w * all_forecasts[comp][-min_len:]
        agg_actual += w * all_actuals[comp][-min_len:]

    mae = mean_absolute_error(agg_actual, agg_forecast)
    kpi = np.sum(np.abs(agg_forecast - agg_actual) <= 0.5)

    p(f"\n  Агрегированный MAE: {mae:.3f}, KPI: {kpi}/{min_len}")

    return {'mae': mae, 'kpi': kpi, 'total': min_len}


def main():
    """Run all experiments."""

    start_time = datetime.now()
    p(f"\nНачало: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load and check data
    sub_mom, weights, sprav, infl = load_all_data()
    agg = aggregate_subcomponents(sub_mom, weights, infl)

    results = []

    # Test component-level aggregation
    for horizon in [1, 2, 12]:
        for train_start in [None, '2016-01-01']:
            result = run_component_level(horizon, train_start)
            if result:
                results.append({
                    'Model': 'Component_Level',
                    'Horizon': f'h={horizon}',
                    'Train_Start': train_start or '2010',
                    'MAE': result['mae'],
                    'KPI': f"{result['kpi']}/{result['total']}"
                })

    # Test top-N subcomponents
    for n_top in [5, 10, 20]:
        for horizon in [1, 12]:
            result = run_top_subcomponents(n_top, horizon, '2016-01-01')
            if result:
                results.append({
                    'Model': f'Top{n_top}_Subcomp',
                    'Horizon': f'h={horizon}',
                    'Train_Start': '2016',
                    'MAE': result['mae'],
                    'KPI': f"{result['kpi']}/{result['total']}"
                })

    # Summary
    p("\n" + "=" * 70)
    p("ИТОГОВАЯ СВОДКА")
    p("=" * 70)

    results_df = pd.DataFrame(results)
    p("\n" + results_df.to_string(index=False))

    # Baseline comparison
    p("\n  Baseline (прямой прогноз Total):")
    for horizon in [1, 2, 12]:
        preds, acts, _ = forecast_with_lags(infl['mom'], horizon, '2016-01-01')
        if preds is not None:
            mae = mean_absolute_error(acts, preds)
            kpi = np.sum(np.abs(preds - acts) <= 0.5)
            p(f"    h={horizon}: MAE={mae:.3f}, KPI={kpi}/{len(preds)}")

    # Save
    results_df.to_csv(RESULTS_DIR / 'subcomponent_v2_results.csv', index=False)

    end_time = datetime.now()
    p(f"\n  Время: {end_time - start_time}")


if __name__ == '__main__':
    main()
