#!/usr/bin/env python3
"""
СУБКОМПОНЕНТНАЯ МОДЕЛЬ
======================
Прогноз общей инфляции через прогнозы 45 субкомпонентов.
Bottom-up подход: прогнозируем каждый субкомпонент, агрегируем с весами.

Автор: Claude Code
Дата: 2025-12-28
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
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Constants
TEST_START = '2022-01-01'
MIN_TRAIN = 36
RANDOM_STATE = 42


def load_subcomponent_data():
    """Load subcomponent data with weights."""
    p("=" * 70)
    p("ЗАГРУЗКА СУБКОМПОНЕНТОВ")
    p("=" * 70)

    # Load MoM data for subcomponents
    sub_mom = pd.read_csv(DATA_DIR / 'raw' / 'sub_mom.csv', sep=';', decimal=',', encoding='utf-8-sig')
    sub_mom['Date'] = pd.to_datetime(sub_mom['Date'], format='%d.%m.%Y')
    sub_mom = sub_mom.set_index('Date').sort_index()

    # Load weights
    weights_df = pd.read_csv(DATA_DIR / 'raw' / 'sub_weight.csv', sep=';', decimal=',')
    weights = dict(zip(weights_df['Item_code'].astype(str), weights_df['Weight']))

    # Load справочник
    sprav = pd.read_csv(DATA_DIR / 'raw' / 'subcomp_sprav.csv', sep=';', decimal=',', encoding='utf-8-sig')

    # Filter columns that have weights
    valid_cols = [c for c in sub_mom.columns if c in weights]
    sub_mom = sub_mom[valid_cols]

    # Normalize index to month start
    sub_mom.index = sub_mom.index.to_period('M').to_timestamp()
    sub_mom = sub_mom[~sub_mom.index.duplicated(keep='last')]

    p(f"  Субкомпонентов: {len(sub_mom.columns)}")
    p(f"  Точек: {len(sub_mom)}")
    p(f"  Период: {sub_mom.index.min()} — {sub_mom.index.max()}")

    # Load actual total for comparison
    infl = pd.read_csv(DATA_DIR / 'inflation_data.csv', sep=';', decimal=',')
    infl['Date'] = pd.to_datetime(infl['Date'])
    infl = infl.set_index('Date').sort_index()
    infl.index = infl.index.to_period('M').to_timestamp()
    infl = infl[~infl.index.duplicated(keep='last')]

    return sub_mom, weights, sprav, infl


def create_features_for_subcomponent(df, col):
    """Create features for a single subcomponent."""
    result = pd.DataFrame(index=df.index)

    s = df[col]

    # Lags
    for lag in [1, 2, 3, 6, 12]:
        result[f'L{lag}'] = s.shift(lag)

    # Differences
    for d in [1, 3]:
        result[f'D{d}'] = s.diff(d)

    # Moving averages
    result['MA3'] = s.rolling(3).mean()
    result['MA6'] = s.rolling(6).mean()

    # Seasonality
    result['month_sin'] = np.sin(2 * np.pi * result.index.month / 12)
    result['month_cos'] = np.cos(2 * np.pi * result.index.month / 12)

    return result


def forecast_subcomponent(df, col, target_shift, train_start=None):
    """Forecast a single subcomponent using Voting ensemble."""

    # Create features
    features_df = create_features_for_subcomponent(df, col)

    # Create target
    target = df[col].shift(-target_shift)

    # Merge
    data = features_df.copy()
    data['target'] = target

    # Clean
    data = data.dropna()

    if len(data) < MIN_TRAIN + 12:
        return None, None

    # Train/test split
    test_start_dt = pd.to_datetime(TEST_START)
    if train_start:
        train_start_dt = pd.to_datetime(train_start)
        data = data[data.index >= train_start_dt]

    train_idx = data.index < test_start_dt
    test_idx = data.index >= test_start_dt

    if train_idx.sum() < MIN_TRAIN or test_idx.sum() < 6:
        return None, None

    # Expanding window backtest
    predictions = []
    actuals = []
    test_dates = data.index[test_idx]

    feature_cols = [c for c in data.columns if c != 'target']

    for test_date in test_dates:
        train_mask = data.index < test_date

        if train_mask.sum() < MIN_TRAIN:
            continue

        X_train = data.loc[train_mask, feature_cols].values
        y_train = data.loc[train_mask, 'target'].values

        X_test = data.loc[[test_date], feature_cols].values
        y_test = data.loc[test_date, 'target']

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model: Voting ensemble
        model = VotingRegressor([
            ('ridge', Ridge(alpha=100.0, random_state=RANDOM_STATE)),
            ('lasso', Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=5000)),
            ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000)),
        ])

        try:
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)[0]
            predictions.append(pred)
            actuals.append(y_test)
        except Exception:
            continue

    if len(predictions) < 6:
        return None, None

    return np.array(predictions), np.array(actuals)


def run_subcomponent_model(horizon=1, train_start=None):
    """Run bottom-up forecast using all subcomponents."""

    p(f"\n{'='*70}")
    p(f"СУБКОМПОНЕНТНАЯ МОДЕЛЬ h={horizon}")
    if train_start:
        p(f"Период обучения: с {train_start}")
    p("=" * 70)

    # Load data
    sub_mom, weights, sprav, infl = load_subcomponent_data()

    # Forecast each subcomponent
    all_forecasts = {}
    all_actuals = {}
    successful = 0
    failed = 0

    p(f"\n  Прогнозирование {len(sub_mom.columns)} субкомпонентов...")

    for col in sub_mom.columns:
        preds, acts = forecast_subcomponent(sub_mom, col, horizon, train_start)

        if preds is not None:
            all_forecasts[col] = preds
            all_actuals[col] = acts
            successful += 1
        else:
            failed += 1

    p(f"  Успешно: {successful}, Пропущено: {failed}")

    if successful < 10:
        p("  ОШИБКА: Слишком мало субкомпонентов с данными")
        return None

    # Aggregate forecasts with weights
    # Find common length
    min_len = min(len(v) for v in all_forecasts.values())

    # Calculate weighted sum
    total_weight = sum(weights[c] for c in all_forecasts.keys())
    p(f"  Суммарный вес: {total_weight:.4f}")

    # Normalize weights
    norm_weights = {c: weights[c] / total_weight for c in all_forecasts.keys()}

    # Weighted forecast
    agg_forecast = np.zeros(min_len)
    agg_actual = np.zeros(min_len)

    for col in all_forecasts.keys():
        agg_forecast += norm_weights[col] * all_forecasts[col][-min_len:]
        agg_actual += norm_weights[col] * all_actuals[col][-min_len:]

    # Compare with direct total forecast
    p("\n  Сравнение:")
    p(f"  {'Метрика':<30} {'Субкомп.':<15} {'Прямой прогноз':<15}")
    p("  " + "-" * 60)

    mae_subcomp = mean_absolute_error(agg_actual, agg_forecast)
    kpi_subcomp = np.sum(np.abs(agg_forecast - agg_actual) <= 0.5)

    p(f"  {'MAE':<30} {mae_subcomp:<15.3f}")
    p(f"  {'KPI hits':<30} {kpi_subcomp}/{min_len}")

    # Compare with actual total inflation
    # Get actual total from inflation_data.csv
    test_dates = pd.date_range(start=TEST_START, periods=min_len, freq='MS')

    # Shift for horizon
    actual_total_dates = test_dates + pd.DateOffset(months=horizon)
    actual_total = infl.loc[actual_total_dates, 'mom'].values if all(d in infl.index for d in actual_total_dates) else None

    if actual_total is not None:
        mae_vs_total = mean_absolute_error(actual_total, agg_forecast)
        p(f"  {'MAE vs реальный Total':<30} {mae_vs_total:<15.3f}")

    return {
        'horizon': horizon,
        'train_start': train_start,
        'mae': mae_subcomp,
        'kpi': kpi_subcomp,
        'total': min_len,
        'n_subcomponents': successful
    }


def run_baseline_comparison(horizon=1, train_start=None):
    """Run direct forecast for comparison."""

    p(f"\n{'='*70}")
    p(f"BASELINE: Прямой прогноз Total h={horizon}")
    p("=" * 70)

    # Load total inflation
    infl = pd.read_csv(DATA_DIR / 'inflation_data.csv', sep=';', decimal=',')
    infl['Date'] = pd.to_datetime(infl['Date'])
    infl = infl.set_index('Date').sort_index()
    infl.index = infl.index.to_period('M').to_timestamp()
    infl = infl[~infl.index.duplicated(keep='last')]

    # Create features
    df = infl[['mom']].copy()

    for lag in [1, 2, 3, 6, 12]:
        df[f'mom_L{lag}'] = df['mom'].shift(lag)

    for d in [1, 3]:
        df[f'mom_D{d}'] = df['mom'].diff(d)

    df['MA3'] = df['mom'].rolling(3).mean()
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    # Target
    df['target'] = df['mom'].shift(-horizon)

    # Clean
    df = df.dropna()

    if train_start:
        df = df[df.index >= pd.to_datetime(train_start)]

    # Split
    test_start_dt = pd.to_datetime(TEST_START)
    train_idx = df.index < test_start_dt
    test_idx = df.index >= test_start_dt

    # Backtest
    predictions = []
    actuals = []
    test_dates = df.index[test_idx]

    feature_cols = [c for c in df.columns if c not in ['target', 'mom']]

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
            ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000)),
        ])

        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)[0]
        predictions.append(pred)
        actuals.append(y_test)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae = mean_absolute_error(actuals, predictions)
    kpi = np.sum(np.abs(predictions - actuals) <= 0.5)

    p(f"  MAE: {mae:.3f}")
    p(f"  KPI: {kpi}/{len(predictions)}")

    return {
        'horizon': horizon,
        'train_start': train_start,
        'mae': mae,
        'kpi': kpi,
        'total': len(predictions)
    }


def main():
    """Main experiment: compare subcomponent vs direct forecast."""

    start_time = datetime.now()
    p(f"\nНачало: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    # Test different training periods and horizons
    train_starts = [None, '2016-01-01']  # From 2010 vs from 2016
    horizons = [1, 2, 12]

    for train_start in train_starts:
        for horizon in horizons:
            p(f"\n{'#'*70}")
            p(f"# ТЕСТ: h={horizon}, train_start={train_start or '2010'}")
            p(f"{'#'*70}")

            # Subcomponent model
            sub_result = run_subcomponent_model(horizon, train_start)

            # Baseline
            base_result = run_baseline_comparison(horizon, train_start)

            if sub_result and base_result:
                improvement = (base_result['mae'] - sub_result['mae']) / base_result['mae'] * 100

                results.append({
                    'Train_Start': train_start or '2010',
                    'Horizon': f'h={horizon}',
                    'Subcomp_MAE': sub_result['mae'],
                    'Baseline_MAE': base_result['mae'],
                    'Improvement': improvement,
                    'Subcomp_KPI': f"{sub_result['kpi']}/{sub_result['total']}",
                    'Baseline_KPI': f"{base_result['kpi']}/{base_result['total']}"
                })

                p(f"\n  РЕЗУЛЬТАТ: Субкомп. MAE={sub_result['mae']:.3f} vs Baseline MAE={base_result['mae']:.3f}")
                p(f"             Улучшение: {improvement:+.1f}%")

    # Summary
    p("\n" + "=" * 70)
    p("ИТОГОВАЯ СВОДКА")
    p("=" * 70)

    results_df = pd.DataFrame(results)
    p("\n" + results_df.to_string(index=False))

    # Save
    results_df.to_csv(RESULTS_DIR / 'subcomponent_comparison.csv', index=False)
    p(f"\n  Результаты сохранены: {RESULTS_DIR / 'subcomponent_comparison.csv'}")

    end_time = datetime.now()
    p(f"\n  Время выполнения: {end_time - start_time}")


if __name__ == '__main__':
    main()
