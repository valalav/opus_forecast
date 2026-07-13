#!/usr/bin/env python3
"""
ФИНАЛЬНОЕ СРАВНЕНИЕ: СУБКОМПОНЕНТНАЯ МОДЕЛЬ
============================================
Корректное сравнение bottom-up подхода с прямым прогнозом.
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

def p(msg):
    print(msg, flush=True)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'archive' / 'results' / 'research'

TEST_START = '2022-01-01'
MIN_TRAIN = 36
RANDOM_STATE = 42


def load_data():
    """Load all data and convert to common format."""

    # Total inflation (index format: 101.49 = +1.49%)
    infl = pd.read_csv(DATA_DIR / 'inflation_data.csv', sep=';', decimal=',')
    infl['Date'] = pd.to_datetime(infl['Date'])
    infl = infl.set_index('Date').sort_index()
    infl.index = infl.index.to_period('M').to_timestamp()
    infl = infl[~infl.index.duplicated(keep='last')]

    # Convert mom from index to percent: 101.49 -> 1.49
    infl['mom_pct'] = infl['mom'] - 100

    # Subcomponents (already in percent format: 2.08 = +2.08%)
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

    return infl, sub_mom, weights, sprav


def create_component_series(sub_mom, weights, sprav):
    """Create 3 component series from subcomponents."""

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

        comp_series = comp_series / comp_weight  # Normalize within component
        components[comp] = comp_series
        component_weights[comp] = comp_weight

    return components, component_weights


def forecast_series(series, horizon, train_start=None):
    """Forecast using Voting ensemble."""

    df = pd.DataFrame({'y': series})

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
    df = df.dropna()

    if train_start:
        df = df[df.index >= pd.to_datetime(train_start)]

    # Split
    test_dt = pd.to_datetime(TEST_START)
    train_mask = df.index < test_dt
    test_mask = df.index >= test_dt

    if train_mask.sum() < MIN_TRAIN or test_mask.sum() < 6:
        return None, None, None

    # Backtest
    preds, acts, dates = [], [], []
    feature_cols = [c for c in df.columns if c not in ['target', 'y']]

    for test_date in df.index[test_mask]:
        tr = df.index < test_date
        if tr.sum() < MIN_TRAIN:
            continue

        X_tr = df.loc[tr, feature_cols].values
        y_tr = df.loc[tr, 'target'].values
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
        dates.append(test_date)

    return np.array(preds), np.array(acts), dates


def run_experiment(horizon, train_start=None):
    """Compare bottom-up vs direct forecast."""

    p(f"\n{'='*70}")
    p(f"ЭКСПЕРИМЕНТ: h={horizon}, train_start={train_start or 'Full'}")
    p("=" * 70)

    infl, sub_mom, weights, sprav = load_data()
    components, comp_weights = create_component_series(sub_mom, weights, sprav)

    # 1. Direct forecast of Total
    p("\n  1. Прямой прогноз Total...")
    direct_preds, direct_acts, direct_dates = forecast_series(infl['mom_pct'], horizon, train_start)

    if direct_preds is None:
        p("     SKIP: недостаточно данных")
        return None

    direct_mae = mean_absolute_error(direct_acts, direct_preds)
    direct_kpi = np.sum(np.abs(direct_preds - direct_acts) <= 0.5)
    p(f"     MAE: {direct_mae:.3f}, KPI: {direct_kpi}/{len(direct_preds)}")

    # 2. Bottom-up: forecast each component, then aggregate
    p("\n  2. Bottom-up (3 компонента)...")
    comp_forecasts = {}
    comp_actuals = {}

    for comp_name, comp_series in components.items():
        preds, acts, dates = forecast_series(comp_series, horizon, train_start)
        if preds is not None:
            comp_forecasts[comp_name] = preds
            comp_actuals[comp_name] = acts
            p(f"     {comp_name[:30]}: OK ({len(preds)} точек)")

    if len(comp_forecasts) < 3:
        p("     SKIP: недостаточно компонентов")
        return None

    # Aggregate with weights
    min_len = min(len(v) for v in comp_forecasts.values())
    total_weight = sum(comp_weights[c] for c in comp_forecasts.keys())

    agg_preds = np.zeros(min_len)
    agg_acts = np.zeros(min_len)

    for comp in comp_forecasts.keys():
        w = comp_weights[comp] / total_weight
        agg_preds += w * comp_forecasts[comp][-min_len:]
        agg_acts += w * comp_actuals[comp][-min_len:]

    bottomup_mae = mean_absolute_error(agg_acts, agg_preds)
    bottomup_kpi = np.sum(np.abs(agg_preds - agg_acts) <= 0.5)
    p(f"     Агрегированный MAE: {bottomup_mae:.3f}, KPI: {bottomup_kpi}/{min_len}")

    # 3. Compare: bottom-up pred vs actual Total
    # Align dates
    if len(direct_preds) >= min_len:
        aligned_direct_acts = direct_acts[-min_len:]
        mae_vs_total = mean_absolute_error(aligned_direct_acts, agg_preds)
        p(f"\n     Bottom-up vs реальный Total MAE: {mae_vs_total:.3f}")

    # Summary
    improvement = (direct_mae - bottomup_mae) / direct_mae * 100

    p(f"\n  ИТОГ:")
    p(f"     Direct MAE:    {direct_mae:.3f}")
    p(f"     Bottom-up MAE: {bottomup_mae:.3f}")
    p(f"     Улучшение:     {improvement:+.1f}%")

    return {
        'horizon': horizon,
        'train_start': train_start or 'Full',
        'direct_mae': direct_mae,
        'direct_kpi': f"{direct_kpi}/{len(direct_preds)}",
        'bottomup_mae': bottomup_mae,
        'bottomup_kpi': f"{bottomup_kpi}/{min_len}",
        'improvement': improvement
    }


def main():
    start_time = datetime.now()
    p(f"\nНачало: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    for horizon in [1, 2, 12]:
        for train_start in [None, '2016-01-01']:
            result = run_experiment(horizon, train_start)
            if result:
                results.append(result)

    # Summary table
    p("\n" + "=" * 70)
    p("ИТОГОВАЯ СВОДКА")
    p("=" * 70)

    df = pd.DataFrame(results)
    p("\n" + df.to_string(index=False))

    # Best results
    p("\n  ВЫВОДЫ:")

    for h in [1, 2, 12]:
        h_data = df[df['horizon'] == h]
        if len(h_data) > 0:
            best = h_data.loc[h_data['improvement'].idxmax()]
            if best['improvement'] > 0:
                p(f"    h={h}: Bottom-up лучше на {best['improvement']:.1f}% (train={best['train_start']})")
            else:
                p(f"    h={h}: Direct лучше (bottom-up хуже на {-best['improvement']:.1f}%)")

    # Save
    df.to_csv(RESULTS_DIR / 'bottomup_vs_direct.csv', index=False)
    p(f"\n  Сохранено: {RESULTS_DIR / 'bottomup_vs_direct.csv'}")

    p(f"\n  Время: {datetime.now() - start_time}")


if __name__ == '__main__':
    main()
