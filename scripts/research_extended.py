#!/usr/bin/env python3
"""
РАСШИРЕННОЕ ИССЛЕДОВАНИЕ
========================
Тестирование дополнительных наборов признаков и ансамблей.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def p(msg):
    print(msg, flush=True)

from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import (
    RandomForestRegressor, StackingRegressor, VotingRegressor
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone

try:
    from ngboost import NGBRegressor
    HAS_NGBOOST = True
except ImportError:
    HAS_NGBOOST = False

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'archive' / 'results' / 'research'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_START = '2022-01-01'
MIN_TRAIN = 36
RANDOM_STATE = 42


def load_data():
    """Load all data sources."""
    p("=" * 70)
    p("ЗАГРУЗКА ДАННЫХ")
    p("=" * 70)

    # Main inflation data
    infl = pd.read_csv(DATA_DIR / 'inflation_data.csv', sep=';', decimal=',')
    infl['Date'] = pd.to_datetime(infl['Date'])
    infl = infl.set_index('Date').sort_index()
    infl.index = infl.index.to_period('M').to_timestamp()
    infl = infl[~infl.index.duplicated(keep='last')]

    # Monthly regional data
    month = pd.read_csv(DATA_DIR / 'month.csv', sep=';', decimal=',', encoding='utf-8-sig')
    month['Date'] = pd.to_datetime(month['Date'], format='%d.%m.%Y')
    month = month.set_index('Date').sort_index()
    month.columns = [f'reg_{c}' if str(c).isdigit() else c for c in month.columns]
    month = month.select_dtypes(include=[np.number])

    # Quarterly data
    quart = pd.read_csv(DATA_DIR / 'quart.csv', sep=';', decimal=',', encoding='utf-8-sig')
    date_col = quart.columns[0]
    quart['Date'] = pd.to_datetime(quart[date_col], format='%d.%m.%Y')
    quart = quart.drop(columns=[date_col]).set_index('Date').sort_index()
    quart.columns = [f'q_{c}' for c in quart.columns]
    quart = quart.select_dtypes(include=[np.number])
    quart = quart[~quart.index.duplicated(keep='last')]
    monthly_idx = pd.date_range(start=quart.index.min(), end=infl.index.max(), freq='MS')
    quart = quart.reindex(monthly_idx).ffill()

    # Brent oil
    try:
        brent = pd.read_csv(DATA_DIR / 'brent_prices.csv')
        brent['Date'] = pd.to_datetime(brent['Date'])
        brent = brent.set_index('Date').sort_index()
    except:
        brent = pd.DataFrame()

    # Merge all
    df = infl.copy()
    df = df.join(month, how='left')
    df = df.join(quart, how='left')
    if not brent.empty:
        df = df.join(brent, how='left')

    p(f"  Загружено {len(df)} точек")
    return df


def create_features(df):
    """Create all feature variations."""
    result = df.copy()
    base_cols = df.columns.tolist()

    for col in base_cols:
        if df[col].isna().sum() > len(df) * 0.5:
            continue

        s = df[col]

        for lag in [1, 2, 3, 6, 12]:
            result[f'{col}_L{lag}'] = s.shift(lag)

        for d in [1, 3, 6]:
            result[f'{col}_D{d}'] = s.diff(d)

        for w in [3, 6]:
            result[f'{col}_MA{w}'] = s.rolling(w).mean()

    result['month_sin'] = np.sin(2 * np.pi * result.index.month / 12)
    result['month_cos'] = np.cos(2 * np.pi * result.index.month / 12)
    result['is_jan'] = (result.index.month == 1).astype(int)
    result['is_jul'] = (result.index.month == 7).astype(int)

    return result


def define_extended_feature_sets(df):
    """Define extended feature sets with more variations."""
    cols = df.columns.tolist()

    feature_sets = {
        # Best from previous research
        'Components': ['mom_L1', 'mom_L2', 'Prod_L1', 'Nonprod_L1', 'Serv_L1',
                      'Prod_D1', 'Nonprod_D1', 'Serv_D1', 'month_sin', 'month_cos'],

        # Components with momentum
        'Components_momentum': ['mom_L1', 'mom_L2', 'mom_D1', 'mom_D3',
                               'Prod_L1', 'Nonprod_L1', 'Serv_L1',
                               'Prod_D1', 'Nonprod_D1', 'Serv_D1',
                               'month_sin', 'month_cos', 'is_jan', 'is_jul'],

        # AR with all lags
        'AR_full': ['mom_L1', 'mom_L2', 'mom_L3', 'mom_L6', 'mom_L12',
                   'mom_D1', 'mom_D3', 'mom_D6',
                   'mom_MA3', 'mom_MA6',
                   'month_sin', 'month_cos', 'is_jan', 'is_jul'],

        # IBVED focused
        'IBVED_focus': ['mom_L1', 'mom_L2',
                       'q_20_L1', 'q_20_L2', 'q_20_MA3',
                       'month_sin', 'month_cos'],

        # Components + IBVED
        'Components_IBVED': ['mom_L1', 'mom_L2',
                            'Prod_L1', 'Nonprod_L1', 'Serv_L1',
                            'q_20_L1', 'q_20_MA3',
                            'month_sin', 'month_cos'],

        # Macro rates
        'Rates': ['mom_L1', 'mom_L2',
                 'Ki_i_L1', 'Ki_i_L6', 'Ki_i_D1', 'Ki_i_D6',
                 'Ruonia_L1', 'Ruonia_L2', 'Ruonia_D1',
                 'month_sin', 'month_cos'],

        # USD focused
        'USD_focus': ['mom_L1', 'mom_L2',
                     'usd_nom_i_L1', 'usd_nom_i_L2', 'usd_nom_i_L6',
                     'usd_nom_i_D1', 'usd_nom_i_D3',
                     'month_sin', 'month_cos'],

        # Brent oil
        'Brent_focus': ['mom_L1', 'mom_L2',
                       'brent_L1', 'brent_L3', 'brent_L6',
                       'brent_D1', 'brent_D3',
                       'month_sin', 'month_cos'],

        # Components + Rates
        'Components_Rates': ['mom_L1', 'mom_L2',
                            'Prod_L1', 'Nonprod_L1', 'Serv_L1',
                            'Ki_i_L6', 'Ruonia_D1',
                            'month_sin', 'month_cos'],

        # Full macro package
        'Full_macro': ['mom_L1', 'mom_L2', 'mom_L3',
                      'Ki_i_L1', 'Ki_i_L6', 'Ruonia_D1',
                      'usd_nom_i_L2', 'brent_L3',
                      'month_sin', 'month_cos', 'is_jan'],

        # Minimal but powerful
        'Minimal_plus': ['mom_L1', 'mom_L2', 'mom_D1',
                        'month_sin', 'month_cos', 'is_jan'],

        # Extended AR with volatility
        'AR_volatility': ['mom_L1', 'mom_L2', 'mom_L3',
                         'mom_D1', 'mom_D3',
                         'mom_MA3', 'mom_MA6',
                         'month_sin', 'month_cos'],

        # Production focused
        'Production': ['mom_L1', 'mom_L2',
                      'Prod_L1', 'Prod_L2', 'Prod_L3',
                      'Prod_D1', 'Prod_D3', 'Prod_MA3',
                      'month_sin', 'month_cos'],

        # Services focused
        'Services': ['mom_L1', 'mom_L2',
                    'Serv_L1', 'Serv_L2', 'Serv_L3',
                    'Serv_D1', 'Serv_D3', 'Serv_MA3',
                    'month_sin', 'month_cos'],

        # Ultimate combination
        'Ultimate': ['mom_L1', 'mom_L2', 'mom_D1',
                    'Prod_L1', 'Serv_L1',
                    'Ki_i_L6',
                    'q_20_L1',
                    'month_sin', 'month_cos', 'is_jan'],
    }

    # Filter to only existing columns
    for name in list(feature_sets.keys()):
        feature_sets[name] = [c for c in feature_sets[name] if c in cols]
        if len(feature_sets[name]) < 3:
            del feature_sets[name]

    return feature_sets


def define_models():
    """Define models including ensembles."""
    base_models = {
        'Lasso': Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=5000),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000),
        'Huber': HuberRegressor(max_iter=500),
        'Ridge_100': Ridge(alpha=100.0, random_state=RANDOM_STATE),
    }

    if HAS_NGBOOST:
        base_models['NGBoost'] = NGBRegressor(n_estimators=100, random_state=RANDOM_STATE, verbose=False)

    # Stacking ensemble
    estimators = [
        ('ridge', Ridge(alpha=100.0, random_state=RANDOM_STATE)),
        ('lasso', Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=5000)),
        ('huber', HuberRegressor(max_iter=500)),
    ]
    base_models['Stacking'] = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=10.0, random_state=RANDOM_STATE),
        cv=3
    )

    # Voting ensemble
    base_models['Voting'] = VotingRegressor(
        estimators=estimators
    )

    return base_models


def backtest_model(model, df, features, horizon, test_start=TEST_START):
    """Run expanding window backtest."""

    if horizon == 1:
        df = df.copy()
        df['target'] = df['mom'].shift(-1)
    elif horizon == 2:
        df = df.copy()
        df['target'] = df['mom'].shift(-2)
    else:
        df = df.copy()
        df['target'] = df['mom'].shift(-12)

    valid_features = [f for f in features if f in df.columns]
    if len(valid_features) < 2:
        return None, None, None, None

    df_clean = df[['target'] + valid_features].dropna()

    if len(df_clean) < MIN_TRAIN + 12:
        return None, None, None, None

    test_start_dt = pd.to_datetime(test_start)
    train_idx = df_clean.index < test_start_dt
    test_idx = df_clean.index >= test_start_dt

    if train_idx.sum() < MIN_TRAIN or test_idx.sum() < 6:
        return None, None, None, None

    predictions = []
    actuals = []

    test_dates = df_clean.index[test_idx]

    for test_date in test_dates:
        train_mask = df_clean.index < test_date

        if train_mask.sum() < MIN_TRAIN:
            continue

        X_train = df_clean.loc[train_mask, valid_features].values
        y_train = df_clean.loc[train_mask, 'target'].values

        X_test = df_clean.loc[[test_date], valid_features].values
        y_test = df_clean.loc[test_date, 'target']

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        try:
            m = clone(model)
            m.fit(X_train_scaled, y_train)
            pred = m.predict(X_test_scaled)[0]

            predictions.append(pred)
            actuals.append(y_test)
        except Exception:
            continue

    if len(predictions) < 6:
        return None, None, None, None

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    mae = mean_absolute_error(actuals, predictions)
    kpi_hits = np.sum(np.abs(predictions - actuals) <= 0.5)
    total = len(predictions)

    return mae, kpi_hits, total, len(valid_features)


def run_extended_research():
    """Main research loop."""

    start_time = datetime.now()
    p(f"\nНачало: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    df = load_data()
    df = create_features(df)

    feature_sets = define_extended_feature_sets(df)
    models = define_models()
    horizons = [1, 2, 12]

    p("\n" + "=" * 70)
    p("РАСШИРЕННОЕ ИССЛЕДОВАНИЕ")
    p("=" * 70)
    p(f"  Моделей: {len(models)}")
    p(f"  Наборов признаков: {len(feature_sets)}")
    p(f"  Горизонтов: {len(horizons)}")
    p(f"  Всего комбинаций: {len(models) * len(feature_sets) * len(horizons)}")

    all_results = []
    total_combos = len(models) * len(feature_sets) * len(horizons)
    combo_num = 0

    for horizon in horizons:
        p(f"\n  === ГОРИЗОНТ h={horizon} ===")

        for model_name, model in models.items():
            for fs_name, features in feature_sets.items():
                combo_num += 1

                mae, kpi, total, n_feat = backtest_model(model, df, features, horizon)

                if mae is not None:
                    all_results.append({
                        'Horizon': f'h={horizon}',
                        'Model': model_name,
                        'FeatureSet': fs_name,
                        'MAE': mae,
                        'KPI_Hits': kpi,
                        'Total': total,
                        'N_Features': n_feat
                    })

                    status = "✓" if mae < 0.5 else "○"
                    p(f"    [{combo_num}/{total_combos}] {status} {model_name:12} + {fs_name:20} → MAE={mae:.3f}, KPI={kpi}/{total}")
                else:
                    p(f"    [{combo_num}/{total_combos}] ✗ {model_name:12} + {fs_name:20} → SKIP")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / 'extended_comparison.csv', index=False)

    p("\n" + "=" * 70)
    p("СВОДКА РАСШИРЕННОГО ИССЛЕДОВАНИЯ")
    p("=" * 70)

    for horizon in ['h=1', 'h=2', 'h=12']:
        h_data = results_df[results_df['Horizon'] == horizon].sort_values('MAE')

        if len(h_data) == 0:
            continue

        p(f"\n  === ТОП-10 для {horizon} ===")
        p(f"  {'Модель':<12} {'Признаки':<20} {'MAE':>8} {'KPI':>8}")
        p("  " + "-" * 55)

        for _, row in h_data.head(10).iterrows():
            p(f"  {row['Model']:<12} {row['FeatureSet']:<20} {row['MAE']:>8.3f} {int(row['KPI_Hits']):>3}/{int(row['Total'])}")

    # Compare with baseline
    p("\n" + "=" * 70)
    p("СРАВНЕНИЕ С BASELINE (Ridge + AR)")
    p("=" * 70)

    baseline = {
        'h=1': 0.547,  # Ridge + AR_minimal from previous test
        'h=2': 0.582,
        'h=12': 0.393
    }

    for horizon in ['h=1', 'h=2', 'h=12']:
        h_data = results_df[results_df['Horizon'] == horizon].sort_values('MAE')
        if len(h_data) > 0:
            best = h_data.iloc[0]
            improvement = (baseline[horizon] - best['MAE']) / baseline[horizon] * 100
            p(f"  {horizon}: {best['Model']} + {best['FeatureSet']} → MAE {best['MAE']:.3f} ({improvement:+.1f}% vs baseline)")

    end_time = datetime.now()
    p(f"\n  Время выполнения: {end_time - start_time}")

    return results_df


if __name__ == '__main__':
    results = run_extended_research()
