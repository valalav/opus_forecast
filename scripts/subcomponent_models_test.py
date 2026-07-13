#!/usr/bin/env python3
"""
ТЕСТ РАЗНЫХ МОДЕЛЕЙ ДЛЯ СУБКОМПОНЕНТОВ
======================================
Проверяем: VotingRegressor, Ridge, NGBoost, Prophet для каждого субкомпонента.
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
HORIZON = 1  # Главный КПЭ

# Check available models
NGBOOST_AVAILABLE = False
PROPHET_AVAILABLE = False

try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    NGBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    pass


def load_data():
    """Load all data."""
    sub = pd.read_csv(DATA_DIR / 'raw' / 'sub_mom.csv', sep=';', decimal=',', encoding='utf-8-sig')
    sub['Date'] = pd.to_datetime(sub['Date'], format='%d.%m.%Y')
    sub = sub.set_index('Date').sort_index()
    sub.index = sub.index.to_period('M').to_timestamp()
    sub = sub[~sub.index.duplicated(keep='last')]

    sprav = pd.read_csv(DATA_DIR / 'raw' / 'subcomp_sprav.csv', sep=';', decimal=',', encoding='utf-8-sig')
    weights = dict(zip(sprav['Item_code'].astype(str), sprav['Weight']))
    names = dict(zip(sprav['Item_code'].astype(str), sprav['Товар']))

    valid_cols = [c for c in sub.columns if c in weights]
    sub = sub[valid_cols]

    return sub, weights, names


def create_features(series):
    """Create features for ML models."""
    df = pd.DataFrame({'y': series})
    for lag in [1, 2, 3, 6, 12]:
        df[f'L{lag}'] = df['y'].shift(lag)
    df['D1'] = df['y'].diff(1)
    df['MA3'] = df['y'].rolling(3).mean()
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    return df


def backtest_voting(series, train_start='2016-01-01'):
    """Backtest VotingRegressor (Ridge + Lasso)."""
    df = create_features(series)
    df['target'] = df['y'].shift(-HORIZON)
    df = df.dropna()

    if train_start:
        df = df[df.index >= pd.to_datetime(train_start)]

    if len(df) < MIN_TRAIN + 6:
        return None

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
        model = VotingRegressor([
            ('ridge', Ridge(alpha=100.0, random_state=RANDOM_STATE)),
            ('lasso', Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=5000)),
        ])

        try:
            model.fit(scaler.fit_transform(X_tr), y_tr)
            preds.append(model.predict(scaler.transform(X_te))[0])
            acts.append(y_te)
        except:
            continue

    if len(preds) < 6:
        return None
    return mean_absolute_error(acts, preds)


def backtest_ridge(series, train_start='2016-01-01'):
    """Backtest pure Ridge."""
    df = create_features(series)
    df['target'] = df['y'].shift(-HORIZON)
    df = df.dropna()

    if train_start:
        df = df[df.index >= pd.to_datetime(train_start)]

    if len(df) < MIN_TRAIN + 6:
        return None

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
        model = Ridge(alpha=100.0, random_state=RANDOM_STATE)

        try:
            model.fit(scaler.fit_transform(X_tr), y_tr)
            preds.append(model.predict(scaler.transform(X_te))[0])
            acts.append(y_te)
        except:
            continue

    if len(preds) < 6:
        return None
    return mean_absolute_error(acts, preds)


def backtest_ngboost(series, train_start='2016-01-01'):
    """Backtest NGBoost."""
    if not NGBOOST_AVAILABLE:
        return None

    df = create_features(series)
    df['target'] = df['y'].shift(-HORIZON)
    df = df.dropna()

    if train_start:
        df = df[df.index >= pd.to_datetime(train_start)]

    if len(df) < MIN_TRAIN + 6:
        return None

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
        model = NGBRegressor(
            Dist=Normal,
            n_estimators=100,
            learning_rate=0.05,
            minibatch_frac=1.0,
            random_state=RANDOM_STATE,
            verbose=False
        )

        try:
            model.fit(scaler.fit_transform(X_tr), y_tr)
            preds.append(model.predict(scaler.transform(X_te))[0])
            acts.append(y_te)
        except:
            continue

    if len(preds) < 6:
        return None
    return mean_absolute_error(acts, preds)


def backtest_prophet(series, train_start='2016-01-01'):
    """Backtest Prophet."""
    if not PROPHET_AVAILABLE:
        return None

    # Prepare data for Prophet
    df = pd.DataFrame({
        'ds': series.index,
        'y': series.values
    })

    if train_start:
        df = df[df['ds'] >= pd.to_datetime(train_start)]

    if len(df) < MIN_TRAIN + 6:
        return None

    test_dt = pd.to_datetime(TEST_START)

    preds, acts = [], []
    test_dates = df[df['ds'] >= test_dt]['ds'].values

    for test_date in test_dates:
        train_df = df[df['ds'] < test_date]
        if len(train_df) < MIN_TRAIN:
            continue

        # Target is HORIZON months ahead
        target_date = pd.to_datetime(test_date) + pd.DateOffset(months=HORIZON)
        if target_date not in series.index:
            continue

        actual = series.loc[target_date]

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=0.05
        )
        model.fit(train_df)

        future = model.make_future_dataframe(periods=HORIZON+1, freq='MS')
        forecast = model.predict(future)

        pred_row = forecast[forecast['ds'] == target_date]
        if len(pred_row) > 0:
            preds.append(pred_row['yhat'].values[0])
            acts.append(actual)

    if len(preds) < 6:
        return None
    return mean_absolute_error(acts, preds)


def main():
    start_time = datetime.now()
    p(f"\nНачало: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    p(f"NGBoost: {'✅' if NGBOOST_AVAILABLE else '❌'}")
    p(f"Prophet: {'✅' if PROPHET_AVAILABLE else '❌'}")

    sub, weights, names = load_data()

    models = ['voting', 'ridge']
    if NGBOOST_AVAILABLE:
        models.append('ngboost')
    if PROPHET_AVAILABLE:
        models.append('prophet')

    p(f"\nТестируем {len(models)} моделей на {len(sub.columns)} субкомпонентах")
    p("=" * 90)

    results = []

    for i, col in enumerate(sub.columns):
        name = names.get(col, col)[:30]
        weight = weights.get(col, 0) * 100

        model_results = {}

        # Test each model
        mae_voting = backtest_voting(sub[col])
        if mae_voting:
            model_results['voting'] = mae_voting

        mae_ridge = backtest_ridge(sub[col])
        if mae_ridge:
            model_results['ridge'] = mae_ridge

        if NGBOOST_AVAILABLE:
            mae_ngboost = backtest_ngboost(sub[col])
            if mae_ngboost:
                model_results['ngboost'] = mae_ngboost

        if PROPHET_AVAILABLE:
            mae_prophet = backtest_prophet(sub[col])
            if mae_prophet:
                model_results['prophet'] = mae_prophet

        if model_results:
            best_model = min(model_results, key=model_results.get)
            best_mae = model_results[best_model]
            voting_mae = model_results.get('voting', best_mae)
            improvement = (voting_mae - best_mae) / voting_mae * 100 if voting_mae > 0 else 0

            results.append({
                'Code': col,
                'Name': name,
                'Weight': weight,
                'Best_Model': best_model,
                'Best_MAE': best_mae,
                'Voting_MAE': voting_mae,
                'Improvement': improvement,
                **{f'MAE_{m}': model_results.get(m, np.nan) for m in models}
            })

            if improvement > 5 or best_model != 'voting':
                p(f"  {col:>3} | {weight:>5.2f}% | {best_model:>8} | MAE {best_mae:.3f} | vs Voting {improvement:+.1f}% | {name}")

        # Progress
        if (i + 1) % 15 == 0:
            p(f"  ... обработано {i+1}/{len(sub.columns)}")

    # Summary
    p("\n" + "=" * 90)
    p("ИТОГОВАЯ СВОДКА")
    p("=" * 90)

    df = pd.DataFrame(results)

    # Best model distribution
    p("\n### Распределение лучших моделей:")
    model_stats = {}
    for model in models:
        subset = df[df['Best_Model'] == model]
        if len(subset) > 0:
            weight_sum = subset['Weight'].sum()
            avg_imp = subset['Improvement'].mean()
            model_stats[model] = {'count': len(subset), 'weight': weight_sum, 'avg_imp': avg_imp}
            p(f"  {model:>10}: {len(subset):>2} субкомп. ({weight_sum:.1f}% веса), ср. улучшение vs voting: {avg_imp:+.1f}%")

    # Top improvements
    if len(df[df['Improvement'] > 0]) > 0:
        p("\n### Топ улучшений (vs VotingRegressor):")
        top = df[df['Improvement'] > 0].nlargest(10, 'Improvement')
        for _, row in top.iterrows():
            p(f"  {row['Code']:>3} | {row['Best_Model']:>8} | +{row['Improvement']:.1f}% | {row['Name']}")

    # Save
    df.to_csv(RESULTS_DIR / 'subcomponent_models_comparison.csv', index=False)
    p(f"\n  Сохранено: {RESULTS_DIR / 'subcomponent_models_comparison.csv'}")

    p(f"\n  Время: {datetime.now() - start_time}")


if __name__ == '__main__':
    main()
