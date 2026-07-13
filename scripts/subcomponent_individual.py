#!/usr/bin/env python3
"""
ИНДИВИДУАЛЬНЫЕ МОДЕЛИ ДЛЯ СУБКОМПОНЕНТОВ
========================================
Тестируем разные подходы для каждого из 45 субкомпонентов.
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
MIN_TRAIN = 24
RANDOM_STATE = 42
HORIZON = 12  # Годовой прогноз (где субкомпоненты работают)


def load_all_data():
    """Load all data sources."""

    # Subcomponents MoM (in % format)
    sub = pd.read_csv(DATA_DIR / 'raw' / 'sub_mom.csv', sep=';', decimal=',', encoding='utf-8-sig')
    sub['Date'] = pd.to_datetime(sub['Date'], format='%d.%m.%Y')
    sub = sub.set_index('Date').sort_index()
    sub.index = sub.index.to_period('M').to_timestamp()
    sub = sub[~sub.index.duplicated(keep='last')]

    # Weights and справочник
    sprav = pd.read_csv(DATA_DIR / 'raw' / 'subcomp_sprav.csv', sep=';', decimal=',', encoding='utf-8-sig')
    weights = dict(zip(sprav['Item_code'].astype(str), sprav['Weight']))
    names = dict(zip(sprav['Item_code'].astype(str), sprav['Товар']))
    components = dict(zip(sprav['Item_code'].astype(str), sprav['Компонент']))

    # Filter valid columns
    valid_cols = [c for c in sub.columns if c in weights]
    sub = sub[valid_cols]

    # Macro data (USD, Ki, Ruonia, Brent)
    infl = pd.read_csv(DATA_DIR / 'inflation_data.csv', sep=';', decimal=',')
    infl['Date'] = pd.to_datetime(infl['Date'])
    infl = infl.set_index('Date').sort_index()
    infl.index = infl.index.to_period('M').to_timestamp()
    infl = infl[~infl.index.duplicated(keep='last')]

    # Brent prices
    try:
        brent = pd.read_csv(DATA_DIR / 'brent_prices.csv')
        brent['Date'] = pd.to_datetime(brent['Date'])
        brent = brent.set_index('Date').sort_index()
        brent.index = brent.index.to_period('M').to_timestamp()
        brent = brent[~brent.index.duplicated(keep='last')]
        infl = infl.join(brent[['brent']], how='left')
    except:
        infl['brent'] = np.nan

    return sub, weights, names, components, infl


def create_features(series, macro_df, approach='baseline'):
    """Create features based on approach."""

    df = pd.DataFrame({'y': series})

    # Basic lags (всегда)
    for lag in [1, 2, 3, 6, 12]:
        df[f'L{lag}'] = df['y'].shift(lag)

    # Differences
    df['D1'] = df['y'].diff(1)
    df['MA3'] = df['y'].rolling(3).mean()

    # Seasonality (всегда)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    # Additional features based on approach
    if approach == 'usd' or approach == 'all':
        # USD features
        if 'usd_nom_i' in macro_df.columns:
            usd = macro_df['usd_nom_i'].reindex(df.index)
            df['usd_L1'] = usd.shift(1)
            df['usd_L3'] = usd.shift(3)
            df['usd_L6'] = usd.shift(6)
            df['usd_D1'] = usd.diff(1)

    if approach == 'brent' or approach == 'all':
        # Brent features
        if 'brent' in macro_df.columns:
            brent = macro_df['brent'].reindex(df.index)
            df['brent_L1'] = brent.shift(1)
            df['brent_L3'] = brent.shift(3)
            df['brent_L6'] = brent.shift(6)
            df['brent_D1'] = brent.diff(1)

    if approach == 'monetary' or approach == 'all':
        # Ki, Ruonia
        if 'Ki_i' in macro_df.columns:
            ki = macro_df['Ki_i'].reindex(df.index)
            df['ki_L3'] = ki.shift(3)
            df['ki_L6'] = ki.shift(6)
        if 'Ruonia' in macro_df.columns:
            ruonia = macro_df['Ruonia'].reindex(df.index)
            df['ruonia_L1'] = ruonia.shift(1)
            df['ruonia_D1'] = ruonia.diff(1)

    if approach == 'seasonal' or approach == 'all':
        # Strong seasonality
        df['is_jan'] = (df.index.month == 1).astype(int)
        df['is_jul'] = (df.index.month == 7).astype(int)
        df['is_dec'] = (df.index.month == 12).astype(int)
        df['quarter_sin'] = np.sin(2 * np.pi * ((df.index.month - 1) // 3) / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * ((df.index.month - 1) // 3) / 4)

    if approach == 'tariff':
        # Tariff indexation (for ЖКХ, transport)
        df['is_jul'] = (df.index.month == 7).astype(int)
        df['is_jan'] = (df.index.month == 1).astype(int)
        # Trend component
        df['trend'] = np.arange(len(df))

    return df


def backtest_subcomponent(series, macro_df, approach, train_start='2016-01-01'):
    """Backtest a single subcomponent with given approach."""

    df = create_features(series, macro_df, approach)

    # Target
    df['target'] = df['y'].shift(-HORIZON)
    df = df.dropna()

    if train_start:
        df = df[df.index >= pd.to_datetime(train_start)]

    if len(df) < MIN_TRAIN + 6:
        return None

    # Backtest
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

        # Handle NaN in features
        if np.any(np.isnan(X_tr)) or np.any(np.isnan(X_te)):
            # Fill NaN with column mean
            col_means = np.nanmean(X_tr, axis=0)
            for i in range(X_tr.shape[1]):
                X_tr[np.isnan(X_tr[:, i]), i] = col_means[i] if not np.isnan(col_means[i]) else 0
                X_te[np.isnan(X_te[:, i]), i] = col_means[i] if not np.isnan(col_means[i]) else 0

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = VotingRegressor([
            ('ridge', Ridge(alpha=100.0, random_state=RANDOM_STATE)),
            ('lasso', Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=5000)),
        ])

        try:
            model.fit(X_tr_s, y_tr)
            pred = model.predict(X_te_s)[0]
            preds.append(pred)
            acts.append(y_te)
        except:
            continue

    if len(preds) < 6:
        return None

    mae = mean_absolute_error(acts, preds)
    return mae


def main():
    start_time = datetime.now()
    p(f"\nНачало: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    p(f"Горизонт: h={HORIZON}")

    sub, weights, names, components, macro = load_all_data()

    approaches = ['baseline', 'usd', 'brent', 'monetary', 'seasonal', 'tariff', 'all']

    p(f"\nСубкомпонентов: {len(sub.columns)}")
    p(f"Подходов: {len(approaches)}")
    p(f"Всего тестов: {len(sub.columns) * len(approaches)}")

    results = []

    for col in sub.columns:
        name = names.get(col, col)[:40]
        comp = components.get(col, 'Unknown')[:15]
        weight = weights.get(col, 0) * 100

        best_mae = float('inf')
        best_approach = None
        approach_results = {}

        for approach in approaches:
            mae = backtest_subcomponent(sub[col], macro, approach)
            if mae is not None:
                approach_results[approach] = mae
                if mae < best_mae:
                    best_mae = mae
                    best_approach = approach

        if best_approach:
            # Calculate improvement vs baseline
            baseline_mae = approach_results.get('baseline', best_mae)
            improvement = (baseline_mae - best_mae) / baseline_mae * 100 if baseline_mae > 0 else 0

            results.append({
                'Code': col,
                'Name': name,
                'Component': comp,
                'Weight': weight,
                'Best_Approach': best_approach,
                'Best_MAE': best_mae,
                'Baseline_MAE': baseline_mae,
                'Improvement': improvement,
                **{f'MAE_{a}': approach_results.get(a, np.nan) for a in approaches}
            })

            if improvement > 5:
                p(f"  {col:>3} | {weight:>5.2f}% | {best_approach:>10} | MAE {best_mae:.3f} | +{improvement:.1f}% | {name}")
            elif improvement < -5:
                p(f"  {col:>3} | {weight:>5.2f}% | {best_approach:>10} | MAE {best_mae:.3f} | {improvement:.1f}% | {name}")

    # Summary
    p("\n" + "=" * 90)
    p("ИТОГОВАЯ СВОДКА")
    p("=" * 90)

    df = pd.DataFrame(results)

    # Best approach distribution
    p("\n### Распределение лучших подходов:")
    approach_counts = df['Best_Approach'].value_counts()
    for approach, count in approach_counts.items():
        weight_sum = df[df['Best_Approach'] == approach]['Weight'].sum()
        p(f"  {approach:>10}: {count:>2} субкомп. ({weight_sum:.1f}% веса)")

    # Top improvements
    p("\n### Топ улучшений (vs baseline):")
    top = df.nlargest(10, 'Improvement')
    for _, row in top.iterrows():
        if row['Improvement'] > 0:
            p(f"  {row['Code']:>3} | {row['Best_Approach']:>10} | +{row['Improvement']:.1f}% | {row['Name']}")

    # Worst results (baseline is best)
    p("\n### Где baseline лучше всего:")
    worst = df.nsmallest(5, 'Improvement')
    for _, row in worst.iterrows():
        if row['Improvement'] < 0:
            p(f"  {row['Code']:>3} | {row['Best_Approach']:>10} | {row['Improvement']:.1f}% | {row['Name']}")

    # By component
    p("\n### По компонентам:")
    for comp in df['Component'].unique():
        comp_df = df[df['Component'] == comp]
        avg_improvement = comp_df['Improvement'].mean()
        p(f"  {comp}: среднее улучшение {avg_improvement:+.1f}%")

    # Save
    df.to_csv(RESULTS_DIR / 'subcomponent_individual_results.csv', index=False)
    p(f"\n  Сохранено: {RESULTS_DIR / 'subcomponent_individual_results.csv'}")

    p(f"\n  Время: {datetime.now() - start_time}")


if __name__ == '__main__':
    main()
