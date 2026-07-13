#!/usr/bin/env python3
"""
ОПТИМИЗИРОВАННАЯ СУБКОМПОНЕНТНАЯ МОДЕЛЬ
=======================================
Использует лучший подход для каждого субкомпонента на основе эмпирических данных.
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

# Оптимальные подходы для каждого субкомпонента (на основе эксперимента)
OPTIMAL_APPROACHES = {
    # SEASONAL (40% веса) — сильная сезонность
    '26': 'seasonal',  # Мясопродукты
    '14': 'seasonal',  # ЖКХ (+13.6%!)
    '29': 'seasonal',  # Одежда и белье
    '24': 'seasonal',  # Молоко и молочная продукция
    '48': 'seasonal',  # Услуги телекоммуникационные
    '30': 'seasonal',  # Парфюмерно-косметические товары
    '43': 'seasonal',  # Трикотажные изделия
    '28': 'seasonal',  # Общественное питание
    '38': 'seasonal',  # Строительные материалы
    '39': 'seasonal',  # Сыр
    '35': 'seasonal',  # Санаторно-оздоровительные услуги
    '67': 'seasonal',  # Услуги в сфере зарубежного туризма

    # USD (16.8% веса) — зависимость от курса доллара
    '53': 'usd',  # Другие продовольственные товары
    '16': 'usd',  # Кондитерские изделия
    '27': 'usd',  # Обувь
    '47': 'usd',  # Услуги пассажирского транспорта
    '18': 'usd',  # Макаронные и крупяные изделия
    '51': 'usd',  # Электротовары
    '44': 'usd',  # Услуги в системе образования
    '52': 'usd',  # Яйца (+14.5%!)

    # BRENT (8.4% веса) — зависимость от нефти
    '42': 'brent',  # Топливо моторное (+15.5%)
    '34': 'brent',  # Рыбопродукты
    '19': 'brent',  # Масло и жиры (+28.9%!)
    '40': 'brent',  # Табачные изделия
    '23': 'brent',  # Меха и меховые изделия (+21.7%)
    '15': 'brent',  # Инструменты и оборудование

    # TARIFF (6.7% веса) — регулируемые тарифы
    '12': 'tariff',  # Бытовые услуги
    '21': 'tariff',  # Медицинские товары (+8.9%)
    '46': 'tariff',  # Услуги организаций культуры

    # MONETARY (2.1% веса) — зависимость от денежно-кредитных условий
    '49': 'monetary',  # Хлеб и хлебобулочные изделия
    '31': 'monetary',  # Персональные компьютеры (+14.7%)
    '41': 'monetary',  # Телерадиотовары

    # ALL (3.4% веса) — комбинированный подход
    '20': 'all',  # Мебель
    '11': 'all',  # Алкогольные напитки

    # BASELINE (22.6% веса) — простая модель лучше
    '33': 'baseline',  # Плодоовощная продукция
    '54': 'baseline',  # Другие непродовольственные товары
    '17': 'baseline',  # Легковые автомобили
    '55': 'baseline',  # Другие услуги
    '50': 'baseline',  # Чай, кофе, какао
    '13': 'baseline',  # Галантерея
    '22': 'baseline',  # Медицинские услуги
    '25': 'baseline',  # Моющие и чистящие средства
    '37': 'baseline',  # Средства связи
    '36': 'baseline',  # Сахар
    '32': 'baseline',  # Печатные издания
}


def load_all_data():
    """Load all data sources."""
    # Subcomponents
    sub = pd.read_csv(DATA_DIR / 'raw' / 'sub_mom.csv', sep=';', decimal=',', encoding='utf-8-sig')
    sub['Date'] = pd.to_datetime(sub['Date'], format='%d.%m.%Y')
    sub = sub.set_index('Date').sort_index()
    sub.index = sub.index.to_period('M').to_timestamp()
    sub = sub[~sub.index.duplicated(keep='last')]

    # Weights
    sprav = pd.read_csv(DATA_DIR / 'raw' / 'subcomp_sprav.csv', sep=';', decimal=',', encoding='utf-8-sig')
    weights = dict(zip(sprav['Item_code'].astype(str), sprav['Weight']))

    # Filter valid
    valid_cols = [c for c in sub.columns if c in weights]
    sub = sub[valid_cols]

    # Macro data
    infl = pd.read_csv(DATA_DIR / 'inflation_data.csv', sep=';', decimal=',')
    infl['Date'] = pd.to_datetime(infl['Date'])
    infl = infl.set_index('Date').sort_index()
    infl.index = infl.index.to_period('M').to_timestamp()
    infl = infl[~infl.index.duplicated(keep='last')]
    infl['mom_pct'] = infl['mom'] - 100

    # Brent
    try:
        brent = pd.read_csv(DATA_DIR / 'brent_prices.csv')
        brent['Date'] = pd.to_datetime(brent['Date'])
        brent = brent.set_index('Date').sort_index()
        brent.index = brent.index.to_period('M').to_timestamp()
        brent = brent[~brent.index.duplicated(keep='last')]
        infl = infl.join(brent[['brent']], how='left')
    except:
        infl['brent'] = np.nan

    return sub, weights, infl


def create_features(series, macro_df, approach):
    """Create features based on approach."""
    df = pd.DataFrame({'y': series})

    # Basic (always)
    for lag in [1, 2, 3, 6, 12]:
        df[f'L{lag}'] = df['y'].shift(lag)
    df['D1'] = df['y'].diff(1)
    df['MA3'] = df['y'].rolling(3).mean()
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    if approach == 'usd' or approach == 'all':
        if 'usd_nom_i' in macro_df.columns:
            usd = macro_df['usd_nom_i'].reindex(df.index)
            df['usd_L1'] = usd.shift(1)
            df['usd_L3'] = usd.shift(3)
            df['usd_L6'] = usd.shift(6)
            df['usd_D1'] = usd.diff(1)

    if approach == 'brent' or approach == 'all':
        if 'brent' in macro_df.columns:
            brent = macro_df['brent'].reindex(df.index)
            df['brent_L1'] = brent.shift(1)
            df['brent_L3'] = brent.shift(3)
            df['brent_L6'] = brent.shift(6)
            df['brent_D1'] = brent.diff(1)

    if approach == 'monetary' or approach == 'all':
        if 'Ki_i' in macro_df.columns:
            ki = macro_df['Ki_i'].reindex(df.index)
            df['ki_L3'] = ki.shift(3)
            df['ki_L6'] = ki.shift(6)
        if 'Ruonia' in macro_df.columns:
            ruonia = macro_df['Ruonia'].reindex(df.index)
            df['ruonia_L1'] = ruonia.shift(1)
            df['ruonia_D1'] = ruonia.diff(1)

    if approach == 'seasonal' or approach == 'all':
        df['is_jan'] = (df.index.month == 1).astype(int)
        df['is_jul'] = (df.index.month == 7).astype(int)
        df['is_dec'] = (df.index.month == 12).astype(int)
        df['quarter_sin'] = np.sin(2 * np.pi * ((df.index.month - 1) // 3) / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * ((df.index.month - 1) // 3) / 4)

    if approach == 'tariff':
        df['is_jul'] = (df.index.month == 7).astype(int)
        df['is_jan'] = (df.index.month == 1).astype(int)
        df['trend'] = np.arange(len(df))

    return df


def forecast_subcomponent(series, macro_df, approach, horizon, train_start='2016-01-01'):
    """Forecast a single subcomponent with optimal approach."""
    df = create_features(series, macro_df, approach)
    df['target'] = df['y'].shift(-horizon)
    df = df.dropna()

    if train_start:
        df = df[df.index >= pd.to_datetime(train_start)]

    if len(df) < MIN_TRAIN + 6:
        return None, None

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

        # Handle NaN
        if np.any(np.isnan(X_tr)) or np.any(np.isnan(X_te)):
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
            target_date = test_date + pd.DateOffset(months=horizon)
            dates.append(target_date)
        except:
            continue

    if len(preds) < 6:
        return None, None

    return pd.Series(preds, index=dates), dates


def run_optimized_model(horizon, train_start='2016-01-01'):
    """Run optimized subcomponent model."""
    p(f"\n{'='*70}")
    p(f"ОПТИМИЗИРОВАННАЯ СУБКОМПОНЕНТНАЯ МОДЕЛЬ h={horizon}")
    p("=" * 70)

    sub, weights, macro = load_all_data()

    p(f"  Субкомпонентов: {len(sub.columns)}")

    forecasts = {}
    approach_counts = {}

    for col in sub.columns:
        approach = OPTIMAL_APPROACHES.get(col, 'baseline')
        approach_counts[approach] = approach_counts.get(approach, 0) + 1

        pred_series, dates = forecast_subcomponent(sub[col], macro, approach, horizon, train_start)
        if pred_series is not None:
            forecasts[col] = pred_series

    p(f"  Успешно: {len(forecasts)}/{len(sub.columns)}")
    p(f"  Подходы: {approach_counts}")

    if len(forecasts) < 10:
        return None

    # Find common dates
    common_dates = forecasts[list(forecasts.keys())[0]].index
    for col in forecasts:
        common_dates = common_dates.intersection(forecasts[col].index)

    # Aggregate
    total_weight = sum(weights[c] for c in forecasts.keys())
    agg_forecast = pd.Series(0.0, index=common_dates)

    for col in forecasts.keys():
        w = weights[col] / total_weight
        agg_forecast += w * forecasts[col].loc[common_dates]

    # Get actual Total
    actual_dates = common_dates.intersection(macro.index)
    if len(actual_dates) < 6:
        return None

    agg_forecast = agg_forecast.loc[actual_dates]
    actual_total = macro.loc[actual_dates, 'mom_pct']

    mae = mean_absolute_error(actual_total, agg_forecast)
    kpi = np.sum(np.abs(agg_forecast - actual_total) <= 0.5)

    p(f"\n  MAE: {mae:.3f}")
    p(f"  KPI: {kpi}/{len(actual_dates)} ({100*kpi/len(actual_dates):.0f}%)")

    return {
        'horizon': horizon,
        'train_start': train_start,
        'mae': mae,
        'kpi': kpi,
        'total': len(actual_dates)
    }


def run_baseline_model(horizon, train_start='2016-01-01'):
    """Run baseline (all subcomponents with same approach)."""
    sub, weights, macro = load_all_data()

    forecasts = {}
    for col in sub.columns:
        pred_series, _ = forecast_subcomponent(sub[col], macro, 'baseline', horizon, train_start)
        if pred_series is not None:
            forecasts[col] = pred_series

    if len(forecasts) < 10:
        return None

    common_dates = forecasts[list(forecasts.keys())[0]].index
    for col in forecasts:
        common_dates = common_dates.intersection(forecasts[col].index)

    total_weight = sum(weights[c] for c in forecasts.keys())
    agg_forecast = pd.Series(0.0, index=common_dates)
    for col in forecasts.keys():
        w = weights[col] / total_weight
        agg_forecast += w * forecasts[col].loc[common_dates]

    actual_dates = common_dates.intersection(macro.index)
    if len(actual_dates) < 6:
        return None

    agg_forecast = agg_forecast.loc[actual_dates]
    actual_total = macro.loc[actual_dates, 'mom_pct']

    mae = mean_absolute_error(actual_total, agg_forecast)
    return {'mae': mae, 'total': len(actual_dates)}


def run_direct_forecast(horizon, train_start='2016-01-01'):
    """Direct forecast of Total."""
    _, _, macro = load_all_data()

    df = pd.DataFrame({'y': macro['mom_pct']})
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
    return {'mae': mae, 'total': len(preds)}


def main():
    start_time = datetime.now()
    p(f"\nНачало: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    for horizon in [1, 2, 12]:
        p(f"\n{'#'*70}")
        p(f"# ГОРИЗОНТ h={horizon}")
        p(f"{'#'*70}")

        # Optimized subcomponent model
        opt = run_optimized_model(horizon, '2016-01-01')

        # Baseline subcomponent (all same approach)
        p("\n  BASELINE (одинаковый подход для всех):")
        base = run_baseline_model(horizon, '2016-01-01')
        if base:
            p(f"  MAE: {base['mae']:.3f}")

        # Direct forecast
        p("\n  DIRECT (прямой прогноз Total):")
        direct = run_direct_forecast(horizon, '2016-01-01')
        p(f"  MAE: {direct['mae']:.3f}")

        if opt and base and direct:
            imp_vs_base = (base['mae'] - opt['mae']) / base['mae'] * 100
            imp_vs_direct = (direct['mae'] - opt['mae']) / direct['mae'] * 100

            p(f"\n  >>> vs Baseline subcomp: {imp_vs_base:+.1f}%")
            p(f"  >>> vs Direct:           {imp_vs_direct:+.1f}%")

            results.append({
                'Horizon': f'h={horizon}',
                'Optimized_MAE': opt['mae'],
                'Baseline_Sub_MAE': base['mae'],
                'Direct_MAE': direct['mae'],
                'vs_Baseline': f"{imp_vs_base:+.1f}%",
                'vs_Direct': f"{imp_vs_direct:+.1f}%"
            })

    # Summary
    p("\n" + "=" * 70)
    p("ИТОГОВАЯ СВОДКА")
    p("=" * 70)

    df = pd.DataFrame(results)
    p("\n" + df.to_string(index=False))

    # Save
    df.to_csv(RESULTS_DIR / 'subcomponent_optimized_results.csv', index=False)
    p(f"\n  Сохранено: {RESULTS_DIR / 'subcomponent_optimized_results.csv'}")

    p(f"\n  Время: {datetime.now() - start_time}")


if __name__ == '__main__':
    main()
