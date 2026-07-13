#!/usr/bin/env python3
"""
БЫСТРОЕ ИССЛЕДОВАНИЕ МОДЕЛЕЙ
============================
Оптимизированная версия для быстрого получения результатов.
Тестирует ключевые модели и наборы признаков.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Flush after each print
def p(msg):
    print(msg, flush=True)

# Sklearn
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Optional imports
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

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
    # Normalize to month start
    infl.index = infl.index.to_period('M').to_timestamp()
    infl = infl[~infl.index.duplicated(keep='last')]

    # Monthly regional data (already in wide format: Date, 1, 2, 3, ...)
    month = pd.read_csv(DATA_DIR / 'month.csv', sep=';', decimal=',', encoding='utf-8-sig')
    month['Date'] = pd.to_datetime(month['Date'], format='%d.%m.%Y')
    month = month.set_index('Date').sort_index()
    # Rename columns
    month.columns = [f'reg_{c}' if str(c).isdigit() else c for c in month.columns]
    month = month.select_dtypes(include=[np.number])

    # Quarterly data (already in wide format: Data, 14, 15, ...)
    quart = pd.read_csv(DATA_DIR / 'quart.csv', sep=';', decimal=',', encoding='utf-8-sig')
    date_col = quart.columns[0]  # First column is date (Data)
    quart['Date'] = pd.to_datetime(quart[date_col], format='%d.%m.%Y')
    quart = quart.drop(columns=[date_col]).set_index('Date').sort_index()
    quart.columns = [f'q_{c}' for c in quart.columns]
    quart = quart.select_dtypes(include=[np.number])
    # Remove duplicates
    quart = quart[~quart.index.duplicated(keep='last')]
    # Convert quarterly to monthly: create monthly index and forward fill
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
    p(f"  Период: {df.index.min()} — {df.index.max()}")
    p(f"  Колонок: {len(df.columns)}")

    return df


def create_features(df):
    """Create all feature variations."""
    p("\n" + "=" * 70)
    p("ГЕНЕРАЦИЯ ПРИЗНАКОВ")
    p("=" * 70)

    result = df.copy()
    base_cols = df.columns.tolist()

    for col in base_cols:
        if df[col].isna().sum() > len(df) * 0.5:  # Skip if >50% missing
            continue

        s = df[col]

        # Lags
        for lag in [1, 2, 3, 6, 12]:
            result[f'{col}_L{lag}'] = s.shift(lag)

        # Differences
        for d in [1, 3, 6]:
            result[f'{col}_D{d}'] = s.diff(d)

        # Moving averages
        for w in [3, 6]:
            result[f'{col}_MA{w}'] = s.rolling(w).mean()

    # Seasonality
    result['month_sin'] = np.sin(2 * np.pi * result.index.month / 12)
    result['month_cos'] = np.cos(2 * np.pi * result.index.month / 12)
    result['is_jan'] = (result.index.month == 1).astype(int)
    result['is_jul'] = (result.index.month == 7).astype(int)

    p(f"  Создано {len(result.columns)} признаков")

    return result


def define_feature_sets(df):
    """Define different feature sets to test."""

    # Get available columns
    cols = df.columns.tolist()

    def filter_cols(patterns, exclude_patterns=None):
        result = []
        for c in cols:
            if any(p in c for p in patterns):
                if exclude_patterns and any(ep in c for ep in exclude_patterns):
                    continue
                result.append(c)
        return result

    feature_sets = {
        # Baseline
        'AR_minimal': ['mom_L1', 'mom_L2', 'mom_L3', 'month_sin', 'month_cos'],

        'AR_extended': ['mom_L1', 'mom_L2', 'mom_L3', 'mom_L6', 'mom_L12',
                       'mom_D1', 'mom_D3', 'mom_MA3', 'month_sin', 'month_cos', 'is_jan', 'is_jul'],

        # Components
        'Components': ['mom_L1', 'mom_L2', 'Prod_L1', 'Nonprod_L1', 'Serv_L1',
                      'Prod_D1', 'Nonprod_D1', 'Serv_D1', 'month_sin', 'month_cos'],

        # Federal macro
        'Federal_macro': ['mom_L1', 'mom_L2', 'Ki_i_L1', 'Ki_i_L6', 'Ruonia_L1', 'Ruonia_L2',
                         'usd_nom_i_L1', 'usd_nom_i_L2', 'month_sin', 'month_cos'],

        # IBVED (indicator 20)
        'IBVED_q': ['mom_L1', 'mom_L2', 'q_20_L1', 'q_20_MA3', 'month_sin', 'month_cos'],

        # Regional macro
        'Regional': ['mom_L1', 'mom_L2', 'reg_1_L1', 'reg_2_L1', 'reg_3_L1',
                    'reg_1_D1', 'reg_2_D1', 'month_sin', 'month_cos'],

        # Best from previous research
        'Best_combined': ['mom_L1', 'mom_L2', 'mom_L3',
                         'Ki_i_L6', 'Ruonia_D1',
                         'Prod_L1', 'Serv_L1',
                         'month_sin', 'month_cos', 'is_jan'],

        # Kitchen sink (all features) - expected to overfit
        'Kitchen_sink': [c for c in cols if not c.startswith('target') and c != 'mom'][:50],
    }

    # Filter to only existing columns
    for name in feature_sets:
        feature_sets[name] = [c for c in feature_sets[name] if c in cols]

    return feature_sets


def define_models():
    """Define models to test."""
    models = {
        # Linear models
        'Ridge': Ridge(alpha=1.0, random_state=RANDOM_STATE),
        'Ridge_10': Ridge(alpha=10.0, random_state=RANDOM_STATE),
        'Ridge_100': Ridge(alpha=100.0, random_state=RANDOM_STATE),
        'Lasso': Lasso(alpha=0.1, random_state=RANDOM_STATE, max_iter=5000),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=RANDOM_STATE, max_iter=5000),
        'Huber': HuberRegressor(max_iter=500),
        'BayesianRidge': BayesianRidge(),

        # Tree-based
        'RF': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1),
        'GradBoost': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=RANDOM_STATE),
    }

    if HAS_XGBOOST:
        models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=3, random_state=RANDOM_STATE, verbosity=0)

    if HAS_LIGHTGBM:
        models['LightGBM'] = LGBMRegressor(n_estimators=100, max_depth=3, random_state=RANDOM_STATE, verbose=-1)

    if HAS_CATBOOST:
        models['CatBoost'] = CatBoostRegressor(iterations=100, depth=4, random_state=RANDOM_STATE, verbose=0)

    if HAS_NGBOOST:
        models['NGBoost'] = NGBRegressor(n_estimators=100, random_state=RANDOM_STATE, verbose=False)

    return models


def backtest_model(model, df, features, horizon, test_start=TEST_START):
    """Run expanding window backtest."""

    # Create target
    if horizon == 1:
        df = df.copy()
        df['target'] = df['mom'].shift(-1)
    elif horizon == 2:
        df = df.copy()
        df['target'] = df['mom'].shift(-2)
    else:  # h=12
        df = df.copy()
        df['target'] = df['mom'].shift(-12)

    # Filter to valid data
    valid_features = [f for f in features if f in df.columns]
    if len(valid_features) < 2:
        return None, None, None, None

    df_clean = df[['target'] + valid_features].dropna()

    if len(df_clean) < MIN_TRAIN + 12:
        return None, None, None, None

    test_start_dt = pd.to_datetime(test_start)

    # Split
    train_idx = df_clean.index < test_start_dt
    test_idx = df_clean.index >= test_start_dt

    if train_idx.sum() < MIN_TRAIN or test_idx.sum() < 6:
        return None, None, None, None

    # Expanding window backtest
    predictions = []
    actuals = []

    test_dates = df_clean.index[test_idx]

    for i, test_date in enumerate(test_dates):
        # Training data: everything before test_date
        train_mask = df_clean.index < test_date

        if train_mask.sum() < MIN_TRAIN:
            continue

        X_train = df_clean.loc[train_mask, valid_features].values
        y_train = df_clean.loc[train_mask, 'target'].values

        X_test = df_clean.loc[[test_date], valid_features].values
        y_test = df_clean.loc[test_date, 'target']

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Clone and fit model
        try:
            from sklearn.base import clone
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


def run_research():
    """Main research loop."""

    start_time = datetime.now()
    p(f"\nНачало: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df = load_data()

    # Create features
    df = create_features(df)

    # Define what to test
    feature_sets = define_feature_sets(df)
    models = define_models()
    horizons = [1, 2, 12]

    p("\n" + "=" * 70)
    p("КОНФИГУРАЦИЯ ТЕСТА")
    p("=" * 70)
    p(f"  Моделей: {len(models)}")
    p(f"  Наборов признаков: {len(feature_sets)}")
    p(f"  Горизонтов: {len(horizons)}")
    p(f"  Всего комбинаций: {len(models) * len(feature_sets) * len(horizons)}")

    # Results storage
    all_results = []

    p("\n" + "=" * 70)
    p("ЗАПУСК БЭКТЕСТОВ")
    p("=" * 70)

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

                    # Print progress
                    status = "✓" if mae < 0.5 else "○"
                    p(f"    [{combo_num}/{total_combos}] {status} {model_name:12} + {fs_name:15} → MAE={mae:.3f}, KPI={kpi}/{total}")
                else:
                    p(f"    [{combo_num}/{total_combos}] ✗ {model_name:12} + {fs_name:15} → SKIP (insufficient data)")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Save results
    results_df.to_csv(RESULTS_DIR / 'model_comparison_full.csv', index=False)
    p(f"\n  Результаты сохранены: {RESULTS_DIR / 'model_comparison_full.csv'}")

    # Print summary
    p("\n" + "=" * 70)
    p("СВОДКА РЕЗУЛЬТАТОВ")
    p("=" * 70)

    for horizon in ['h=1', 'h=2', 'h=12']:
        h_data = results_df[results_df['Horizon'] == horizon].sort_values('MAE')

        if len(h_data) == 0:
            continue

        p(f"\n  === ТОП-5 для {horizon} ===")
        p(f"  {'Модель':<12} {'Признаки':<15} {'MAE':>8} {'KPI':>8}")
        p("  " + "-" * 50)

        for _, row in h_data.head(5).iterrows():
            p(f"  {row['Model']:<12} {row['FeatureSet']:<15} {row['MAE']:>8.3f} {int(row['KPI_Hits']):>3}/{int(row['Total'])}")

    # Best overall
    best = results_df.loc[results_df['MAE'].idxmin()]
    p(f"\n  ЛУЧШАЯ КОМБИНАЦИЯ: {best['Model']} + {best['FeatureSet']} ({best['Horizon']})")
    p(f"  MAE: {best['MAE']:.3f}, KPI: {int(best['KPI_Hits'])}/{int(best['Total'])}")

    # Timing
    end_time = datetime.now()
    duration = end_time - start_time
    p(f"\n  Время выполнения: {duration}")

    # Generate report
    generate_report(results_df)

    return results_df


def generate_report(results_df):
    """Generate markdown report."""

    report = []
    report.append("# ОТЧЁТ: Сравнение моделей и признаков")
    report.append(f"\nДата: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")

    # Summary
    report.append("## 1. СВОДКА")
    report.append("")
    report.append(f"- Протестировано комбинаций: {len(results_df)}")
    report.append(f"- Моделей: {results_df['Model'].nunique()}")
    report.append(f"- Наборов признаков: {results_df['FeatureSet'].nunique()}")
    report.append("")

    # Best by horizon
    report.append("## 2. ЛУЧШИЕ МОДЕЛИ ПО ГОРИЗОНТАМ")
    report.append("")

    for horizon in ['h=1', 'h=2', 'h=12']:
        h_data = results_df[results_df['Horizon'] == horizon].sort_values('MAE')

        if len(h_data) == 0:
            continue

        report.append(f"### {horizon}")
        report.append("")
        report.append("| # | Модель | Признаки | MAE | KPI |")
        report.append("|---|--------|----------|-----|-----|")

        for i, (_, row) in enumerate(h_data.head(10).iterrows()):
            report.append(f"| {i+1} | {row['Model']} | {row['FeatureSet']} | {row['MAE']:.3f} | {int(row['KPI_Hits'])}/{int(row['Total'])} |")

        report.append("")

    # Best feature sets
    report.append("## 3. ЛУЧШИЕ НАБОРЫ ПРИЗНАКОВ (по среднему MAE)")
    report.append("")

    fs_avg = results_df.groupby('FeatureSet')['MAE'].mean().sort_values()

    report.append("| Набор | Средний MAE |")
    report.append("|-------|-------------|")

    for fs, mae in fs_avg.items():
        report.append(f"| {fs} | {mae:.3f} |")

    report.append("")

    # Best models
    report.append("## 4. ЛУЧШИЕ МОДЕЛИ (по среднему MAE)")
    report.append("")

    model_avg = results_df.groupby('Model')['MAE'].mean().sort_values()

    report.append("| Модель | Средний MAE |")
    report.append("|--------|-------------|")

    for model, mae in model_avg.items():
        report.append(f"| {model} | {mae:.3f} |")

    report.append("")

    # Recommendations
    report.append("## 5. РЕКОМЕНДАЦИИ")
    report.append("")

    best = results_df.loc[results_df['MAE'].idxmin()]
    report.append(f"**Лучшая комбинация:** {best['Model']} + {best['FeatureSet']} на {best['Horizon']}")
    report.append(f"- MAE: {best['MAE']:.3f}")
    report.append(f"- KPI: {int(best['KPI_Hits'])}/{int(best['Total'])}")

    # Save report
    report_path = RESULTS_DIR / 'MODEL_COMPARISON_REPORT.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))

    p(f"\n  Отчёт сохранён: {report_path}")


if __name__ == '__main__':
    results = run_research()
