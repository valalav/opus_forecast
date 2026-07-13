#!/usr/bin/env python3
"""
КОМПЛЕКСНОЕ ИССЛЕДОВАНИЕ ПРИЗНАКОВ ДЛЯ ПРОГНОЗИРОВАНИЯ ИНФЛЯЦИИ КБР

Цель: раз и навсегда определить, какие признаки полезны для прогноза.

Данные:
- Инфляция КБР (mom) с 2010
- Федеральные макро: USD, Ki (ключевая ставка), Ruonia, Brent
- Региональные макро КБР: PPI, wages, retail, construction, etc.
- SA данные с 2016

Методы:
1. Корреляционный анализ (Pearson, Spearman)
2. Cross-correlation для определения оптимального лага
3. Granger causality test
4. Feature importance (permutation, mutual information)
5. Бэктест моделей с разными наборами признаков

Автор: Claude Code
Дата: 2025-12-28
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import Ridge, HuberRegressor, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'archive' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Constants
START_DATE = '2016-01-01'  # Start from when SA data is available
TEST_START = '2022-01-01'  # Hold-out test period
MIN_TRAIN = 36  # Minimum training samples


def load_all_data():
    """Load and merge all data sources."""
    print("=" * 80)
    print("1. ЗАГРУЗКА ДАННЫХ")
    print("=" * 80)

    # 1. Inflation + federal macro
    infl = pd.read_csv(DATA_DIR / 'inflation_data.csv', sep=';', decimal=',', on_bad_lines='skip')
    infl['Date'] = pd.to_datetime(infl['Date'], format='%d.%m.%Y')
    infl['Date'] = infl['Date'].dt.to_period('M').dt.to_timestamp()
    infl = infl.set_index('Date').sort_index()
    print(f"  Инфляция + федеральные макро: {len(infl)} точек, {infl.index.min()} — {infl.index.max()}")
    print(f"    Колонки: {list(infl.columns)}")

    # 2. Brent oil prices
    brent = pd.read_csv(DATA_DIR / 'brent_prices.csv')
    brent['Date'] = pd.to_datetime(brent['Date'])
    brent = brent.set_index('Date').sort_index()
    print(f"  Нефть Brent: {len(brent)} точек")

    # 3. Regional monthly data
    regional = pd.read_csv(DATA_DIR / 'month.csv', sep=';', decimal=',')
    regional['Date'] = pd.to_datetime(regional['Date'], format='%d.%m.%Y')
    regional = regional.set_index('Date').sort_index()

    reg_names = {
        '1': 'reg_ind_prod', '2': 'reg_shipped', '3': 'reg_construction',
        '6': 'reg_retail', '7': 'reg_services', '8': 'reg_profit',
        '9': 'reg_payables', '10': 'reg_receivables', '11': 'reg_ppi',
        '12': 'reg_agri_prices', '13': 'reg_invest_prices',
        '18': 'reg_wage', '19': 'reg_wage_agri'
    }
    existing = [c for c in reg_names.keys() if c in regional.columns]
    regional = regional[existing].rename(columns={k: reg_names[k] for k in existing})
    print(f"  Региональные месячные: {len(regional)} точек, {len(regional.columns)} показателей")

    # 4. Regional quarterly data
    quarterly = pd.read_csv(DATA_DIR / 'quart.csv', sep=';', decimal=',')
    quarterly['Date'] = pd.to_datetime(quarterly['Data'], format='%d.%m.%Y', errors='coerce')
    quarterly = quarterly.dropna(subset=['Date'])
    quarterly = quarterly.set_index('Date').sort_index()

    quart_names = {
        '14': 'reg_housing_primary', '15': 'reg_housing_secondary',
        '16': 'reg_income_nom', '17': 'reg_income_real', '20': 'reg_unknown20'
    }
    existing = [c for c in quart_names.keys() if c in quarterly.columns]
    quarterly = quarterly[existing].rename(columns={k: quart_names[k] for k in existing})
    quarterly = quarterly[~quarterly.index.duplicated(keep='last')]
    quarterly = quarterly.resample('MS').ffill()
    print(f"  Региональные квартальные: {len(quarterly)} точек, {len(quarterly.columns)} показателей")

    # Merge all
    df = infl[['mom', 'Nonprod', 'Prod', 'Serv', 'usd_nom_i', 'Ki_i', 'Ruonia', 'Ki']].copy()
    df = df.rename(columns={
        'Nonprod': 'nonprod', 'Prod': 'prod', 'Serv': 'serv',
        'usd_nom_i': 'usd', 'Ki_i': 'ki_i', 'Ki': 'ki'
    })

    # Join other data
    df = df.join(brent[['brent', 'brent_pct']], how='left')
    df = df.join(regional, how='left')
    df = df.join(quarterly, how='left')

    # Filter to start date
    df = df[df.index >= START_DATE]

    # Forward fill quarterly data
    for col in quarterly.columns:
        if col in df.columns:
            df[col] = df[col].ffill()

    print(f"\n  Объединенный датасет: {len(df)} точек, {len(df.columns)} базовых колонок")
    print(f"  Период: {df.index.min()} — {df.index.max()}")

    return df


def create_all_features(df):
    """Create comprehensive feature set with all lags and transformations."""
    print("\n" + "=" * 80)
    print("2. СОЗДАНИЕ ПРИЗНАКОВ")
    print("=" * 80)

    # Target: next month inflation (h=1 forecast)
    df['target'] = df['mom'].shift(-1)

    # Base columns for feature engineering
    base_cols = [c for c in df.columns if c not in ['target']]

    features_created = 0

    for col in base_cols:
        if df[col].isna().sum() > len(df) * 0.5:  # Skip if >50% missing
            continue

        # Raw lagged values (lags 0-12)
        for lag in range(0, 13):
            df[f'{col}_L{lag}'] = df[col].shift(lag)
            features_created += 1

        # First differences (changes)
        for lag in [1, 3, 6, 12]:
            df[f'{col}_D{lag}'] = df[col].diff(lag)
            features_created += 1

        # Moving averages
        for window in [3, 6, 12]:
            df[f'{col}_MA{window}'] = df[col].rolling(window).mean()
            features_created += 1

        # Volatility (rolling std)
        for window in [3, 6]:
            df[f'{col}_STD{window}'] = df[col].rolling(window).std()
            features_created += 1

    # Seasonality features
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['quarter'] = df.index.quarter
    df['is_q1'] = (df['quarter'] == 1).astype(int)
    df['is_jan'] = (df['month'] == 1).astype(int)
    df['is_jul'] = (df['month'] == 7).astype(int)  # Tariff month
    df['is_dec'] = (df['month'] == 12).astype(int)

    print(f"  Создано {features_created} признаков")
    print(f"  Всего колонок: {len(df.columns)}")

    return df


def analyze_correlations(df, top_n=50):
    """Analyze correlations with target."""
    print("\n" + "=" * 80)
    print("3. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
    print("=" * 80)

    target = df['target'].dropna()
    feature_cols = [c for c in df.columns if c not in ['target', 'month', 'quarter']]

    correlations = []

    for col in feature_cols:
        if col not in df.columns:
            continue

        valid = df[[col, 'target']].dropna()
        if len(valid) < 20:
            continue

        try:
            pearson_r, pearson_p = pearsonr(valid[col], valid['target'])
            spearman_r, spearman_p = spearmanr(valid[col], valid['target'])

            correlations.append({
                'Feature': col,
                'Pearson_r': pearson_r,
                'Pearson_p': pearson_p,
                'Spearman_r': spearman_r,
                'Spearman_p': spearman_p,
                'Abs_Pearson': abs(pearson_r),
                'N_obs': len(valid)
            })
        except:
            pass

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Abs_Pearson', ascending=False)

    print(f"\n  Топ-{top_n} признаков по абсолютной корреляции (Pearson):")
    print("-" * 80)
    print(f"  {'Признак':<40} {'Pearson':>10} {'p-value':>10} {'N':>6}")
    print("-" * 80)

    for _, row in corr_df.head(top_n).iterrows():
        sig = "***" if row['Pearson_p'] < 0.001 else "**" if row['Pearson_p'] < 0.01 else "*" if row['Pearson_p'] < 0.05 else ""
        print(f"  {row['Feature']:<40} {row['Pearson_r']:>+10.3f} {row['Pearson_p']:>10.4f} {row['N_obs']:>6} {sig}")

    # Group by base feature
    print("\n  Лучший лаг для каждого базового признака:")
    print("-" * 80)

    base_features = set()
    for col in feature_cols:
        # Extract base name (remove _L0, _D1, _MA3, etc.)
        for suffix in ['_L', '_D', '_MA', '_STD']:
            if suffix in col:
                base = col.split(suffix)[0]
                base_features.add(base)
                break

    best_lags = []
    for base in sorted(base_features):
        # Find all lag variations
        lag_corrs = corr_df[corr_df['Feature'].str.startswith(f'{base}_L')]
        if len(lag_corrs) > 0:
            best = lag_corrs.iloc[0]
            best_lags.append({
                'Base': base,
                'Best_Feature': best['Feature'],
                'Correlation': best['Pearson_r'],
                'p_value': best['Pearson_p']
            })

    best_lags_df = pd.DataFrame(best_lags).sort_values('Correlation', key=abs, ascending=False)

    for _, row in best_lags_df.head(20).iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"  {row['Base']:<25} → {row['Best_Feature']:<20} r={row['Correlation']:>+.3f} {sig}")

    return corr_df, best_lags_df


def granger_causality_test(df, max_lag=6):
    """Test Granger causality for key indicators."""
    print("\n" + "=" * 80)
    print("4. ТЕСТ ПРИЧИННОСТИ ГРЕЙНДЖЕРА")
    print("=" * 80)

    from statsmodels.tsa.stattools import grangercausalitytests

    # Key indicators to test
    indicators = ['usd', 'ki', 'Ruonia', 'brent', 'reg_ppi', 'reg_wage',
                  'reg_retail', 'reg_agri_prices', 'nonprod', 'prod', 'serv']

    results = []

    print(f"\n  Тестируем: {indicators}")
    print(f"  H0: X не причина Грейнджера для инфляции")
    print("-" * 80)

    for indicator in indicators:
        if indicator not in df.columns:
            continue

        test_df = df[['mom', indicator]].dropna()
        if len(test_df) < 30:
            continue

        try:
            test_result = grangercausalitytests(test_df[['mom', indicator]], maxlag=max_lag, verbose=False)

            # Get minimum p-value across all lags
            min_pvalue = 1.0
            best_lag = 1
            for lag in range(1, max_lag + 1):
                pvalue = test_result[lag][0]['ssr_ftest'][1]
                if pvalue < min_pvalue:
                    min_pvalue = pvalue
                    best_lag = lag

            results.append({
                'Indicator': indicator,
                'Best_Lag': best_lag,
                'P_value': min_pvalue,
                'Significant': min_pvalue < 0.05
            })

            sig = "✓ ЗНАЧИМ" if min_pvalue < 0.05 else "✗"
            print(f"  {indicator:<20} лаг={best_lag} p={min_pvalue:.4f} {sig}")

        except Exception as e:
            print(f"  {indicator:<20} ОШИБКА: {str(e)[:30]}")

    granger_df = pd.DataFrame(results)

    print("\n  ВЫВОД по Грейнджеру:")
    significant = granger_df[granger_df['Significant']]
    if len(significant) > 0:
        print(f"  Значимые причины инфляции: {list(significant['Indicator'])}")
    else:
        print(f"  Нет статистически значимых опережающих индикаторов!")

    return granger_df


def mutual_information_analysis(df, top_n=30):
    """Calculate mutual information between features and target."""
    print("\n" + "=" * 80)
    print("5. ВЗАИМНАЯ ИНФОРМАЦИЯ (MUTUAL INFORMATION)")
    print("=" * 80)

    # Prepare data
    target = df['target']
    feature_cols = [c for c in df.columns if c not in ['target', 'month', 'quarter']
                    and not c.startswith('is_') and 'sin' not in c and 'cos' not in c]

    # Select features with enough data
    valid_features = []
    for col in feature_cols:
        valid = df[[col, 'target']].dropna()
        if len(valid) >= 50:
            valid_features.append(col)

    # Prepare X and y
    X = df[valid_features].copy()
    y = df['target'].copy()

    # Drop rows with NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    if len(X) < 30:
        print("  Недостаточно данных для анализа MI")
        return None

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Calculate MI
    mi_scores = mutual_info_regression(X_scaled, y, random_state=42)

    mi_df = pd.DataFrame({
        'Feature': valid_features,
        'MI_Score': mi_scores
    }).sort_values('MI_Score', ascending=False)

    print(f"\n  Топ-{top_n} признаков по взаимной информации:")
    print("-" * 60)
    for _, row in mi_df.head(top_n).iterrows():
        bar = "█" * int(row['MI_Score'] * 50)
        print(f"  {row['Feature']:<35} {row['MI_Score']:.3f} {bar}")

    return mi_df


def feature_importance_rf(df, top_n=30):
    """Calculate feature importance using Random Forest."""
    print("\n" + "=" * 80)
    print("6. ВАЖНОСТЬ ПРИЗНАКОВ (RANDOM FOREST)")
    print("=" * 80)

    # Use only lagged features (L0-L6) to reduce dimensionality
    feature_cols = [c for c in df.columns if '_L' in c and int(c.split('_L')[-1]) <= 6]
    feature_cols += ['month_sin', 'month_cos', 'is_jan', 'is_jul', 'is_dec']

    valid_features = [c for c in feature_cols if c in df.columns]

    X = df[valid_features].copy()
    y = df['target'].copy()

    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    if len(X) < 30:
        print("  Недостаточно данных")
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train RF
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_scaled, y)

    importance_df = pd.DataFrame({
        'Feature': valid_features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    print(f"\n  Топ-{top_n} признаков по важности RF:")
    print("-" * 60)
    for _, row in importance_df.head(top_n).iterrows():
        bar = "█" * int(row['Importance'] * 100)
        print(f"  {row['Feature']:<35} {row['Importance']:.3f} {bar}")

    return importance_df


def backtest_feature_sets(df):
    """Backtest different feature sets."""
    print("\n" + "=" * 80)
    print("7. БЭКТЕСТ МОДЕЛЕЙ С РАЗНЫМИ НАБОРАМИ ПРИЗНАКОВ")
    print("=" * 80)

    # Define feature sets to test
    feature_sets = {
        # Baseline: only AR features
        'AR_only': ['mom_L0', 'mom_L1', 'mom_L2', 'mom_L5', 'mom_L11',
                    'month_sin', 'month_cos', 'is_jan', 'is_jul'],

        # Federal macro
        'Federal_macro': ['mom_L0', 'mom_L1', 'mom_L2',
                          'usd_L1', 'usd_L2', 'usd_D1',
                          'ki_L1', 'ki_L2',
                          'brent_L1', 'brent_L2',
                          'month_sin', 'month_cos'],

        # Regional macro
        'Regional_macro': ['mom_L0', 'mom_L1', 'mom_L2',
                           'reg_ppi_L1', 'reg_ppi_L2',
                           'reg_wage_L1', 'reg_wage_D1',
                           'reg_agri_prices_L1',
                           'reg_retail_L1',
                           'month_sin', 'month_cos'],

        # Components
        'Components': ['mom_L0', 'mom_L1', 'mom_L2',
                       'nonprod_L0', 'nonprod_L1',
                       'prod_L0', 'prod_L1',
                       'serv_L0', 'serv_L1',
                       'month_sin', 'month_cos'],

        # Best correlated (will be filled dynamically)
        'Best_correlated': [],

        # All available
        'Kitchen_sink': [],

        # Minimal (just mom lags + seasonality)
        'Minimal': ['mom_L0', 'mom_L1', 'month_sin', 'month_cos'],
    }

    # Fill best_correlated with top features
    feature_cols = [c for c in df.columns if '_L' in c and not c.startswith('target')]
    corrs = []
    for col in feature_cols:
        valid = df[[col, 'target']].dropna()
        if len(valid) >= 30:
            r, p = pearsonr(valid[col], valid['target'])
            if p < 0.05:
                corrs.append((col, abs(r)))

    corrs.sort(key=lambda x: x[1], reverse=True)
    feature_sets['Best_correlated'] = [c[0] for c in corrs[:15]] + ['month_sin', 'month_cos']

    # Kitchen sink: all L0-L2 features
    feature_sets['Kitchen_sink'] = [c for c in df.columns
                                    if ('_L0' in c or '_L1' in c or '_L2' in c)
                                    and c in df.columns][:50]
    feature_sets['Kitchen_sink'] += ['month_sin', 'month_cos', 'is_jan', 'is_jul']

    # Run backtests
    test_dates = pd.date_range(TEST_START, df.index.max() - pd.DateOffset(months=1), freq='MS')

    results = []
    for test_date in test_dates:
        if test_date not in df.index:
            continue

        actual = df.loc[test_date, 'target']
        if pd.isna(actual):
            continue

        row = {'Date': test_date, 'Actual': actual}

        for name, features in feature_sets.items():
            avail = [f for f in features if f in df.columns]
            if len(avail) < 3:
                row[name] = np.nan
                continue

            train_df = df[df.index < test_date][['target'] + avail].dropna()
            if len(train_df) < MIN_TRAIN:
                row[name] = np.nan
                continue

            X_train = train_df[avail].values
            y_train = train_df['target'].values

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            test_row = df[avail].ffill().loc[[test_date]]
            if test_row.isna().any().any():
                row[name] = np.nan
                continue

            X_test_scaled = scaler.transform(test_row.values)

            # Use HuberRegressor (robust to outliers)
            model = HuberRegressor(epsilon=1.35)
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)[0]

            row[name] = pred

        results.append(row)

    results_df = pd.DataFrame(results)

    # Calculate metrics
    print(f"\n  Период тестирования: {TEST_START} — {df.index.max()}")
    print("-" * 80)
    print(f"  {'Набор признаков':<25} {'MAE':>8} {'vs AR':>10} {'KPI':>10} {'N_feat':>8}")
    print("-" * 80)

    metrics = []
    baseline_mae = None

    for name in feature_sets.keys():
        if name not in results_df.columns:
            continue

        valid = results_df[[name, 'Actual']].dropna()
        if len(valid) == 0:
            continue

        errors = (valid[name] - valid['Actual']).abs()
        mae = errors.mean()
        kpi = (errors <= 0.5).sum()
        total = len(valid)
        n_features = len([f for f in feature_sets[name] if f in df.columns])

        if baseline_mae is None:
            baseline_mae = mae

        vs = (mae - baseline_mae) / baseline_mae * 100

        print(f"  {name:<25} {mae:>8.3f} {vs:>+9.1f}% {kpi:>5}/{total} {n_features:>8}")

        metrics.append({
            'FeatureSet': name,
            'MAE': mae,
            'vs_Baseline': vs,
            'KPI_Hits': kpi,
            'Total': total,
            'N_Features': n_features
        })

    return pd.DataFrame(metrics), results_df


def generate_report(corr_df, best_lags_df, granger_df, mi_df, rf_df, backtest_metrics):
    """Generate final report with recommendations."""
    print("\n" + "=" * 80)
    print("8. ФИНАЛЬНЫЙ ОТЧЕТ")
    print("=" * 80)

    report = []
    report.append("# ИССЛЕДОВАНИЕ ПРИЗНАКОВ ДЛЯ ПРОГНОЗИРОВАНИЯ ИНФЛЯЦИИ КБР")
    report.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")
    report.append("## РЕЗЮМЕ")
    report.append("")

    # Best performing feature set
    if len(backtest_metrics) > 0:
        best = backtest_metrics.loc[backtest_metrics['MAE'].idxmin()]
        report.append(f"**Лучший набор признаков:** {best['FeatureSet']}")
        report.append(f"- MAE: {best['MAE']:.3f}")
        report.append(f"- KPI hits: {int(best['KPI_Hits'])}/{int(best['Total'])}")
        report.append("")

    # Top correlated features
    report.append("## ТОП-20 ПРИЗНАКОВ ПО КОРРЕЛЯЦИИ")
    report.append("")
    report.append("| Признак | Pearson r | p-value |")
    report.append("|---------|-----------|---------|")
    for _, row in corr_df.head(20).iterrows():
        sig = "***" if row['Pearson_p'] < 0.001 else "**" if row['Pearson_p'] < 0.01 else "*" if row['Pearson_p'] < 0.05 else ""
        report.append(f"| {row['Feature']} | {row['Pearson_r']:+.3f} | {row['Pearson_p']:.4f} {sig} |")
    report.append("")

    # Granger causality
    report.append("## ПРИЧИННОСТЬ ГРЕЙНДЖЕРА")
    report.append("")
    if granger_df is not None and len(granger_df) > 0:
        significant = granger_df[granger_df['Significant']]
        if len(significant) > 0:
            report.append("**Статистически значимые опережающие индикаторы:**")
            for _, row in significant.iterrows():
                report.append(f"- {row['Indicator']} (лаг {row['Best_Lag']}, p={row['P_value']:.4f})")
        else:
            report.append("**Нет статистически значимых опережающих индикаторов.**")
    report.append("")

    # Backtest results
    report.append("## РЕЗУЛЬТАТЫ БЭКТЕСТА")
    report.append("")
    report.append("| Набор признаков | MAE | vs Baseline | KPI | N признаков |")
    report.append("|-----------------|-----|-------------|-----|-------------|")
    for _, row in backtest_metrics.sort_values('MAE').iterrows():
        report.append(f"| {row['FeatureSet']} | {row['MAE']:.3f} | {row['vs_Baseline']:+.1f}% | {int(row['KPI_Hits'])}/{int(row['Total'])} | {int(row['N_Features'])} |")
    report.append("")

    # Key findings
    report.append("## КЛЮЧЕВЫЕ ВЫВОДЫ")
    report.append("")

    # Check if regional macro helps
    if 'Regional_macro' in backtest_metrics['FeatureSet'].values:
        reg = backtest_metrics[backtest_metrics['FeatureSet'] == 'Regional_macro'].iloc[0]
        ar = backtest_metrics[backtest_metrics['FeatureSet'] == 'AR_only'].iloc[0]
        if reg['MAE'] < ar['MAE']:
            report.append(f"1. ✓ Региональные макропоказатели УЛУЧШАЮТ прогноз на {ar['MAE'] - reg['MAE']:.3f}")
        else:
            report.append(f"1. ✗ Региональные макропоказатели НЕ УЛУЧШАЮТ прогноз (+{reg['MAE'] - ar['MAE']:.3f})")

    if 'Federal_macro' in backtest_metrics['FeatureSet'].values:
        fed = backtest_metrics[backtest_metrics['FeatureSet'] == 'Federal_macro'].iloc[0]
        ar = backtest_metrics[backtest_metrics['FeatureSet'] == 'AR_only'].iloc[0]
        if fed['MAE'] < ar['MAE']:
            report.append(f"2. ✓ Федеральные макропоказатели УЛУЧШАЮТ прогноз на {ar['MAE'] - fed['MAE']:.3f}")
        else:
            report.append(f"2. ✗ Федеральные макропоказатели НЕ УЛУЧШАЮТ прогноз (+{fed['MAE'] - ar['MAE']:.3f})")

    if 'Components' in backtest_metrics['FeatureSet'].values:
        comp = backtest_metrics[backtest_metrics['FeatureSet'] == 'Components'].iloc[0]
        ar = backtest_metrics[backtest_metrics['FeatureSet'] == 'AR_only'].iloc[0]
        if comp['MAE'] < ar['MAE']:
            report.append(f"3. ✓ Компоненты инфляции УЛУЧШАЮТ прогноз на {ar['MAE'] - comp['MAE']:.3f}")
        else:
            report.append(f"3. ✗ Компоненты инфляции НЕ УЛУЧШАЮТ прогноз (+{comp['MAE'] - ar['MAE']:.3f})")

    report.append("")
    report.append("## РЕКОМЕНДАЦИИ")
    report.append("")

    best_set = backtest_metrics.loc[backtest_metrics['MAE'].idxmin(), 'FeatureSet']
    report.append(f"**Рекомендуемый набор признаков:** {best_set}")
    report.append("")
    report.append("**Признаки, которые стоит использовать:**")

    # Get best features
    if corr_df is not None:
        top_features = corr_df[corr_df['Pearson_p'] < 0.01].head(10)['Feature'].tolist()
        for f in top_features:
            report.append(f"- {f}")

    report_text = "\n".join(report)

    # Save report
    report_path = RESULTS_DIR / 'feature_research_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n  Отчет сохранен: {report_path}")
    print("\n" + report_text)

    return report_text


def main():
    print("╔" + "═" * 78 + "╗")
    print("║" + " КОМПЛЕКСНОЕ ИССЛЕДОВАНИЕ ПРИЗНАКОВ ДЛЯ ПРОГНОЗИРОВАНИЯ ИНФЛЯЦИИ ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # 1. Load all data
    df = load_all_data()

    # 2. Create features
    df = create_all_features(df)

    # 3. Correlation analysis
    corr_df, best_lags_df = analyze_correlations(df)

    # 4. Granger causality
    granger_df = granger_causality_test(df)

    # 5. Mutual information
    mi_df = mutual_information_analysis(df)

    # 6. Feature importance (RF)
    rf_df = feature_importance_rf(df)

    # 7. Backtest different feature sets
    backtest_metrics, backtest_results = backtest_feature_sets(df)

    # 8. Generate report
    report = generate_report(corr_df, best_lags_df, granger_df, mi_df, rf_df, backtest_metrics)

    # Save detailed results
    corr_df.to_csv(RESULTS_DIR / 'feature_correlations.csv', index=False)
    backtest_metrics.to_csv(RESULTS_DIR / 'feature_backtest_metrics.csv', index=False)

    print("\n" + "=" * 80)
    print("ИССЛЕДОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)

    return df, corr_df, granger_df, mi_df, rf_df, backtest_metrics


if __name__ == '__main__':
    results = main()
