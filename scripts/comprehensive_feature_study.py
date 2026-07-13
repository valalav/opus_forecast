#!/usr/bin/env python3
"""
КОМПЛЕКСНОЕ ИССЛЕДОВАНИЕ ПРИЗНАКОВ ДЛЯ ПРОГНОЗИРОВАНИЯ ИНФЛЯЦИИ КБР
====================================================================

Особенности:
- Учёт разной длины временных рядов для каждого признака
- Корреляции для 3 горизонтов: h=1, h=2, h=12
- Матрица взаимных корреляций
- Бэктест на всех горизонтах

Данные:
- ИБВЭД квартальный (20) с 2019
- ИБВЭД месячный (21) с 2022
- Региональные месячные с 2016
- Федеральные макро с 2010

Автор: Claude Code
Дата: 2025-12-28
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'archive' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Test period
TEST_START = '2022-01-01'
MIN_TRAIN = 24


def load_all_data():
    """Load and merge all data sources with proper handling."""
    print("=" * 80)
    print("1. ЗАГРУЗКА ДАННЫХ")
    print("=" * 80)

    # 1. Inflation + federal macro
    infl = pd.read_csv(DATA_DIR / 'inflation_data.csv', sep=';', decimal=',', on_bad_lines='skip')
    infl['Date'] = pd.to_datetime(infl['Date'], format='%d.%m.%Y')
    infl['Date'] = infl['Date'].dt.to_period('M').dt.to_timestamp()
    infl = infl.set_index('Date').sort_index()
    print(f"  Инфляция + федеральные: {len(infl)} точек ({infl.index.min().strftime('%Y-%m')} — {infl.index.max().strftime('%Y-%m')})")

    # 2. Brent
    brent = pd.read_csv(DATA_DIR / 'brent_prices.csv')
    brent['Date'] = pd.to_datetime(brent['Date'])
    brent = brent.set_index('Date').sort_index()
    print(f"  Brent: {len(brent)} точек")

    # 3. Regional monthly
    regional = pd.read_csv(DATA_DIR / 'month.csv', sep=';', decimal=',')
    regional['Date'] = pd.to_datetime(regional['Date'], format='%d.%m.%Y')
    regional = regional.set_index('Date').sort_index()

    reg_names = {
        '1': 'reg_ind_prod', '2': 'reg_shipped', '3': 'reg_construction',
        '6': 'reg_retail', '7': 'reg_services', '8': 'reg_profit',
        '9': 'reg_payables', '10': 'reg_receivables', '11': 'reg_ppi',
        '12': 'reg_agri_prices', '13': 'reg_invest_prices',
        '18': 'reg_wage', '19': 'reg_wage_agri',
        '21': 'ibved_m',  # ИБВЭД месячный с 2022!
    }
    existing = [c for c in reg_names.keys() if c in regional.columns]
    regional = regional[existing].rename(columns={k: reg_names[k] for k in existing})
    print(f"  Региональные месячные: {len(regional)} точек, {len(regional.columns)} показателей")

    # Check IBVED monthly availability
    if 'ibved_m' in regional.columns:
        ibved_m_valid = regional['ibved_m'].dropna()
        print(f"    ИБВЭД месячный (21): {len(ibved_m_valid)} точек ({ibved_m_valid.index.min().strftime('%Y-%m')} — {ibved_m_valid.index.max().strftime('%Y-%m')})")

    # 4. Quarterly data
    quarterly = pd.read_csv(DATA_DIR / 'quart.csv', sep=';', decimal=',')
    quarterly['Date'] = pd.to_datetime(quarterly['Data'], format='%d.%m.%Y', errors='coerce')
    quarterly = quarterly.dropna(subset=['Date'])
    quarterly = quarterly.set_index('Date').sort_index()

    quart_names = {
        '14': 'reg_housing_primary', '15': 'reg_housing_secondary',
        '16': 'reg_income_nom', '17': 'reg_income_real',
        '20': 'ibved_q',  # ИБВЭД квартальный с 2019!
    }
    existing = [c for c in quart_names.keys() if c in quarterly.columns]
    quarterly = quarterly[existing].rename(columns={k: quart_names[k] for k in existing})
    quarterly = quarterly[~quarterly.index.duplicated(keep='last')]

    # Check IBVED quarterly availability
    if 'ibved_q' in quarterly.columns:
        ibved_q_valid = quarterly['ibved_q'].dropna()
        print(f"    ИБВЭД квартальный (20): {len(ibved_q_valid)} точек ({ibved_q_valid.index.min().strftime('%Y-%m')} — {ibved_q_valid.index.max().strftime('%Y-%m')})")

    # Resample quarterly to monthly
    quarterly = quarterly.resample('MS').ffill()
    print(f"  Квартальные (→ месяц): {len(quarterly)} точек")

    # Merge all
    df = infl[['mom', 'Nonprod', 'Prod', 'Serv', 'usd_nom_i', 'Ki_i', 'Ruonia', 'Ki']].copy()
    df = df.rename(columns={
        'Nonprod': 'nonprod', 'Prod': 'prod', 'Serv': 'serv',
        'usd_nom_i': 'usd', 'Ki_i': 'ki_i', 'Ki': 'ki'
    })

    df = df.join(brent[['brent', 'brent_pct']], how='left')
    df = df.join(regional, how='left')
    df = df.join(quarterly, how='left')

    # Filter to 2016+
    df = df[df.index >= '2016-01-01']

    print(f"\n  Объединённый датасет: {len(df)} точек ({df.index.min().strftime('%Y-%m')} — {df.index.max().strftime('%Y-%m')})")

    # Show data availability for each column
    print("\n  Доступность данных по признакам:")
    print("-" * 60)

    availability = []
    for col in df.columns:
        valid = df[col].dropna()
        if len(valid) > 0:
            availability.append({
                'Feature': col,
                'N': len(valid),
                'Start': valid.index.min().strftime('%Y-%m'),
                'End': valid.index.max().strftime('%Y-%m'),
                'Missing%': (1 - len(valid) / len(df)) * 100
            })

    avail_df = pd.DataFrame(availability).sort_values('N', ascending=False)
    for _, row in avail_df.iterrows():
        print(f"    {row['Feature']:<25} N={row['N']:>3} ({row['Start']} — {row['End']}) missing={row['Missing%']:.0f}%")

    return df, avail_df


def create_features(df, max_lag=12):
    """Create all features with lags."""
    print("\n" + "=" * 80)
    print("2. СОЗДАНИЕ ПРИЗНАКОВ")
    print("=" * 80)

    # Base columns
    base_cols = [c for c in df.columns if c != 'mom']

    # Create lagged features
    for col in ['mom'] + base_cols:
        if df[col].isna().sum() > len(df) * 0.7:  # Skip if >70% missing
            continue

        # Lags
        for lag in range(0, max_lag + 1):
            df[f'{col}_L{lag}'] = df[col].shift(lag)

        # Differences
        for d in [1, 3, 6, 12]:
            df[f'{col}_D{d}'] = df[col].diff(d)

        # Moving averages
        for w in [3, 6]:
            df[f'{col}_MA{w}'] = df[col].rolling(w).mean()

        # Volatility
        df[f'{col}_STD3'] = df[col].rolling(3).std()

    # Seasonality
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['is_jan'] = (df['month'] == 1).astype(int)
    df['is_jul'] = (df['month'] == 7).astype(int)
    df['is_dec'] = (df['month'] == 12).astype(int)

    print(f"  Создано {len(df.columns)} колонок")

    return df


def calc_correlations_by_horizon(df):
    """Calculate correlations for each forecast horizon."""
    print("\n" + "=" * 80)
    print("3. КОРРЕЛЯЦИИ ПО ГОРИЗОНТАМ")
    print("=" * 80)

    results = {}

    for horizon, horizon_name in [(1, 'h=1'), (2, 'h=2'), (12, 'h=12')]:
        print(f"\n  --- {horizon_name} ---")

        # Target: future inflation
        if horizon == 12:
            # For h=12: average inflation over next 12 months
            target = df['mom'].rolling(12).mean().shift(-12)
        else:
            target = df['mom'].shift(-horizon)

        df_temp = df.copy()
        df_temp['target'] = target

        # Calculate correlations
        feature_cols = [c for c in df.columns if c not in ['mom', 'month', 'target']
                       and not c.startswith('is_')]

        correlations = []
        for col in feature_cols:
            valid = df_temp[[col, 'target']].dropna()
            if len(valid) < 20:
                continue

            try:
                r, p = pearsonr(valid[col], valid['target'])
                correlations.append({
                    'Feature': col,
                    'Horizon': horizon_name,
                    'Pearson_r': r,
                    'P_value': p,
                    'Abs_r': abs(r),
                    'N': len(valid),
                    'Significant': p < 0.05
                })
            except:
                pass

        corr_df = pd.DataFrame(correlations).sort_values('Abs_r', ascending=False)
        results[horizon_name] = corr_df

        # Print top 15
        print(f"\n  Топ-15 признаков для {horizon_name}:")
        print(f"  {'Признак':<35} {'r':>8} {'p':>8} {'N':>5} {'Знач.':>5}")
        print("  " + "-" * 65)

        for _, row in corr_df.head(15).iterrows():
            sig = "✓" if row['Significant'] else ""
            print(f"  {row['Feature']:<35} {row['Pearson_r']:>+8.3f} {row['P_value']:>8.4f} {row['N']:>5} {sig:>5}")

        # Save to CSV
        corr_df.to_csv(RESULTS_DIR / f'feature_correlations_{horizon_name.replace("=", "")}.csv', index=False)

    return results


def calc_mutual_correlations(df):
    """Calculate mutual correlations between key features."""
    print("\n" + "=" * 80)
    print("4. ВЗАИМНЫЕ КОРРЕЛЯЦИИ ПРИЗНАКОВ")
    print("=" * 80)

    # Select key features (L0 and L1 versions of main indicators)
    key_features = [
        'mom_L0', 'mom_L1',
        'ki_i_L0', 'ki_L0', 'ki_L6',
        'Ruonia_L0', 'Ruonia_L2', 'Ruonia_D1',
        'usd_L0', 'usd_L2',
        'brent_L0', 'brent_L5', 'brent_STD3',
        'prod_L0', 'nonprod_L0', 'serv_L0',
        'reg_ppi_L0', 'reg_ppi_L3',
        'ibved_q_L0', 'ibved_q_L1', 'ibved_q_MA3',
        'ibved_m_L0', 'ibved_m_L1',
    ]

    # Filter to existing features
    available = [f for f in key_features if f in df.columns]
    print(f"  Доступно {len(available)} из {len(key_features)} ключевых признаков")

    # Calculate correlation matrix
    corr_matrix = df[available].corr()

    # Find highly correlated pairs (|r| > 0.7)
    print("\n  Высокая корреляция (|r| > 0.7):")
    print("  " + "-" * 60)

    high_corr_pairs = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.7:
                high_corr_pairs.append({
                    'Feature1': available[i],
                    'Feature2': available[j],
                    'Correlation': r
                })
                print(f"  {available[i]:<25} ↔ {available[j]:<25} r={r:+.3f}")

    if len(high_corr_pairs) == 0:
        print("  Нет пар с |r| > 0.7")

    # Save correlation matrix
    corr_matrix.to_csv(RESULTS_DIR / 'feature_mutual_correlations.csv')

    return corr_matrix, high_corr_pairs


def run_backtests(df, correlations):
    """Run backtests for different feature sets on all horizons."""
    print("\n" + "=" * 80)
    print("5. БЭКТЕСТЫ НА ВСЕХ ГОРИЗОНТАХ")
    print("=" * 80)

    # Define feature sets
    feature_sets = {
        'Baseline_AR': [
            'mom_L0', 'mom_L1', 'mom_L2', 'mom_L5', 'mom_L11',
            'month_sin', 'month_cos', 'is_jan', 'is_jul'
        ],

        'Federal_Macro': [
            'mom_L0', 'mom_L1', 'mom_L2',
            'ki_L6', 'ki_D6',
            'Ruonia_L2', 'Ruonia_D1',
            'usd_L2', 'usd_D1',
            'brent_L5', 'brent_STD3',
            'month_sin', 'month_cos'
        ],

        'Components': [
            'mom_L0', 'mom_L1',
            'prod_L0', 'prod_L1', 'prod_D3',
            'nonprod_L0', 'nonprod_L3',
            'serv_L0', 'serv_L1',
            'month_sin', 'month_cos'
        ],

        'IBVED_Quarterly': [
            'mom_L0', 'mom_L1', 'mom_L2',
            'ibved_q_L1', 'ibved_q_MA3',
            'reg_ppi_L3',
            'month_sin', 'month_cos'
        ],

        'IBVED_Monthly': [
            'mom_L0', 'mom_L1', 'mom_L2',
            'ibved_m_L0', 'ibved_m_L1',
            'reg_ppi_L3',
            'month_sin', 'month_cos'
        ],

        'Best_Combined': [
            'mom_L0', 'mom_L1', 'mom_L2',
            'ki_L6', 'Ruonia_D1',
            'ibved_q_L1',
            'prod_L0', 'serv_L1',
            'month_sin', 'month_cos'
        ],

        'Minimal': [
            'mom_L0', 'mom_L1',
            'month_sin', 'month_cos'
        ],
    }

    all_results = []

    for horizon in [1, 2, 12]:
        horizon_name = f'h={horizon}'
        print(f"\n  === ГОРИЗОНТ {horizon_name} ===")

        # Create target
        if horizon == 12:
            target = df['mom'].rolling(12).mean().shift(-12)
        else:
            target = df['mom'].shift(-horizon)

        df_bt = df.copy()
        df_bt['target'] = target

        # Test dates
        if horizon == 12:
            test_end = df.index.max() - pd.DateOffset(months=12)
        else:
            test_end = df.index.max() - pd.DateOffset(months=horizon)

        test_dates = pd.date_range(TEST_START, test_end, freq='MS')
        test_dates = [d for d in test_dates if d in df_bt.index]

        print(f"  Период теста: {TEST_START} — {test_end.strftime('%Y-%m')} ({len(test_dates)} точек)")

        results_horizon = []

        for name, features in feature_sets.items():
            # Filter to available features
            avail = [f for f in features if f in df_bt.columns]

            if len(avail) < 3:
                print(f"    {name}: недостаточно признаков ({len(avail)}/{len(features)})")
                continue

            predictions = []
            actuals = []

            for test_date in test_dates:
                actual = df_bt.loc[test_date, 'target']
                if pd.isna(actual):
                    continue

                # Training data
                train_df = df_bt[df_bt.index < test_date][['target'] + avail].dropna()

                if len(train_df) < MIN_TRAIN:
                    continue

                X_train = train_df[avail].values
                y_train = train_df['target'].values

                # Test data
                test_row = df_bt[avail].ffill().loc[[test_date]]
                if test_row.isna().any().any():
                    continue

                X_test = test_row.values

                # Scale and predict
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = HuberRegressor(epsilon=1.35)
                model.fit(X_train_scaled, y_train)
                pred = model.predict(X_test_scaled)[0]

                predictions.append(pred)
                actuals.append(actual)

            if len(predictions) < 5:
                continue

            # Calculate metrics
            errors = np.array(actuals) - np.array(predictions)
            abs_errors = np.abs(errors)

            mae = np.mean(abs_errors)
            rmse = np.sqrt(np.mean(errors ** 2))
            kpi_hits = np.sum(abs_errors <= 0.5)
            max_err = np.max(abs_errors)

            results_horizon.append({
                'FeatureSet': name,
                'Horizon': horizon_name,
                'MAE': mae,
                'RMSE': rmse,
                'KPI_Hits': kpi_hits,
                'Total': len(predictions),
                'KPI%': kpi_hits / len(predictions) * 100,
                'MaxError': max_err,
                'N_Features': len(avail)
            })

        # Sort and print
        results_df = pd.DataFrame(results_horizon).sort_values('MAE')

        print(f"\n  {'Набор':<20} {'MAE':>8} {'RMSE':>8} {'KPI':>8} {'MaxErr':>8} {'N':>4}")
        print("  " + "-" * 65)

        baseline_mae = results_df[results_df['FeatureSet'] == 'Baseline_AR']['MAE'].values
        baseline_mae = baseline_mae[0] if len(baseline_mae) > 0 else None

        for _, row in results_df.iterrows():
            vs = ""
            if baseline_mae and row['FeatureSet'] != 'Baseline_AR':
                diff = (row['MAE'] - baseline_mae) / baseline_mae * 100
                vs = f" ({diff:+.1f}%)"

            print(f"  {row['FeatureSet']:<20} {row['MAE']:>8.3f} {row['RMSE']:>8.3f} {int(row['KPI_Hits']):>3}/{int(row['Total'])} {row['MaxError']:>8.2f} {int(row['N_Features']):>4}{vs}")

        all_results.extend(results_horizon)

    # Save all results
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(RESULTS_DIR / 'backtest_all_horizons.csv', index=False)

    return all_results_df


def generate_final_report(df, correlations, backtest_results):
    """Generate final markdown report."""
    print("\n" + "=" * 80)
    print("6. ФИНАЛЬНЫЙ ОТЧЁТ")
    print("=" * 80)

    report = []
    report.append("# ФИНАЛЬНЫЙ ОТЧЁТ: Исследование признаков для прогнозирования инфляции КБР")
    report.append(f"\nДата: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"Период данных: 2016-01 — {df.index.max().strftime('%Y-%m')}")
    report.append(f"Период теста: 2022-01 — {df.index.max().strftime('%Y-%m')}")

    # Summary table
    report.append("\n## СВОДНАЯ ТАБЛИЦА БЭКТЕСТОВ")
    report.append("\n| Набор признаков | h=1 MAE | h=2 MAE | h=12 MAE | Рекомендация |")
    report.append("|-----------------|---------|---------|----------|--------------|")

    bt = backtest_results
    feature_sets = bt['FeatureSet'].unique()

    recommendations = {}

    for fs in feature_sets:
        row = f"| {fs} |"
        fs_data = bt[bt['FeatureSet'] == fs]

        maes = {}
        for h in ['h=1', 'h=2', 'h=12']:
            h_data = fs_data[fs_data['Horizon'] == h]
            if len(h_data) > 0:
                mae = h_data['MAE'].values[0]
                maes[h] = mae
                row += f" {mae:.3f} |"
            else:
                row += " — |"

        # Recommendation logic
        if len(maes) >= 2:
            baseline = bt[(bt['FeatureSet'] == 'Baseline_AR')]
            baseline_h1 = baseline[baseline['Horizon'] == 'h=1']['MAE'].values

            if len(baseline_h1) > 0 and 'h=1' in maes:
                if maes['h=1'] < baseline_h1[0] * 0.97:  # >3% improvement
                    row += " ✓ Использовать |"
                    recommendations[fs] = "✓"
                elif maes['h=1'] > baseline_h1[0] * 1.03:  # >3% worse
                    row += " ✗ Не использовать |"
                    recommendations[fs] = "✗"
                else:
                    row += " ~ Без разницы |"
                    recommendations[fs] = "~"
            else:
                row += " ? |"
        else:
            row += " ? |"

        report.append(row)

    # Best features by horizon
    report.append("\n## ЛУЧШИЕ ПРИЗНАКИ ПО ГОРИЗОНТАМ")

    for horizon_name, corr_df in correlations.items():
        report.append(f"\n### {horizon_name}")
        report.append("\n| Признак | Корреляция | p-value | N |")
        report.append("|---------|------------|---------|---|")

        sig = corr_df[corr_df['Significant']].head(10)
        for _, row in sig.iterrows():
            report.append(f"| {row['Feature']} | {row['Pearson_r']:+.3f} | {row['P_value']:.4f} | {int(row['N'])} |")

    # Key findings
    report.append("\n## КЛЮЧЕВЫЕ ВЫВОДЫ")
    report.append("")

    # Find best model for each horizon
    for h in ['h=1', 'h=2', 'h=12']:
        h_data = bt[bt['Horizon'] == h].sort_values('MAE')
        if len(h_data) > 0:
            best = h_data.iloc[0]
            baseline = bt[(bt['FeatureSet'] == 'Baseline_AR') & (bt['Horizon'] == h)]
            if len(baseline) > 0:
                diff = (best['MAE'] - baseline['MAE'].values[0]) / baseline['MAE'].values[0] * 100
                report.append(f"- **{h}**: Лучший — {best['FeatureSet']} (MAE {best['MAE']:.3f}, {diff:+.1f}% vs baseline)")

    # IBVED analysis
    report.append("")
    ibved_q = bt[bt['FeatureSet'] == 'IBVED_Quarterly']
    ibved_m = bt[bt['FeatureSet'] == 'IBVED_Monthly']
    baseline = bt[bt['FeatureSet'] == 'Baseline_AR']

    if len(ibved_q) > 0 and len(baseline) > 0:
        for h in ['h=1', 'h=2']:
            iq = ibved_q[ibved_q['Horizon'] == h]
            bl = baseline[baseline['Horizon'] == h]
            if len(iq) > 0 and len(bl) > 0:
                diff = (iq['MAE'].values[0] - bl['MAE'].values[0]) / bl['MAE'].values[0] * 100
                verdict = "✓ ПОМОГАЕТ" if diff < -3 else "✗ НЕ ПОМОГАЕТ" if diff > 3 else "~ БЕЗ РАЗНИЦЫ"
                report.append(f"- **ИБВЭД квартальный** на {h}: {diff:+.1f}% {verdict}")

    if len(ibved_m) > 0 and len(baseline) > 0:
        for h in ['h=1', 'h=2']:
            im = ibved_m[ibved_m['Horizon'] == h]
            bl = baseline[baseline['Horizon'] == h]
            if len(im) > 0 and len(bl) > 0:
                diff = (im['MAE'].values[0] - bl['MAE'].values[0]) / bl['MAE'].values[0] * 100
                verdict = "✓ ПОМОГАЕТ" if diff < -3 else "✗ НЕ ПОМОГАЕТ" if diff > 3 else "~ БЕЗ РАЗНИЦЫ"
                report.append(f"- **ИБВЭД месячный** на {h}: {diff:+.1f}% {verdict}")

    # Recommendations
    report.append("\n## РЕКОМЕНДАЦИИ")
    report.append("")

    good = [k for k, v in recommendations.items() if v == "✓"]
    bad = [k for k, v in recommendations.items() if v == "✗"]

    if good:
        report.append("**Рекомендуется использовать:**")
        for g in good:
            report.append(f"- {g}")

    if bad:
        report.append("\n**Не рекомендуется:**")
        for b in bad:
            report.append(f"- {b}")

    report.append("\n---")
    report.append("\n*Критерий: улучшение MAE >3% vs Baseline_AR*")

    # Save report
    report_text = "\n".join(report)
    report_path = RESULTS_DIR / 'FEATURE_RESEARCH_FINAL.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n  Отчёт сохранён: {report_path}")
    print("\n" + report_text)

    return report_text


def main():
    print("╔" + "═" * 78 + "╗")
    print("║" + " КОМПЛЕКСНОЕ ИССЛЕДОВАНИЕ ПРИЗНАКОВ ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    # 1. Load data
    df, availability = load_all_data()

    # 2. Create features
    df = create_features(df)

    # 3. Calculate correlations by horizon
    correlations = calc_correlations_by_horizon(df)

    # 4. Mutual correlations
    corr_matrix, high_corr = calc_mutual_correlations(df)

    # 5. Backtests
    backtest_results = run_backtests(df, correlations)

    # 6. Final report
    report = generate_final_report(df, correlations, backtest_results)

    print("\n" + "=" * 80)
    print("ИССЛЕДОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 80)

    return df, correlations, backtest_results


if __name__ == '__main__':
    results = main()
