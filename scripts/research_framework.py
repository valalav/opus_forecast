#!/usr/bin/env python3
"""
МАСШТАБНЫЙ ИССЛЕДОВАТЕЛЬСКИЙ ФРЕЙМВОРК
======================================

Цель: раз и навсегда определить лучшую комбинацию признаков и моделей
для прогнозирования инфляции КБР.

Этапы:
1. Загрузка и подготовка всех данных
2. Генерация всех возможных признаков
3. Feature Selection (5 методов)
4. Тестирование моделей (15+ моделей)
5. Оптимизация гиперпараметров (Optuna)
6. Финальный отчёт

Модели:
- Линейные: Ridge, Lasso, ElasticNet, Huber, BayesianRidge, SGD
- Tree-based: RandomForest, ExtraTrees, GradientBoosting, XGBoost, LightGBM, CatBoost
- Ensemble: Stacking, Voting
- Probabilistic: NGBoost

Автор: Claude Code
Дата: 2025-12-28
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge,
    SGDRegressor, Lars, LassoLars
)
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    StackingRegressor, VotingRegressor, AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import (
    RFE, SelectFromModel, mutual_info_regression,
    f_regression, SelectKBest
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

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

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'archive' / 'results' / 'research'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Constants
TEST_START = '2022-01-01'
MIN_TRAIN = 36
RANDOM_STATE = 42


class DataLoader:
    """Load and prepare all data sources."""

    def __init__(self):
        self.df = None
        self.availability = None

    def load(self):
        """Load all data."""
        print("=" * 80)
        print("ЗАГРУЗКА ДАННЫХ")
        print("=" * 80)

        # 1. Inflation + federal macro
        infl = pd.read_csv(DATA_DIR / 'inflation_data.csv', sep=';', decimal=',', on_bad_lines='skip')
        infl['Date'] = pd.to_datetime(infl['Date'], format='%d.%m.%Y')
        infl['Date'] = infl['Date'].dt.to_period('M').dt.to_timestamp()
        infl = infl.set_index('Date').sort_index()

        # 2. Brent
        brent = pd.read_csv(DATA_DIR / 'brent_prices.csv')
        brent['Date'] = pd.to_datetime(brent['Date'])
        brent = brent.set_index('Date').sort_index()

        # 3. Regional monthly
        regional = pd.read_csv(DATA_DIR / 'month.csv', sep=';', decimal=',')
        regional['Date'] = pd.to_datetime(regional['Date'], format='%d.%m.%Y')
        regional = regional.set_index('Date').sort_index()

        reg_names = {
            '1': 'reg_ind_prod', '2': 'reg_shipped', '3': 'reg_construction',
            '6': 'reg_retail', '7': 'reg_services', '8': 'reg_profit',
            '9': 'reg_payables', '10': 'reg_receivables', '11': 'reg_ppi',
            '12': 'reg_agri_prices', '13': 'reg_invest_prices',
            '18': 'reg_wage', '19': 'reg_wage_agri', '21': 'ibved_m',
        }
        existing = [c for c in reg_names.keys() if c in regional.columns]
        regional = regional[existing].rename(columns={k: reg_names[k] for k in existing})

        # 4. Quarterly
        quarterly = pd.read_csv(DATA_DIR / 'quart.csv', sep=';', decimal=',')
        quarterly['Date'] = pd.to_datetime(quarterly['Data'], format='%d.%m.%Y', errors='coerce')
        quarterly = quarterly.dropna(subset=['Date'])
        quarterly = quarterly.set_index('Date').sort_index()

        quart_names = {
            '14': 'reg_housing_primary', '15': 'reg_housing_secondary',
            '16': 'reg_income_nom', '17': 'reg_income_real', '20': 'ibved_q',
        }
        existing = [c for c in quart_names.keys() if c in quarterly.columns]
        quarterly = quarterly[existing].rename(columns={k: quart_names[k] for k in existing})
        quarterly = quarterly[~quarterly.index.duplicated(keep='last')]
        quarterly = quarterly.resample('MS').ffill()

        # Merge
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

        self.df = df
        print(f"  Загружено {len(df)} точек, {len(df.columns)} базовых признаков")

        return df


class FeatureEngineer:
    """Create comprehensive feature set."""

    def __init__(self, max_lag=12):
        self.max_lag = max_lag
        self.feature_names = []

    def transform(self, df):
        """Create all features."""
        print("\n" + "=" * 80)
        print("ГЕНЕРАЦИЯ ПРИЗНАКОВ")
        print("=" * 80)

        base_cols = list(df.columns)

        for col in base_cols:
            if df[col].isna().sum() > len(df) * 0.7:
                continue

            # Lags
            for lag in range(0, self.max_lag + 1):
                df[f'{col}_L{lag}'] = df[col].shift(lag)

            # Differences
            for d in [1, 3, 6, 12]:
                df[f'{col}_D{d}'] = df[col].diff(d)

            # Moving averages
            for w in [3, 6, 12]:
                df[f'{col}_MA{w}'] = df[col].rolling(w).mean()

            # Volatility
            for w in [3, 6]:
                df[f'{col}_STD{w}'] = df[col].rolling(w).std()

            # Momentum
            df[f'{col}_MOM3'] = df[col] / df[col].shift(3) - 1
            df[f'{col}_MOM6'] = df[col] / df[col].shift(6) - 1

        # Seasonality
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df.index.quarter / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df.index.quarter / 4)

        # Calendar dummies
        df['is_jan'] = (df['month'] == 1).astype(int)
        df['is_jul'] = (df['month'] == 7).astype(int)
        df['is_dec'] = (df['month'] == 12).astype(int)
        df['is_q1'] = (df.index.quarter == 1).astype(int)
        df['is_q4'] = (df.index.quarter == 4).astype(int)

        # Interactions (top correlated features)
        if 'mom_L0' in df.columns and 'ki_i_L0' in df.columns:
            df['mom_x_ki'] = df['mom_L0'] * df['ki_i_L0']
        if 'prod_L0' in df.columns and 'nonprod_L0' in df.columns:
            df['prod_x_nonprod'] = df['prod_L0'] * df['nonprod_L0']

        self.feature_names = [c for c in df.columns if c not in ['mom', 'month']]

        print(f"  Создано {len(df.columns)} признаков")

        return df


class ModelTester:
    """Test multiple models with different feature sets."""

    def __init__(self, test_start=TEST_START, min_train=MIN_TRAIN):
        self.test_start = test_start
        self.min_train = min_train
        self.results = []

    def get_models(self):
        """Get all models to test."""
        models = {
            # Linear models
            'Ridge': Ridge(alpha=1.0),
            'Ridge_a10': Ridge(alpha=10.0),
            'Ridge_a01': Ridge(alpha=0.1),
            'Lasso': Lasso(alpha=0.1, max_iter=5000),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
            'Huber': HuberRegressor(epsilon=1.35),
            'Huber_e2': HuberRegressor(epsilon=2.0),
            'BayesianRidge': BayesianRidge(),
            'SGD': SGDRegressor(max_iter=5000, random_state=RANDOM_STATE),

            # Tree-based
            'RF': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
            'RF_deep': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
            'GradBoost': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=RANDOM_STATE),
            'AdaBoost': AdaBoostRegressor(n_estimators=50, random_state=RANDOM_STATE),

            # Other
            'KNN_5': KNeighborsRegressor(n_neighbors=5),
            'KNN_10': KNeighborsRegressor(n_neighbors=10),
            'SVR': SVR(kernel='rbf', C=1.0),
        }

        # Add optional models
        if HAS_XGBOOST:
            models['XGBoost'] = XGBRegressor(n_estimators=100, max_depth=3, random_state=RANDOM_STATE, verbosity=0)
            models['XGBoost_deep'] = XGBRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, verbosity=0)

        if HAS_LIGHTGBM:
            models['LightGBM'] = LGBMRegressor(n_estimators=100, max_depth=3, random_state=RANDOM_STATE, verbose=-1)
            models['LightGBM_deep'] = LGBMRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE, verbose=-1)

        if HAS_CATBOOST:
            models['CatBoost'] = CatBoostRegressor(iterations=100, depth=3, random_state=RANDOM_STATE, verbose=0)

        if HAS_NGBOOST:
            models['NGBoost'] = NGBRegressor(n_estimators=100, random_state=RANDOM_STATE, verbose=False)

        return models

    def get_feature_sets(self, df):
        """Define feature sets to test."""
        # Get available features
        all_features = [c for c in df.columns if c not in ['mom', 'month', 'target']]

        feature_sets = {
            # Minimal baselines
            'Minimal_2': ['mom_L0', 'mom_L1'],
            'Minimal_season': ['mom_L0', 'mom_L1', 'month_sin', 'month_cos'],

            # AR variants
            'AR_3': ['mom_L0', 'mom_L1', 'mom_L2', 'month_sin', 'month_cos'],
            'AR_6': ['mom_L0', 'mom_L1', 'mom_L2', 'mom_L3', 'mom_L5', 'mom_L6', 'month_sin', 'month_cos'],
            'AR_12': ['mom_L0', 'mom_L1', 'mom_L2', 'mom_L5', 'mom_L11', 'month_sin', 'month_cos'],
            'AR_full': [f'mom_L{i}' for i in range(13)] + ['month_sin', 'month_cos'],

            # With differences
            'AR_diff': ['mom_L0', 'mom_L1', 'mom_L2', 'mom_D1', 'mom_D3', 'month_sin', 'month_cos'],
            'AR_MA': ['mom_L0', 'mom_L1', 'mom_MA3', 'mom_MA6', 'month_sin', 'month_cos'],
            'AR_vol': ['mom_L0', 'mom_L1', 'mom_STD3', 'mom_STD6', 'month_sin', 'month_cos'],

            # Components
            'Comp_simple': ['mom_L0', 'prod_L0', 'nonprod_L0', 'serv_L0', 'month_sin', 'month_cos'],
            'Comp_lags': ['mom_L0', 'mom_L1', 'prod_L0', 'prod_L1', 'nonprod_L0', 'nonprod_L1',
                         'serv_L0', 'serv_L1', 'month_sin', 'month_cos'],
            'Comp_diff': ['mom_L0', 'mom_L1', 'prod_L0', 'prod_D3', 'nonprod_L0', 'nonprod_D3',
                         'serv_L0', 'serv_D1', 'month_sin', 'month_cos'],

            # Federal macro
            'Fed_ki': ['mom_L0', 'mom_L1', 'ki_L0', 'ki_L6', 'ki_D6', 'month_sin', 'month_cos'],
            'Fed_ruonia': ['mom_L0', 'mom_L1', 'Ruonia_L0', 'Ruonia_L2', 'Ruonia_D1', 'month_sin', 'month_cos'],
            'Fed_usd': ['mom_L0', 'mom_L1', 'usd_L0', 'usd_L2', 'usd_D1', 'month_sin', 'month_cos'],
            'Fed_brent': ['mom_L0', 'mom_L1', 'brent_L0', 'brent_L5', 'brent_STD3', 'month_sin', 'month_cos'],
            'Fed_all': ['mom_L0', 'mom_L1', 'ki_L6', 'Ruonia_D1', 'usd_L2', 'brent_L5', 'month_sin', 'month_cos'],

            # Regional
            'Reg_ppi': ['mom_L0', 'mom_L1', 'reg_ppi_L0', 'reg_ppi_L3', 'reg_ppi_D1', 'month_sin', 'month_cos'],
            'Reg_wage': ['mom_L0', 'mom_L1', 'reg_wage_L0', 'reg_wage_L3', 'reg_wage_D1', 'month_sin', 'month_cos'],
            'Reg_retail': ['mom_L0', 'mom_L1', 'reg_retail_L0', 'reg_retail_L6', 'month_sin', 'month_cos'],
            'Reg_services': ['mom_L0', 'mom_L1', 'reg_services_L0', 'reg_services_L7', 'month_sin', 'month_cos'],

            # IBVED
            'IBVED_q': ['mom_L0', 'mom_L1', 'ibved_q_L0', 'ibved_q_L1', 'ibved_q_MA3', 'month_sin', 'month_cos'],
            'IBVED_m': ['mom_L0', 'mom_L1', 'ibved_m_L0', 'ibved_m_L1', 'month_sin', 'month_cos'],

            # Combined best
            'Best_v1': ['mom_L0', 'mom_L1', 'prod_L0', 'serv_L0', 'month_sin', 'month_cos'],
            'Best_v2': ['mom_L0', 'mom_L1', 'mom_D3', 'prod_L0', 'serv_L0', 'month_sin', 'month_cos', 'is_jan', 'is_jul'],
            'Best_v3': ['mom_L0', 'mom_L1', 'prod_L0', 'nonprod_L0', 'serv_L0', 'ki_L6', 'month_sin', 'month_cos'],

            # With calendar
            'Calendar': ['mom_L0', 'mom_L1', 'month_sin', 'month_cos', 'is_jan', 'is_jul', 'is_dec', 'is_q1'],
        }

        # Filter to available features
        for name in list(feature_sets.keys()):
            feature_sets[name] = [f for f in feature_sets[name] if f in all_features]
            if len(feature_sets[name]) < 2:
                del feature_sets[name]

        return feature_sets

    def run_backtest(self, df, model, features, horizon):
        """Run single backtest."""
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

        test_dates = pd.date_range(self.test_start, test_end, freq='MS')
        test_dates = [d for d in test_dates if d in df_bt.index]

        predictions = []
        actuals = []

        for test_date in test_dates:
            actual = df_bt.loc[test_date, 'target']
            if pd.isna(actual):
                continue

            # Training data
            train_df = df_bt[df_bt.index < test_date][['target'] + features].dropna()

            if len(train_df) < self.min_train:
                continue

            X_train = train_df[features].values
            y_train = train_df['target'].values

            # Test data
            test_row = df_bt[features].ffill().loc[[test_date]]
            if test_row.isna().any().any():
                continue

            X_test = test_row.values

            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Fit and predict
            try:
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_train_scaled, y_train)
                pred = model_copy.predict(X_test_scaled)[0]
                predictions.append(pred)
                actuals.append(actual)
            except Exception as e:
                continue

        if len(predictions) < 5:
            return None

        # Calculate metrics
        errors = np.array(actuals) - np.array(predictions)
        abs_errors = np.abs(errors)

        return {
            'MAE': np.mean(abs_errors),
            'RMSE': np.sqrt(np.mean(errors ** 2)),
            'KPI_Hits': np.sum(abs_errors <= 0.5),
            'Total': len(predictions),
            'MaxError': np.max(abs_errors),
            'MeanError': np.mean(errors),  # Bias
        }

    def run_all(self, df):
        """Run all model-feature combinations."""
        print("\n" + "=" * 80)
        print("ТЕСТИРОВАНИЕ МОДЕЛЕЙ")
        print("=" * 80)

        models = self.get_models()
        feature_sets = self.get_feature_sets(df)

        print(f"  Моделей: {len(models)}")
        print(f"  Наборов признаков: {len(feature_sets)}")
        print(f"  Горизонтов: 3 (h=1, h=2, h=12)")
        print(f"  Всего комбинаций: {len(models) * len(feature_sets) * 3}")

        results = []

        for horizon in [1, 2, 12]:
            print(f"\n  === ГОРИЗОНТ h={horizon} ===")

            for fs_name, features in feature_sets.items():
                for model_name, model in models.items():
                    metrics = self.run_backtest(df, model, features, horizon)

                    if metrics is None:
                        continue

                    result = {
                        'Horizon': f'h={horizon}',
                        'Model': model_name,
                        'FeatureSet': fs_name,
                        'N_Features': len(features),
                        **metrics
                    }
                    results.append(result)

            # Print top 10 for this horizon
            horizon_results = [r for r in results if r['Horizon'] == f'h={horizon}']
            horizon_df = pd.DataFrame(horizon_results).sort_values('MAE')

            print(f"\n  Топ-10 для h={horizon}:")
            print(f"  {'Модель':<15} {'Признаки':<20} {'MAE':>8} {'KPI':>8}")
            print("  " + "-" * 55)

            for _, row in horizon_df.head(10).iterrows():
                print(f"  {row['Model']:<15} {row['FeatureSet']:<20} {row['MAE']:>8.3f} {int(row['KPI_Hits']):>3}/{int(row['Total'])}")

        self.results = pd.DataFrame(results)

        # Save results
        self.results.to_csv(RESULTS_DIR / 'model_comparison_full.csv', index=False)

        return self.results


class FeatureSelector:
    """Advanced feature selection methods."""

    def __init__(self):
        self.selected_features = {}

    def correlation_filter(self, df, target, threshold=0.1, max_features=20):
        """Select features by correlation."""
        correlations = []
        for col in df.columns:
            if col == 'target':
                continue
            valid = df[[col, 'target']].dropna()
            if len(valid) < 20:
                continue
            try:
                r, p = pearsonr(valid[col], valid['target'])
                if abs(r) >= threshold and p < 0.05:
                    correlations.append((col, abs(r)))
            except:
                pass

        correlations.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in correlations[:max_features]]

    def mutual_info_select(self, X, y, k=20):
        """Select features by mutual information."""
        selector = SelectKBest(score_func=mutual_info_regression, k=min(k, X.shape[1]))
        selector.fit(X, y)
        mask = selector.get_support()
        return mask

    def rfe_select(self, X, y, n_features=15):
        """Recursive Feature Elimination."""
        model = Ridge(alpha=1.0)
        rfe = RFE(model, n_features_to_select=min(n_features, X.shape[1]))
        rfe.fit(X, y)
        return rfe.support_

    def lasso_select(self, X, y, alpha=0.1):
        """Select features using Lasso."""
        lasso = Lasso(alpha=alpha, max_iter=5000)
        lasso.fit(X, y)
        return np.abs(lasso.coef_) > 0.001

    def run_all(self, df, horizon=1):
        """Run all feature selection methods."""
        print("\n" + "=" * 80)
        print(f"ОТБОР ПРИЗНАКОВ (h={horizon})")
        print("=" * 80)

        # Create target
        if horizon == 12:
            target = df['mom'].rolling(12).mean().shift(-12)
        else:
            target = df['mom'].shift(-horizon)

        df_fs = df.copy()
        df_fs['target'] = target

        # Get features
        feature_cols = [c for c in df_fs.columns
                       if c not in ['mom', 'month', 'target']
                       and '_L' in c and int(c.split('_L')[-1].split('_')[0]) <= 6]

        # Prepare data
        data = df_fs[feature_cols + ['target']].dropna()
        X = data[feature_cols].values
        y = data['target'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        results = {}

        # 1. Correlation
        print("\n  1. Корреляционный отбор:")
        corr_features = self.correlation_filter(df_fs, 'target')
        results['correlation'] = corr_features[:15]
        print(f"     Отобрано: {len(results['correlation'])} признаков")
        for f in results['correlation'][:5]:
            print(f"       - {f}")

        # 2. Mutual Information
        print("\n  2. Mutual Information:")
        mi_mask = self.mutual_info_select(X_scaled, y, k=15)
        results['mutual_info'] = [feature_cols[i] for i, m in enumerate(mi_mask) if m]
        print(f"     Отобрано: {len(results['mutual_info'])} признаков")

        # 3. RFE
        print("\n  3. Recursive Feature Elimination:")
        rfe_mask = self.rfe_select(X_scaled, y, n_features=15)
        results['rfe'] = [feature_cols[i] for i, m in enumerate(rfe_mask) if m]
        print(f"     Отобрано: {len(results['rfe'])} признаков")

        # 4. Lasso
        print("\n  4. Lasso Selection:")
        lasso_mask = self.lasso_select(X_scaled, y)
        results['lasso'] = [feature_cols[i] for i, m in enumerate(lasso_mask) if m]
        print(f"     Отобрано: {len(results['lasso'])} признаков")

        # 5. Intersection (features selected by multiple methods)
        print("\n  5. Пересечение (≥2 методов):")
        all_selected = results['correlation'] + results['mutual_info'] + results['rfe'] + results['lasso']
        from collections import Counter
        counts = Counter(all_selected)
        intersection = [f for f, c in counts.items() if c >= 2]
        results['intersection'] = intersection
        print(f"     Отобрано: {len(results['intersection'])} признаков")
        for f in results['intersection'][:10]:
            print(f"       - {f} (выбран {counts[f]} методами)")

        self.selected_features[horizon] = results

        return results


def generate_final_report(model_results, feature_selection):
    """Generate comprehensive final report."""
    print("\n" + "=" * 80)
    print("ФИНАЛЬНЫЙ ОТЧЁТ")
    print("=" * 80)

    report = []
    report.append("# ФИНАЛЬНЫЙ ОТЧЁТ: Масштабное исследование признаков и моделей")
    report.append(f"\nДата: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append("")

    # Summary statistics
    report.append("## 1. СТАТИСТИКА ИССЛЕДОВАНИЯ")
    report.append("")
    report.append(f"- Моделей протестировано: {model_results['Model'].nunique()}")
    report.append(f"- Наборов признаков: {model_results['FeatureSet'].nunique()}")
    report.append(f"- Всего комбинаций: {len(model_results)}")
    report.append("")

    # Best models by horizon
    report.append("## 2. ЛУЧШИЕ МОДЕЛИ ПО ГОРИЗОНТАМ")
    report.append("")

    for horizon in ['h=1', 'h=2', 'h=12']:
        h_data = model_results[model_results['Horizon'] == horizon].sort_values('MAE')

        if len(h_data) == 0:
            continue

        report.append(f"### {horizon}")
        report.append("")
        report.append("| Место | Модель | Признаки | MAE | KPI | N_feat |")
        report.append("|-------|--------|----------|-----|-----|--------|")

        for i, (_, row) in enumerate(h_data.head(10).iterrows()):
            report.append(f"| {i+1} | {row['Model']} | {row['FeatureSet']} | {row['MAE']:.3f} | {int(row['KPI_Hits'])}/{int(row['Total'])} | {int(row['N_Features'])} |")

        report.append("")

    # Best feature sets
    report.append("## 3. ЛУЧШИЕ НАБОРЫ ПРИЗНАКОВ")
    report.append("")

    # Average MAE by feature set across horizons
    fs_avg = model_results.groupby('FeatureSet')['MAE'].mean().sort_values()

    report.append("| Набор признаков | Средний MAE | h=1 | h=2 | h=12 |")
    report.append("|-----------------|-------------|-----|-----|------|")

    for fs in fs_avg.head(15).index:
        fs_data = model_results[model_results['FeatureSet'] == fs]
        avg = fs_data['MAE'].mean()

        h1 = fs_data[fs_data['Horizon'] == 'h=1']['MAE'].min()
        h2 = fs_data[fs_data['Horizon'] == 'h=2']['MAE'].min()
        h12 = fs_data[fs_data['Horizon'] == 'h=12']['MAE'].min()

        h1_str = f"{h1:.3f}" if not pd.isna(h1) else "—"
        h2_str = f"{h2:.3f}" if not pd.isna(h2) else "—"
        h12_str = f"{h12:.3f}" if not pd.isna(h12) else "—"

        report.append(f"| {fs} | {avg:.3f} | {h1_str} | {h2_str} | {h12_str} |")

    report.append("")

    # Best models
    report.append("## 4. ЛУЧШИЕ МОДЕЛИ (усреднённо)")
    report.append("")

    model_avg = model_results.groupby('Model')['MAE'].mean().sort_values()

    report.append("| Модель | Средний MAE | Стабильность (std) |")
    report.append("|--------|-------------|-------------------|")

    for model in model_avg.head(15).index:
        m_data = model_results[model_results['Model'] == model]
        avg = m_data['MAE'].mean()
        std = m_data['MAE'].std()
        report.append(f"| {model} | {avg:.3f} | {std:.3f} |")

    report.append("")

    # Key findings
    report.append("## 5. КЛЮЧЕВЫЕ ВЫВОДЫ")
    report.append("")

    # Find best overall
    best = model_results.loc[model_results['MAE'].idxmin()]
    report.append(f"**Лучшая комбинация:** {best['Model']} + {best['FeatureSet']} на {best['Horizon']}")
    report.append(f"- MAE: {best['MAE']:.3f}")
    report.append(f"- KPI: {int(best['KPI_Hits'])}/{int(best['Total'])}")
    report.append("")

    # Recommendations
    report.append("## 6. РЕКОМЕНДАЦИИ")
    report.append("")

    # Find consistently good combinations
    good_combos = model_results.groupby(['Model', 'FeatureSet']).agg({
        'MAE': 'mean',
        'KPI_Hits': 'sum',
        'Total': 'sum'
    }).sort_values('MAE').head(10)

    report.append("**Рекомендуемые комбинации (стабильные на всех горизонтах):**")
    report.append("")

    for (model, fs), row in good_combos.iterrows():
        kpi_pct = row['KPI_Hits'] / row['Total'] * 100 if row['Total'] > 0 else 0
        report.append(f"- {model} + {fs}: MAE={row['MAE']:.3f}, KPI={kpi_pct:.0f}%")

    report.append("")
    report.append("---")
    report.append(f"\n*Сгенерировано {datetime.now().strftime('%Y-%m-%d %H:%M')}*")

    # Save report
    report_text = "\n".join(report)
    report_path = RESULTS_DIR / 'RESEARCH_FINAL_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n  Отчёт сохранён: {report_path}")
    print("\n" + report_text)

    return report_text


def main():
    print("╔" + "═" * 78 + "╗")
    print("║" + " МАСШТАБНОЕ ИССЛЕДОВАНИЕ ПРИЗНАКОВ И МОДЕЛЕЙ ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")

    # 1. Load data
    loader = DataLoader()
    df = loader.load()

    # 2. Create features
    engineer = FeatureEngineer(max_lag=12)
    df = engineer.transform(df)

    # 3. Feature selection
    selector = FeatureSelector()
    for h in [1, 2, 12]:
        selector.run_all(df, horizon=h)

    # 4. Test all models
    tester = ModelTester()
    results = tester.run_all(df)

    # 5. Generate report
    report = generate_final_report(results, selector.selected_features)

    print("\n" + "=" * 80)
    print("ИССЛЕДОВАНИЕ ЗАВЕРШЕНО")
    print(f"Результаты: {RESULTS_DIR}")
    print("=" * 80)

    return df, results, selector


if __name__ == '__main__':
    df, results, selector = main()
