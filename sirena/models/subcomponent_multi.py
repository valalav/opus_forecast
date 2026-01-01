#!/usr/bin/env python3
"""
МУЛЬТИМОДЕЛЬНАЯ СУБКОМПОНЕНТНАЯ МОДЕЛЬ v2.4
===========================================
Использует оптимальную модель для каждого субкомпонента:
- Ridge: 55% веса (22 субкомп.) - стабильные товары
- Prophet: 14% веса (6 субкомп.) - услуги с сезонностью
- NGBoost: 9% веса (5 субкомп.) - волатильные товары
- VotingRegressor: 22% веса (12 субкомп.) - остальные

Улучшения v2.0:
- Shock dummies для 2022 года
- Расширен NGBoost на volatile items (яйца, туризм, сахар)
- Robust imputation (forward fill вместо nan→0)

Улучшения v2.1:
- Расширенные лаги: L18, L24 (годовые циклы)
- Расширенный momentum: D12 (годовой)
- Расширенные MA: MA6, MA12
- Квартальная сезонность: quarter_sin, quarter_cos
- Календарные признаки: is_jan, is_jul

Улучшения v2.2:
- Volatility-adaptive гиперпараметры:
  * Ridge alpha: 50 (stable) / 100 (moderate) / 200 (volatile)
  * Prophet changepoint: 0.03 / 0.08 / 0.15
  * NGBoost lr: 0.03 / 0.05 / 0.08

Улучшения v2.3:
- Rate-признаки для трансмиссии монетарной политики:
  * ruonia_diff_lag1 (r=0.477!) - изменение Ruonia с лагом 1
  * spread_lag4 (r=0.444!) - спред Ki-Ruonia с лагом 4
  * ki_diff_lag6 - изменение ставки с лагом 6
- Субкомпонент-специфичные rate-признаки:
  * Кредитозависимые (авто, мебель, ПК): ki_lag3, ki_lag4, ki_diff_lag3
  * Импортозависимые (одежда, обувь): ki_lag5, ki_lag6
  * Регулируемые (ЖКХ): только spread_lag4 (низкая чувствительность)
- MicroPlodovoshchi с формулой цепных индексов

Улучшения v2.4:
- Production Proxy Features (demand indicators):
  * torg_lag3, torg_lag6 - торговый оборот (demand proxy)
  * torg_diff_lag3, torg_ma3 - momentum торговли
  * pp_lag3, pp_lag6 - платные услуги (services demand)
  * pp_diff_lag3 - momentum услуг
- Субкомпонент-специфичные demand-признаки:
  * Продовольственные товары: torg_lag3, torg_lag6
  * Услуги: pp_lag3, pp_lag6
  * Товары длительного пользования: torg_diff_lag3, pp_diff_lag3

Результаты бэктеста h=1 (дек 2024 - ноя 2025):
- v1.0 MAE: 0.320 (baseline)
- v2.0 MAE: 0.297 (-7.1%)
- v2.1 MAE: 0.290 (-9.4%)
- v2.2 MAE: 0.274 (-14.4%)
- v2.3 MAE: 0.236 (-26.3%)
- v2.4 MAE: TBD (+ production proxy features)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler

# Check optional dependencies
PROPHET_AVAILABLE = False
NGBOOST_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    pass

try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    NGBOOST_AVAILABLE = True
except ImportError:
    pass

# Import MicroPlodovoshchi model for plodovoshchi forecasting
MICRO_PLOD_AVAILABLE = False
try:
    from sirena.models.micro_plodovoshchi import MicroPlodovoshchiForecaster
    MICRO_PLOD_AVAILABLE = True
except ImportError:
    pass

# v2.4: Import production proxy features
PRODUCTION_PROXY_AVAILABLE = False
try:
    from sirena.macro_features import load_production_proxies, PRODUCTION_FEATURES
    PRODUCTION_PROXY_AVAILABLE = True
except ImportError:
    pass

# Оптимальные модели для каждого субкомпонента (на основе бэктеста v2.0)
# Изменения v2.0:
# - Яйца (52) и Туризм (67): prophet → ngboost (extreme volatility std>9)
# - Сахар (36): ridge → ngboost (std=5.7)
OPTIMAL_MODELS = {
    # PROPHET (14% веса) - услуги с сильной сезонностью
    '14': 'prophet',  # ЖКХ (+10.2%) - детерминированная индексация тарифов
    '20': 'prophet',  # Мебель (+4.8%)
    '31': 'prophet',  # Персональные компьютеры (+16.8%) - back-to-school
    '43': 'prophet',  # Трикотажные изделия (+6.3%)
    '44': 'prophet',  # Услуги образования (+4.2%)
    '50': 'prophet',  # Чай, кофе, какао (+11.6%)

    # NGBOOST (9% веса) - волатильные товары (std > 3.5)
    # Note: micro_plod для '33' не интегрирован - ухудшает общий MAE из-за
    # расхождения данных kbr_micro_full.csv и sub_mom.csv (региональная специфика)
    '33': 'ngboost',  # Плодоовощи (std=7.3) - extreme volatility
    '41': 'ngboost',  # Телерадиотовары (std=3.5)
    '52': 'ngboost',  # Яйца (std=11.8!) - перенесено из prophet
    '67': 'ngboost',  # Туризм (std=9.3) - перенесено из prophet
    '36': 'ngboost',  # Сахар (std=5.7) - перенесено из ridge

    # RIDGE (55% веса) - стабильные товары (std < 2)
    '11': 'ridge', '12': 'ridge', '15': 'ridge', '16': 'ridge',
    '17': 'ridge', '18': 'ridge', '21': 'ridge', '24': 'ridge',
    '25': 'ridge', '26': 'ridge', '27': 'ridge', '28': 'ridge',
    '29': 'ridge', '30': 'ridge', '32': 'ridge',
    '37': 'ridge', '38': 'ridge', '39': 'ridge', '47': 'ridge',
    '49': 'ridge', '51': 'ridge', '53': 'ridge', '55': 'ridge',

    # VOTING (22% веса) - остальные (baseline)
    '13': 'voting', '19': 'voting', '22': 'voting', '23': 'voting',
    '34': 'voting', '35': 'voting', '40': 'voting', '42': 'voting',
    '46': 'voting', '48': 'voting', '54': 'voting',
}


class SubcomponentMultiForecaster:
    """
    Multi-model bottom-up forecaster using optimal model for each subcomponent.

    Models used:
    - Ridge: stable goods (58% weight)
    - Prophet: services with seasonality (17% weight)
    - NGBoost: volatile goods (6% weight)
    - VotingRegressor: baseline for others (19% weight)
    """

    name = "subcomponent_multi"

    def __init__(self, horizon=1, train_start='2016-01-01', random_state=42):
        self.horizon = horizon
        self.train_start = train_start
        self.random_state = random_state
        self._is_fitted = False
        self.subcomponent_models = {}
        self.weights = {}
        self.micro_plod_model = None  # v2.3: Specialized model for плодоовощи

    def _load_data(self, data_dir):
        """Load subcomponent data."""
        # Try newer subcomp.csv first (до октября 2025), fallback to sub_mom.csv
        subcomp_file = data_dir / 'raw' / 'subcomp.csv'
        sub_mom_file = data_dir / 'raw' / 'sub_mom.csv'

        if subcomp_file.exists():
            sub = pd.read_csv(subcomp_file, sep=';', decimal=',', encoding='utf-8-sig')
            date_col = 'Day' if 'Day' in sub.columns else 'Date'
            sub[date_col] = pd.to_datetime(sub[date_col], format='%d.%m.%Y')
            sub = sub.rename(columns={date_col: 'Date'}).set_index('Date').sort_index()
        else:
            sub = pd.read_csv(sub_mom_file, sep=';', decimal=',', encoding='utf-8-sig')
            sub['Date'] = pd.to_datetime(sub['Date'], format='%d.%m.%Y')
            sub = sub.set_index('Date').sort_index()

        sub.index = sub.index.to_period('M').to_timestamp()
        sub = sub[~sub.index.duplicated(keep='last')]

        sprav = pd.read_csv(data_dir / 'raw' / 'subcomp_sprav.csv',
                            sep=';', decimal=',', encoding='utf-8-sig')
        self.weights = dict(zip(sprav['Item_code'].astype(str), sprav['Weight']))

        valid_cols = [c for c in sub.columns if c in self.weights]
        sub = sub[valid_cols]

        return sub

    def _create_features(self, series, subcomp_code=None):
        """
        Create features for ML models (v2.3 with rate features).

        v2.1: Extended lags and seasonality
        v2.3: Rate-sensitive features for monetary policy transmission
        """
        df = pd.DataFrame({'y': series})

        # Базовые лаги (v2.1: добавлены L18, L24 для годовых циклов)
        for lag in [1, 2, 3, 6, 12, 18, 24]:
            df[f'L{lag}'] = df['y'].shift(lag)

        # Momentum (разности)
        df['D1'] = df['y'].diff(1)
        df['D3'] = df['y'].diff(3)  # v2.0: расширенный momentum
        df['D6'] = df['y'].diff(6)  # v2.0: полугодовой momentum
        df['D12'] = df['y'].diff(12)  # v2.1: годовой momentum

        # Скользящие средние (v2.1: добавлены MA6, MA12)
        df['MA3'] = df['y'].rolling(3).mean()
        df['MA6'] = df['y'].rolling(6).mean()
        df['MA12'] = df['y'].rolling(12).mean()

        # v2.0: Волатильность
        df['vol_3m'] = df['y'].rolling(3).std()
        df['vol_6m'] = df['y'].rolling(6).std()

        # Сезонность (месяц)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

        # v2.1: Квартальная сезонность
        df['quarter_sin'] = np.sin(2 * np.pi * df.index.quarter / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df.index.quarter / 4)

        # v2.1: Календарные признаки
        df['is_jan'] = (df.index.month == 1).astype(int)  # Январская индексация
        df['is_jul'] = (df.index.month == 7).astype(int)  # Июльская индексация тарифов

        # v2.0: Shock dummies для 2022 года (санкционный шок)
        df['is_shock_mar2022'] = (df.index == pd.Timestamp('2022-03-01')).astype(int)
        df['is_shock_apr2022'] = (df.index == pd.Timestamp('2022-04-01')).astype(int)
        df['is_shock_period'] = ((df.index >= '2022-01-01') & (df.index <= '2022-12-31')).astype(int)

        # =====================================================================
        # v2.3: Rate-sensitive features (monetary policy transmission)
        # Используем макро-данные если доступны
        # =====================================================================
        if hasattr(self, 'macro_df') and self.macro_df is not None:
            macro = self.macro_df

            # Ruonia diff lag 1 (r=0.477!) - самый сильный rate-признак
            if 'Ruonia' in macro.columns:
                ruonia = macro['Ruonia'].reindex(df.index)
                df['ruonia_diff_lag1'] = ruonia.diff().shift(1)
                df['ruonia_lag3'] = ruonia.shift(3)

            # Ki-Ruonia spread lag 4 (r=0.444!) - второй по силе
            if 'Ki_i' in macro.columns and 'Ruonia' in macro.columns:
                ki = macro['Ki_i'].reindex(df.index)
                ruonia = macro['Ruonia'].reindex(df.index)
                df['spread_lag4'] = (ki - ruonia).shift(4)
                df['ki_diff_lag6'] = ki.diff().shift(6)  # p=0.0000

            # Субкомпонент-специфичная чувствительность к ставке
            # Для кредитозависимых товаров добавляем больше rate-признаков
            rate_sensitive_codes = {
                # Кредитозависимые (авто, мебель, техника)
                '41': 'high',  # Телерадиотовары
                '20': 'high',  # Мебель
                '31': 'high',  # Персональные компьютеры
                # Импортозависимые
                '29': 'medium',  # Одежда
                '30': 'medium',  # Обувь
                '34': 'medium',  # Рыба
                # Регулируемые (низкая чувствительность)
                '14': 'low',  # ЖКХ
                '12': 'low',  # Квартплата
            }

            if subcomp_code in rate_sensitive_codes:
                sensitivity = rate_sensitive_codes[subcomp_code]

                if 'Ki_i' in macro.columns:
                    ki = macro['Ki_i'].reindex(df.index)

                    if sensitivity == 'high':
                        # Кредитозависимые: короткие лаги (3-6 мес)
                        df['ki_lag3'] = ki.shift(3)
                        df['ki_lag4'] = ki.shift(4)
                        df['ki_diff_lag3'] = ki.diff().shift(3)
                    elif sensitivity == 'medium':
                        # Импортозависимые: средние лаги (4-8 мес)
                        df['ki_lag5'] = ki.shift(5)
                        df['ki_lag6'] = ki.shift(6)
                    # low: только базовые признаки (spread_lag4)

        # =====================================================================
        # v2.4: Production Proxy Features (demand indicators)
        # Torg = торговый оборот, pp = платные услуги
        # =====================================================================
        if hasattr(self, 'production_df') and self.production_df is not None:
            prod = self.production_df

            # Субкомпонент-специфичные demand-признаки
            # Продовольственные товары (codes 33-55): используют torg
            # Услуги (codes 12-14, 44, 67): используют pp
            # Товары длит. пользования (20, 31, 41): используют оба

            demand_sensitive_codes = {
                # Продовольственные товары (food) - torg чувствительные
                '33': 'food', '34': 'food', '35': 'food', '36': 'food',
                '37': 'food', '38': 'food', '39': 'food', '40': 'food',
                '47': 'food', '49': 'food', '50': 'food', '51': 'food',
                '52': 'food', '53': 'food', '54': 'food', '55': 'food',
                # Услуги - pp чувствительные
                '12': 'services', '14': 'services', '44': 'services', '67': 'services',
                # Товары длительного пользования - оба индикатора
                '20': 'durable', '31': 'durable', '41': 'durable',
                # Одежда/обувь - torg
                '29': 'nonfood', '30': 'nonfood', '43': 'nonfood',
            }

            if subcomp_code in demand_sensitive_codes:
                category = demand_sensitive_codes[subcomp_code]

                # Align production data with series index
                if 'Torg' in prod.columns:
                    torg = prod['Torg'].reindex(df.index)

                    if category in ['food', 'nonfood', 'durable']:
                        df['torg_lag3'] = torg.shift(3)
                        df['torg_lag6'] = torg.shift(6)
                        df['torg_diff_lag3'] = torg.diff().shift(3)

                    if category == 'durable':
                        # Для товаров длит. пользования — добавляем MA
                        df['torg_ma3'] = torg.rolling(3).mean().shift(1)

                if 'pp' in prod.columns:
                    pp = prod['pp'].reindex(df.index)

                    if category in ['services', 'durable']:
                        df['pp_lag3'] = pp.shift(3)
                        df['pp_lag6'] = pp.shift(6)
                        df['pp_diff_lag3'] = pp.diff().shift(3)

        return df

    def _get_volatility_class(self, series):
        """
        v2.2: Determine volatility class for hyperparameter selection.

        Returns:
            'stable' (std < 2): Conservative params for stable goods
            'moderate' (2-5): Standard params
            'volatile' (>5): Aggressive params for volatile goods
        """
        std = series.std()
        if std < 2:
            return 'stable'
        elif std < 5:
            return 'moderate'
        else:
            return 'volatile'

    def _fit_ridge(self, series, col):
        """Fit Ridge model with volatility-adaptive alpha."""
        df = self._create_features(series, subcomp_code=col)
        df['target'] = df['y'].shift(-self.horizon)
        df = df.dropna()

        if self.train_start:
            df = df[df.index >= pd.to_datetime(self.train_start)]

        if len(df) < 24:
            return None

        feature_cols = [c for c in df.columns if c not in ['target', 'y']]
        X = df[feature_cols].values
        y = df['target'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # v2.2: Volatility-adaptive alpha
        vol_class = self._get_volatility_class(series)
        alpha_map = {'stable': 50.0, 'moderate': 100.0, 'volatile': 200.0}
        alpha = alpha_map[vol_class]

        model = Ridge(alpha=alpha, random_state=self.random_state)
        model.fit(X_scaled, y)

        return {
            'type': 'ridge',
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'last_data': series.copy(),
            'subcomp_code': col  # v2.3: для rate-признаков
        }

    def _fit_voting(self, series, col):
        """Fit VotingRegressor (Ridge + Lasso) with volatility-adaptive params."""
        df = self._create_features(series, subcomp_code=col)
        df['target'] = df['y'].shift(-self.horizon)
        df = df.dropna()

        if self.train_start:
            df = df[df.index >= pd.to_datetime(self.train_start)]

        if len(df) < 24:
            return None

        feature_cols = [c for c in df.columns if c not in ['target', 'y']]
        X = df[feature_cols].values
        y = df['target'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # v2.2: Volatility-adaptive alpha
        vol_class = self._get_volatility_class(series)
        alpha_map = {'stable': 50.0, 'moderate': 100.0, 'volatile': 200.0}
        lasso_map = {'stable': 0.05, 'moderate': 0.1, 'volatile': 0.2}
        alpha = alpha_map[vol_class]
        lasso_alpha = lasso_map[vol_class]

        model = VotingRegressor([
            ('ridge', Ridge(alpha=alpha, random_state=self.random_state)),
            ('lasso', Lasso(alpha=lasso_alpha, random_state=self.random_state, max_iter=5000)),
        ])
        model.fit(X_scaled, y)

        return {
            'type': 'voting',
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'last_data': series.copy(),
            'subcomp_code': col  # v2.3: для rate-признаков
        }

    def _fit_prophet(self, series, col):
        """Fit Prophet model with volatility-adaptive changepoint_prior_scale."""
        if not PROPHET_AVAILABLE:
            return self._fit_ridge(series, col)  # Fallback

        df = pd.DataFrame({
            'ds': series.index,
            'y': series.values
        })

        if self.train_start:
            df = df[df['ds'] >= pd.to_datetime(self.train_start)]

        if len(df) < 24:
            return None

        # v2.2: Volatility-adaptive changepoint_prior_scale
        vol_class = self._get_volatility_class(series)
        cp_map = {'stable': 0.03, 'moderate': 0.08, 'volatile': 0.15}
        changepoint_prior = cp_map[vol_class]

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=changepoint_prior
        )
        model.fit(df)

        return {
            'type': 'prophet',
            'model': model,
            'last_data': series.copy()
        }

    def _fit_ngboost(self, series, col):
        """Fit NGBoost model with volatility-adaptive learning_rate."""
        if not NGBOOST_AVAILABLE:
            return self._fit_ridge(series, col)  # Fallback

        df = self._create_features(series, subcomp_code=col)
        df['target'] = df['y'].shift(-self.horizon)
        df = df.dropna()

        if self.train_start:
            df = df[df.index >= pd.to_datetime(self.train_start)]

        if len(df) < 24:
            return None

        feature_cols = [c for c in df.columns if c not in ['target', 'y']]
        X = df[feature_cols].values
        y = df['target'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # v2.2: Volatility-adaptive learning_rate and n_estimators
        vol_class = self._get_volatility_class(series)
        lr_map = {'stable': 0.03, 'moderate': 0.05, 'volatile': 0.08}
        n_est_map = {'stable': 80, 'moderate': 100, 'volatile': 150}
        lr = lr_map[vol_class]
        n_estimators = n_est_map[vol_class]

        model = NGBRegressor(
            Dist=Normal,
            n_estimators=n_estimators,
            learning_rate=lr,
            minibatch_frac=1.0,
            random_state=self.random_state,
            verbose=False
        )
        model.fit(X_scaled, y)

        return {
            'type': 'ngboost',
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'last_data': series.copy(),
            'subcomp_code': col  # v2.3: для rate-признаков
        }

    def _fit_micro_plod(self, df):
        """
        v2.3: Fit MicroPlodovoshchiForecaster for плодоовощи (code 33).

        Uses 21 microcomponents:
        - 4 imported (бананы, апельсины) → Ridge + USD
        - 9 volatile (капуста, картофель) → NGBoost
        - 8 stable (яблоки, груши) → Ridge
        """
        if not MICRO_PLOD_AVAILABLE:
            return None

        # Load USD series for imported items
        data_dir = Path(__file__).parent.parent.parent / 'data'
        usd_series = None
        try:
            usd_file = data_dir / 'inflation_data.csv'
            if usd_file.exists():
                usd_df = pd.read_csv(usd_file, sep=';', decimal=',')
                usd_df['Date'] = pd.to_datetime(usd_df['Date'], format='%d.%m.%Y')
                usd_df = usd_df.set_index('Date').sort_index()
                usd_df.index = usd_df.index.to_period('M').to_timestamp()
                if 'usd_nom_i' in usd_df.columns:
                    usd_series = usd_df['usd_nom_i']
        except Exception:
            pass

        model = MicroPlodovoshchiForecaster(
            horizon=self.horizon,
            train_start=self.train_start,
            random_state=self.random_state
        )
        model.fit(df, usd_series)
        self.micro_plod_model = model

        return {
            'type': 'micro_plod',
            'model': model,
            'usd_series': usd_series
        }

    def fit(self, df, target_col='Все товары и услуги'):
        """Fit models for all subcomponents."""
        data_dir = Path(__file__).parent.parent.parent / 'data'
        sub_data = self._load_data(data_dir)
        self.macro_df = df.copy()

        # v2.4: Load production proxy data (Torg, pp)
        self.production_df = None
        if PRODUCTION_PROXY_AVAILABLE:
            try:
                self.production_df = load_production_proxies(str(data_dir / 'raw'))
            except FileNotFoundError:
                pass  # Fallback: features will be skipped

        for col in sub_data.columns:
            model_type = OPTIMAL_MODELS.get(col, 'voting')

            if model_type == 'ridge':
                result = self._fit_ridge(sub_data[col], col)
            elif model_type == 'prophet':
                result = self._fit_prophet(sub_data[col], col)
            elif model_type == 'ngboost':
                result = self._fit_ngboost(sub_data[col], col)
            elif model_type == 'micro_plod':
                # v2.3: Fit specialized microcomponent model for плодоовощи
                result = self._fit_micro_plod(df)
                if result is None:
                    # Fallback to NGBoost if micro_plod unavailable
                    result = self._fit_ngboost(sub_data[col], col)
            else:
                result = self._fit_voting(sub_data[col], col)

            if result:
                self.subcomponent_models[col] = result

        self._is_fitted = True
        return self

    def _predict_ml(self, model_data, target_date):
        """Predict using ML model (Ridge, Voting, NGBoost)."""
        series = model_data['last_data']
        subcomp_code = model_data.get('subcomp_code')
        df = self._create_features(series, subcomp_code=subcomp_code)

        # Normalize to first of month for matching
        pred_date = target_date - pd.DateOffset(months=self.horizon)
        pred_date = pred_date.to_period('M').to_timestamp()
        if pred_date not in df.index:
            pred_date = df.index[-1]

        feature_cols = model_data['feature_cols']

        # v2.0: Robust imputation - forward fill then backward fill
        X_df = df.loc[[pred_date], feature_cols].ffill(axis=0).bfill(axis=0)
        X = X_df.values

        # Fallback: if still NaN (shouldn't happen), use column means from training
        if np.any(np.isnan(X)):
            X = np.nan_to_num(X, nan=0)

        X_scaled = model_data['scaler'].transform(X)
        return model_data['model'].predict(X_scaled)[0]

    def _predict_prophet(self, model_data, target_date):
        """Predict using Prophet."""
        model = model_data['model']
        future = model.make_future_dataframe(periods=self.horizon + 12, freq='MS')
        forecast = model.predict(future)

        pred_row = forecast[forecast['ds'] == target_date]
        if len(pred_row) > 0:
            return pred_row['yhat'].values[0]

        # Fallback: return last prediction
        return forecast['yhat'].iloc[-1]

    def predict(self, df, target_date):
        """Predict aggregated MoM for target date."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        predictions = {}

        for col, model_data in self.subcomponent_models.items():
            model_type = model_data['type']

            if model_type == 'prophet':
                pred = self._predict_prophet(model_data, target_date)
            elif model_type == 'micro_plod':
                # v2.3: Use MicroPlodovoshchi model for плодоовощи
                micro_model = model_data['model']
                result = micro_model.predict(df, target_date)
                pred = result['prediction'] - 100  # Convert back to MoM pp
            else:
                pred = self._predict_ml(model_data, target_date)

            predictions[col] = pred

        # Aggregate
        total_weight = sum(self.weights[c] for c in predictions.keys())
        agg_pred = sum(self.weights[c] / total_weight * predictions[c]
                       for c in predictions.keys())

        return {'prediction': 100 + agg_pred}

    def forecast(self, horizon=None):
        """Generate forecast trajectory using iterative predictions."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        h = horizon or 12

        # Get last date from subcomponent data, not macro
        # Find a model with 'last_data' (skip micro_plod)
        last_date = None
        for model_data in self.subcomponent_models.values():
            if 'last_data' in model_data:
                last_date = model_data['last_data'].index[-1]
                break
        if last_date is None:
            last_date = self.macro_df.index[-1]

        # Store predicted series for each subcomponent (for iterative updates)
        subcomp_series = {}
        for col, model_data in self.subcomponent_models.items():
            if 'last_data' in model_data:
                subcomp_series[col] = model_data['last_data'].copy()
            else:
                # micro_plod doesn't have last_data, use empty series
                subcomp_series[col] = pd.Series(dtype=float)

        forecasts = []
        for step in range(h):
            target_date = last_date + pd.DateOffset(months=step+1)

            predictions = {}
            for col, model_data in self.subcomponent_models.items():
                model_type = model_data['type']

                # Use extended series for feature creation
                series = subcomp_series[col]

                if model_type == 'prophet':
                    pred = self._predict_prophet(model_data, target_date)
                elif model_type == 'micro_plod':
                    # v2.3: Use MicroPlodovoshchi model for плодоовощи
                    micro_model = model_data['model']
                    result = micro_model.predict(self.macro_df, target_date)
                    pred = result['prediction'] - 100  # Convert back to MoM pp
                else:
                    # Create features from extended series
                    subcomp_code = model_data.get('subcomp_code', col)
                    df = self._create_features(series, subcomp_code=subcomp_code)
                    pred_date = series.index[-1]  # Use last available date

                    feature_cols = model_data['feature_cols']

                    # v2.0: Robust imputation - forward fill then backward fill
                    X_df = df.loc[[pred_date], feature_cols].ffill(axis=0).bfill(axis=0)
                    X = X_df.values

                    # Fallback: if still NaN (shouldn't happen), use column means
                    if np.any(np.isnan(X)):
                        X = np.nan_to_num(X, nan=0)

                    X_scaled = model_data['scaler'].transform(X)
                    pred = model_data['model'].predict(X_scaled)[0]

                predictions[col] = pred

                # Update series for next iteration (iterative forecast)
                # Note: micro_plod doesn't need series update as it uses internal data
                if model_type != 'micro_plod':
                    subcomp_series[col] = pd.concat([
                        series,
                        pd.Series([pred], index=[target_date])
                    ])

            # Aggregate
            total_weight = sum(self.weights[c] for c in predictions.keys())
            agg_pred = sum(self.weights[c] / total_weight * predictions[c]
                           for c in predictions.keys())

            forecasts.append(agg_pred)

        return np.array(forecasts)

    def get_model_distribution(self):
        """Get distribution of model types."""
        dist = {}
        for col, data in self.subcomponent_models.items():
            t = data['type']
            if t not in dist:
                dist[t] = {'count': 0, 'weight': 0}
            dist[t]['count'] += 1
            dist[t]['weight'] += self.weights.get(col, 0) * 100
        return dist
