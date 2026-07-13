"""
NGBoost Shock + Macro Features
==============================

NGBoost Shock с добавлением макро-признаков:
- USD (курс доллара)
- Ki (ключевая ставка)
- Ruonia (межбанковская ставка)
- Brent (цена нефти) — опционально

Базовая модель: NGBoost Shock (MAE 0.298)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Dict, Any, Optional
import warnings

from .base import BaseForecaster
from .registry import ModelRegistry

try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    NGBOOST_AVAILABLE = True
except ImportError:
    NGBOOST_AVAILABLE = False


@ModelRegistry.register("ngboost_shock_macro")
class NGBoostShockMacroForecaster(BaseForecaster):
    """NGBoost с shock dummies + макро-признаки (USD, Ki, Ruonia)."""

    name = "ngboost_shock_macro"
    MIN_TRAIN_SIZE = 36

    OUTLIER_YEARS = [2010]

    N_ESTIMATORS = 200
    LEARNING_RATE = 0.05
    MINIBATCH_FRAC = 0.8

    ETS_WEIGHTS = {
        1: 0.9, 2: 0.0, 3: 0.5, 4: 0.3,
        5: 0.9, 6: 0.5, 7: 0.0, 8: 0.5,
        9: 0.9, 10: 0.9, 11: 0.0, 12: 0.0
    }

    # Базовые признаки (из NGBoost Shock)
    BASE_FEATURES = [
        'y_lag1', 'y_lag2', 'y_lag12',
        'y_lag3', 'y_lag6',
        'y_ma3', 'y_ma6',
        'd_y_lag1', 'd_y_lag3',
        'y_vol3', 'y_vol6',
        'month_sin', 'month_cos',
        'quarter_sin', 'quarter_cos',
        'is_jan', 'is_dec',
        'is_tariff_month', 'is_q1', 'is_summer',
        'food_lag1', 'nonfood_lag1', 'services_lag1',
        'seasonal_norm', 'deviation_lag1',
    ]

    SHOCK_DUMMIES = [
        'is_shock_dec2014',
        'is_shock_jan2015',
        'is_shock_mar2022',
        'is_shock_apr2022',
        'is_shock_2022',
    ]

    # НОВЫЕ макро-признаки
    MACRO_FEATURES = [
        # USD (курс доллара)
        'usd_lag1', 'usd_lag3',
        'usd_pct_lag1', 'usd_pct_lag3',
        # Ki (ключевая ставка)
        'ki_lag1', 'ki_lag3',
        'ki_diff_lag1', 'ki_diff_lag6',
        # Ruonia
        'ruonia_lag1',
        'ruonia_diff_lag1', 'ruonia_diff_lag3',
        # Спред Ki - Ruonia
        'spread_lag3', 'spread_lag6',
    ]

    def __init__(self, n_estimators: int = None, learning_rate: float = None,
                 use_brent: bool = False, **kwargs):
        super().__init__(**kwargs)

        if not NGBOOST_AVAILABLE:
            raise ImportError("NGBoost не установлен")

        self.n_estimators = n_estimators or self.N_ESTIMATORS
        self.learning_rate = learning_rate or self.LEARNING_RATE
        self.use_brent = use_brent

        self.model = None
        self.scaler = None
        self.seasonal_norm = None
        self._features = None
        self._macro_data = None

    def _load_macro_data(self) -> pd.DataFrame:
        """Загрузить макро-данные из inflation_data.csv."""
        try:
            df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')

            # Фиксим числовые колонки
            for col in ['usd_nom_i', 'Ki', 'Ruonia']:
                if col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].astype(str).str.replace(',', '.')
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Парсим дату
            df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
            if df['Date'].isna().any():
                df['Date'] = pd.to_datetime(df['Date'])

            df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
            df = df.set_index('Date').sort_index()

            # Выбираем нужные колонки
            macro = pd.DataFrame(index=df.index)
            macro['usd'] = df['usd_nom_i'] if 'usd_nom_i' in df.columns else np.nan
            macro['ki'] = df['Ki'] if 'Ki' in df.columns else np.nan
            macro['ruonia'] = df['Ruonia'] if 'Ruonia' in df.columns else np.nan

            return macro

        except Exception as e:
            print(f"Ошибка загрузки макро-данных: {e}")
            return pd.DataFrame()

    def _add_shock_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавить shock dummy переменные."""
        df = df.copy()
        df['is_shock_dec2014'] = ((df.index.year == 2014) & (df.index.month == 12)).astype(int)
        df['is_shock_jan2015'] = ((df.index.year == 2015) & (df.index.month == 1)).astype(int)
        df['is_shock_mar2022'] = ((df.index.year == 2022) & (df.index.month == 3)).astype(int)
        df['is_shock_apr2022'] = ((df.index.year == 2022) & (df.index.month == 4)).astype(int)
        df['is_shock_2022'] = (df.index.year == 2022).astype(int)
        return df

    def _add_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавить макро-признаки."""
        df = df.copy()

        # Загружаем макро-данные если ещё не загружены
        if self._macro_data is None or self._macro_data.empty:
            self._macro_data = self._load_macro_data()

        if self._macro_data.empty:
            # Если макро-данных нет, заполняем нулями
            for feat in self.MACRO_FEATURES:
                df[feat] = 0
            return df

        # Мержим макро-данные
        df = df.join(self._macro_data, how='left')

        # === USD признаки ===
        if 'usd' in df.columns:
            df['usd_lag1'] = df['usd'].shift(1)
            df['usd_lag3'] = df['usd'].shift(3)
            df['usd_pct'] = df['usd'].pct_change() * 100
            df['usd_pct_lag1'] = df['usd_pct'].shift(1)
            df['usd_pct_lag3'] = df['usd_pct'].shift(3)
        else:
            df['usd_lag1'] = 0
            df['usd_lag3'] = 0
            df['usd_pct_lag1'] = 0
            df['usd_pct_lag3'] = 0

        # === Ki признаки ===
        if 'ki' in df.columns:
            df['ki_lag1'] = df['ki'].shift(1)
            df['ki_lag3'] = df['ki'].shift(3)
            df['ki_diff'] = df['ki'].diff()
            df['ki_diff_lag1'] = df['ki_diff'].shift(1)
            df['ki_diff_lag6'] = df['ki_diff'].shift(6)
        else:
            df['ki_lag1'] = 0
            df['ki_lag3'] = 0
            df['ki_diff_lag1'] = 0
            df['ki_diff_lag6'] = 0

        # === Ruonia признаки ===
        if 'ruonia' in df.columns:
            df['ruonia_lag1'] = df['ruonia'].shift(1)
            df['ruonia_diff'] = df['ruonia'].diff()
            df['ruonia_diff_lag1'] = df['ruonia_diff'].shift(1)
            df['ruonia_diff_lag3'] = df['ruonia_diff'].shift(3)
        else:
            df['ruonia_lag1'] = 0
            df['ruonia_diff_lag1'] = 0
            df['ruonia_diff_lag3'] = 0

        # === Спред Ki - Ruonia ===
        if 'ki' in df.columns and 'ruonia' in df.columns:
            df['spread'] = df['ki'] - df['ruonia']
            df['spread_lag3'] = df['spread'].shift(3)
            df['spread_lag6'] = df['spread'].shift(6)
        else:
            df['spread_lag3'] = 0
            df['spread_lag6'] = 0

        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка всех признаков."""
        df = df.copy()

        df['month'] = df.index.month
        df['year'] = df.index.year
        df['quarter'] = df.index.quarter

        y = df['Все товары и услуги']

        # Лаги
        df['y_lag1'] = y.shift(1)
        df['y_lag2'] = y.shift(2)
        df['y_lag12'] = y.shift(12)
        df['y_lag3'] = y.shift(3)
        df['y_lag6'] = y.shift(6)

        # MA
        df['y_ma3'] = y.rolling(3).mean().shift(1)
        df['y_ma6'] = y.rolling(6).mean().shift(1)

        # Momentum
        df['d_y_lag1'] = (y.shift(1) - y.shift(2))
        df['d_y_lag3'] = (y.shift(1) - y.shift(4))

        # Volatility
        df['y_vol3'] = y.rolling(3).std().shift(1)
        df['y_vol6'] = y.rolling(6).std().shift(1)

        # Сезонность
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
        df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)

        # Календарь
        df['is_jan'] = (df['month'] == 1).astype(int)
        df['is_dec'] = (df['month'] == 12).astype(int)
        df['is_tariff_month'] = (df['month'] == 7).astype(int)
        df['is_q1'] = (df['quarter'] == 1).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)

        # Компоненты
        if 'Продовольственные товары' in df.columns:
            df['food_lag1'] = df['Продовольственные товары'].shift(1)
        else:
            df['food_lag1'] = df['y_lag1']

        if 'Непродовольственные товары' in df.columns:
            df['nonfood_lag1'] = df['Непродовольственные товары'].shift(1)
        else:
            df['nonfood_lag1'] = df['y_lag1']

        if 'Услуги' in df.columns:
            df['services_lag1'] = df['Услуги'].shift(1)
        else:
            df['services_lag1'] = df['y_lag1']

        # Shock dummies
        df = self._add_shock_dummies(df)

        # Макро-признаки (НОВОЕ!)
        df = self._add_macro_features(df)

        return df

    def _compute_seasonal_norm(self, df: pd.DataFrame) -> pd.Series:
        """Сезонная норма (исключаем 2022)."""
        clean_df = df[df['year'] != 2022]
        return clean_df.groupby('month')['Все товары и услуги'].mean()

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'NGBoostShockMacroForecaster':
        """Обучение модели."""
        self._validate_data(df, target_col)

        # Загружаем макро-данные
        self._macro_data = self._load_macro_data()

        df_prep = self._prepare_features(df)
        self.seasonal_norm = self._compute_seasonal_norm(df_prep)

        df_prep['seasonal_norm'] = df_prep['month'].map(self.seasonal_norm)
        df_prep['deviation_lag1'] = df_prep['y_lag1'] - df_prep['month'].shift(1).map(self.seasonal_norm)

        # Все признаки: база + shock + macro
        self._features = self.BASE_FEATURES.copy() + self.SHOCK_DUMMIES + self.MACRO_FEATURES

        # Исключаем только 2010
        train_df = df_prep[~df_prep['year'].isin(self.OUTLIER_YEARS)]
        train_clean = train_df.dropna(subset=self._features + [target_col])

        if len(train_clean) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Недостаточно данных: {len(train_clean)} < {self.MIN_TRAIN_SIZE}")

        X = train_clean[self._features].values
        y = train_clean[target_col].values

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = NGBRegressor(
                Dist=Normal,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                minibatch_frac=self.MINIBATCH_FRAC,
                verbose=False
            )
            self.model.fit(X_scaled, y)

        self._is_fitted = True
        self._last_train_date = df.index.max()

        return self

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Прогноз с доверительными интервалами."""
        self._check_fitted()

        df_prep = self._prepare_features(df)
        df_prep['seasonal_norm'] = df_prep['month'].map(self.seasonal_norm)
        df_prep['deviation_lag1'] = df_prep['y_lag1'] - df_prep['month'].shift(1).map(self.seasonal_norm)

        test_row = df_prep.loc[[target_date]]
        X_test = self.scaler.transform(test_row[self._features].values)

        dist = self.model.pred_dist(X_test)
        pred_mean = dist.mean()[0]
        pred_std = dist.std()[0]

        ci_lower = dist.ppf(0.05)[0]
        ci_upper = dist.ppf(0.95)[0]

        target_month = target_date.month
        pred_ets = self.seasonal_norm.get(target_month, 100.0)

        ets_weight = self.ETS_WEIGHTS.get(target_month, 0.3)
        pred_combined = (1 - ets_weight) * pred_mean + ets_weight * pred_ets

        ci_lower_adj = (1 - ets_weight) * ci_lower + ets_weight * pred_ets
        ci_upper_adj = (1 - ets_weight) * ci_upper + ets_weight * pred_ets

        return {
            'date': target_date,
            'prediction': pred_combined,
            'pred_ngboost': pred_mean,
            'pred_ets': pred_ets,
            'std': pred_std,
            'ci_lower': ci_lower_adj,
            'ci_upper': ci_upper_adj,
            'model': self.name
        }

    def forecast(self, horizon: int = 12) -> np.ndarray:
        self._check_fitted()
        last_month = self._last_train_date.month if self._last_train_date else 1
        return np.array([self.seasonal_norm.get(((last_month + i) % 12) + 1, 100.0) for i in range(horizon)])

    def backtest(self, df: pd.DataFrame, start_date: str = '2023-01-01',
                 target_col: str = 'Все товары и услуги') -> pd.DataFrame:
        start = pd.Timestamp(start_date)
        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]

        results = []
        for target_date in test_dates:
            train_df = df[df.index < target_date].copy()
            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
                continue
            try:
                model = NGBoostShockMacroForecaster(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate
                )
                model.fit(train_df, target_col)
                test_df = df[df.index <= target_date].copy()
                pred = model.predict(test_df, target_date)
                actual = df.loc[target_date, target_col]
                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred['prediction'],
                    'error': actual - pred['prediction']
                })
            except Exception:
                continue

        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        self._check_fitted()
        importance = pd.DataFrame({
            'feature': self._features,
            'importance': self.model.feature_importances_
        })
        importance['category'] = importance['feature'].apply(
            lambda x: 'shock' if x in self.SHOCK_DUMMIES
            else 'macro' if x in self.MACRO_FEATURES
            else 'base'
        )
        return importance.sort_values('importance', ascending=False)
