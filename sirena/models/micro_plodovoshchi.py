#!/usr/bin/env python3
"""
МИКРОКОМПОНЕНТНАЯ МОДЕЛЬ ПЛОДООВОЩЕЙ v2.0
=========================================
Прогнозирует субкомпонент "Плодоовощи" (код 33) через 21 микрокомпонент:
- 6 импортных товаров → Ridge + USD экзогенная
- 9 сезонных отечественных → NGBoost (высокая волатильность)
- 6 стабильных → Ridge

КЛЮЧЕВОЕ ИЗМЕНЕНИЕ v2.0:
Агрегация через ЦЕПНЫЕ ИНДЕКСЫ (как у Росстата), а не прямое среднее MoM!

Формула:
    Cum_i(t) = Cum_i(t-1) × (1 + MoM_i(t)/100)
    Agg_Cum(t) = Σ(Cum_i(t) × W_i) / Σ(W_i)
    Sub_MoM(t) = (Agg_Cum(t) / Agg_Cum(t-1) - 1) × 100

Использование:
    from sirena.models import MicroPlodovoshchiForecaster

    model = MicroPlodovoshchiForecaster()
    model.fit(df, usd_series)
    prediction = model.predict(df, target_date, usd_forecast)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Check optional dependencies
NGBOOST_AVAILABLE = False
try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    NGBOOST_AVAILABLE = True
except ImportError:
    pass


# Классификация микрокомпонентов плодоовощей
IMPORTED_ITEMS = [121, 115, 343, 167]  # Бананы, Апельсины, Лимоны, Виноград
VOLATILE_ITEMS = [249, 252, 435, 723, 349, 382, 506, 574, 1087]  # Капуста, Картофель, Огурцы и др.
STABLE_ITEMS = [753, 204, 447, 432, 618, 971, 972, 973]  # Яблоки, Груши, Орехи и др.

# Все коды плодоовощей
PLODOVOSHCHI_CODES = IMPORTED_ITEMS + VOLATILE_ITEMS + STABLE_ITEMS


class MicroPlodovoshchiForecaster:
    """
    Forecaster for Fruits & Vegetables (субкомп. 33) using 21 microcomponents.

    Key features:
    - Uses USD as exogenous variable for imported items
    - Applies NGBoost for volatile items (капуста, картофель, огурцы)
    - Ridge for stable items
    """

    name = "micro_plodovoshchi"

    def __init__(self, horizon=1, train_start='2016-01-01', random_state=42):
        self.horizon = horizon
        self.train_start = train_start
        self.random_state = random_state
        self._is_fitted = False
        self.models = {}
        self.weights = {}
        self.scalers = {}

    def _load_data(self, data_dir):
        """Load microcomponent data for плодоовощи."""
        # Load microcomponent time series (format: mm/dd/yy)
        micro = pd.read_csv(data_dir / 'kbr_micro_full.csv')
        micro['Day'] = pd.to_datetime(micro['Day'], format='%m/%d/%y %H:%M:%S')
        micro = micro.set_index('Day')

        # Filter to плодоовощи only
        micro = micro[micro['Item_code'].isin(PLODOVOSHCHI_CODES)]

        # Pivot to wide format
        pivot = micro.pivot_table(index=micro.index, columns='Item_code', values='MoM')
        pivot = pivot - 100  # Convert to percentage points

        # Normalize to start of month
        pivot.index = pivot.index.to_period('M').to_timestamp()
        pivot = pivot[~pivot.index.duplicated(keep='last')]

        # Load weights
        sprav = pd.read_csv(data_dir / 'raw' / 'micro_sprav.csv',
                           sep=';', decimal=',', encoding='utf-8-sig')
        self.weights = dict(zip(sprav['Item_code'], sprav['Weight']))

        # Calculate cumulative indices for chain index aggregation
        # Cum_i(t) = Cum_i(t-1) × (1 + MoM_i(t)/100), starting from 100 each January
        self.cumulative_indices = pd.DataFrame(index=pivot.index, columns=pivot.columns)

        for col in pivot.columns:
            cum = 100.0
            for idx in pivot.index:
                mom = pivot.loc[idx, col]
                if pd.isna(mom):
                    self.cumulative_indices.loc[idx, col] = np.nan
                else:
                    # Reset to 100 each January (start of year)
                    if idx.month == 1:
                        cum = 100.0 * (1 + mom / 100)
                    else:
                        cum = cum * (1 + mom / 100)
                    self.cumulative_indices.loc[idx, col] = cum

        return pivot

    def _create_features(self, series, usd_series=None, is_imported=False):
        """Create features for ML models."""
        df = pd.DataFrame({'y': series})

        # Базовые лаги
        for lag in [1, 2, 3, 6, 12]:
            df[f'L{lag}'] = df['y'].shift(lag)

        # Momentum
        df['D1'] = df['y'].diff(1)
        df['D12'] = df['y'].diff(12)  # Годовой momentum

        # Скользящие средние
        df['MA3'] = df['y'].rolling(3).mean()
        df['MA12'] = df['y'].rolling(12).mean()

        # Волатильность
        df['vol_3m'] = df['y'].rolling(3).std()

        # Сезонность
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

        # Сезонные месяцы для плодоовощей
        df['is_harvest'] = df.index.month.isin([8, 9, 10]).astype(int)  # Урожай
        df['is_winter'] = df.index.month.isin([1, 2, 3]).astype(int)    # Зима (дефицит)

        # USD для импортных товаров
        if is_imported and usd_series is not None:
            # Align USD series with our index
            usd_aligned = usd_series.reindex(df.index)
            df['usd_pct'] = usd_aligned.pct_change() * 100
            df['usd_pct_lag1'] = df['usd_pct'].shift(1)
            df['usd_pct_lag3'] = df['usd_pct'].shift(3)

        # Shock dummies
        df['is_shock_2022'] = ((df.index >= '2022-03-01') & (df.index <= '2022-06-30')).astype(int)

        return df

    def _fit_ridge(self, series, item_code, usd_series=None):
        """Fit Ridge model for stable/imported items."""
        is_imported = item_code in IMPORTED_ITEMS
        df = self._create_features(series, usd_series, is_imported)
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

        # Меньше регуляризации для импортных (USD важен)
        alpha = 50.0 if is_imported else 100.0
        model = Ridge(alpha=alpha, random_state=self.random_state)
        model.fit(X_scaled, y)

        return {
            'type': 'ridge',
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'last_data': series.copy(),
            'is_imported': is_imported
        }

    def _fit_ngboost(self, series, item_code, usd_series=None):
        """Fit NGBoost model for volatile items."""
        if not NGBOOST_AVAILABLE:
            return self._fit_ridge(series, item_code, usd_series)

        df = self._create_features(series, usd_series, is_imported=False)
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

        # Aggressive params for volatile items
        model = NGBRegressor(
            Dist=Normal,
            n_estimators=150,
            learning_rate=0.08,
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
            'is_imported': False
        }

    def fit(self, df, usd_series=None, target_col='Все товары и услуги'):
        """
        Fit models for all microcomponents.

        Args:
            df: Main inflation DataFrame (for macro alignment)
            usd_series: USD/RUB exchange rate series (optional, for imported items)
            target_col: Unused, for API compatibility
        """
        data_dir = Path(__file__).parent.parent.parent / 'data'
        micro_data = self._load_data(data_dir)

        # Store USD series
        self.usd_series = usd_series

        for code in micro_data.columns:
            series = micro_data[code].dropna()

            if len(series) < 36:
                continue

            if code in VOLATILE_ITEMS:
                result = self._fit_ngboost(series, code, usd_series)
            else:
                result = self._fit_ridge(series, code, usd_series)

            if result:
                self.models[code] = result

        self._is_fitted = True
        print(f"Fitted {len(self.models)} microcomponent models for плодоовощи")
        return self

    def _predict_item(self, model_data, target_date, usd_forecast=None):
        """Predict single microcomponent."""
        series = model_data['last_data']
        is_imported = model_data.get('is_imported', False)

        # Create features
        usd = self.usd_series if is_imported else None
        df = self._create_features(series, usd, is_imported)

        # Get prediction date
        pred_date = target_date - pd.DateOffset(months=self.horizon)
        pred_date = pred_date.to_period('M').to_timestamp()
        if pred_date not in df.index:
            pred_date = df.index[-1]

        feature_cols = model_data['feature_cols']
        X = df.loc[[pred_date], feature_cols].ffill(axis=0).bfill(axis=0).values

        if np.any(np.isnan(X)):
            X = np.nan_to_num(X, nan=0)

        X_scaled = model_data['scaler'].transform(X)
        return model_data['model'].predict(X_scaled)[0]

    def predict(self, df, target_date, usd_forecast=None):
        """
        Predict aggregated плодоовощи for target date using chain index formula.

        Args:
            df: Main inflation DataFrame
            target_date: Target date for prediction
            usd_forecast: Forecasted USD/RUB for imported items

        Returns:
            dict with 'prediction' (100 + MoM in pp)
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        # Get MoM predictions for each microcomponent
        predictions = {}
        for code, model_data in self.models.items():
            pred = self._predict_item(model_data, target_date, usd_forecast)
            predictions[code] = pred

        # Chain Index Aggregation (Rosstat formula)
        # 1. Get previous month's cumulative indices
        target_date_norm = pd.Timestamp(target_date).to_period('M').to_timestamp()
        prev_date = target_date_norm - pd.DateOffset(months=1)

        # For January, previous cumulative = 100 (base)
        if target_date_norm.month == 1:
            prev_cum_agg = 100.0
        else:
            # Calculate previous aggregated cumulative index
            prev_cum_weighted = 0.0
            total_weight = 0.0
            for code in predictions.keys():
                if code in self.cumulative_indices.columns:
                    prev_cum = self.cumulative_indices.loc[prev_date, code] if prev_date in self.cumulative_indices.index else 100.0
                    if pd.notna(prev_cum):
                        w = self.weights.get(code, 0.01)
                        prev_cum_weighted += prev_cum * w
                        total_weight += w
            prev_cum_agg = prev_cum_weighted / total_weight if total_weight > 0 else 100.0

        # 2. Calculate new cumulative indices for each item
        new_cum_weighted = 0.0
        total_weight = 0.0
        for code, mom_pred in predictions.items():
            # Get previous cumulative for this item
            if target_date_norm.month == 1:
                prev_cum_item = 100.0
            else:
                if code in self.cumulative_indices.columns and prev_date in self.cumulative_indices.index:
                    prev_cum_item = self.cumulative_indices.loc[prev_date, code]
                    if pd.isna(prev_cum_item):
                        prev_cum_item = 100.0
                else:
                    prev_cum_item = 100.0

            # New cumulative = prev × (1 + MoM/100)
            new_cum_item = prev_cum_item * (1 + mom_pred / 100)

            w = self.weights.get(code, 0.01)
            new_cum_weighted += new_cum_item * w
            total_weight += w

        new_cum_agg = new_cum_weighted / total_weight if total_weight > 0 else 100.0

        # 3. Sub_MoM = (Agg_Cum(t) / Agg_Cum(t-1) - 1) × 100
        if prev_cum_agg > 0:
            agg_mom = (new_cum_agg / prev_cum_agg - 1) * 100
        else:
            # Fallback to direct average
            agg_mom = sum(
                self.weights.get(c, 0.01) / total_weight * predictions[c]
                for c in predictions.keys()
            )

        return {
            'prediction': 100 + agg_mom,
            'component_predictions': predictions,
            'chain_index': {
                'prev_cum_agg': prev_cum_agg,
                'new_cum_agg': new_cum_agg,
                'agg_mom': agg_mom
            }
        }

    def get_model_distribution(self):
        """Get distribution of model types."""
        dist = {'ngboost': 0, 'ridge_imported': 0, 'ridge_stable': 0}
        for code, data in self.models.items():
            if data['type'] == 'ngboost':
                dist['ngboost'] += 1
            elif data.get('is_imported'):
                dist['ridge_imported'] += 1
            else:
                dist['ridge_stable'] += 1
        return dist
