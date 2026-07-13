#!/usr/bin/env python3
"""
ОПТИМИЗИРОВАННАЯ МИКРОКОМПОНЕНТНАЯ МОДЕЛЬ v1
=============================================
Использует разные модели для разных категорий товаров.

Стратегия:
- stable (std < 2): Huber - робастный к выбросам
- medium (2 <= std < 5): Huber - робастный к выбросам
- volatile (5 <= std < 15): Ridge_500 - высокая регуляризация
- ultra_volatile (std >= 15): Субкомпонентный fallback - непредсказуемы

Результаты тестирования:
- Huber лучше на 85% веса (stable + medium)
- Ridge_500 лучше для volatile (11% веса)
- Ultra-volatile (2.6% веса) лучше прогнозировать на уровне субкомпонента
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler


class MicroOptimizedForecaster:
    """
    Optimized microcomponent forecaster with category-specific models.
    """

    name = "micro_optimized"

    # Volatility thresholds
    STABLE_THRESHOLD = 2.0
    MEDIUM_THRESHOLD = 5.0
    VOLATILE_THRESHOLD = 15.0

    def __init__(self, horizon=1, train_start='2016-01-01', random_state=42,
                 use_subcomp_for_ultra_volatile=True):
        self.horizon = horizon
        self.train_start = train_start
        self.random_state = random_state
        self.use_subcomp_fallback = use_subcomp_for_ultra_volatile

        self._is_fitted = False
        self.micro_models = {}
        self.subcomp_models = {}
        self.weights = {}
        self.categories = {}
        self.micro_to_subcomp = {}
        self.subcomp_weights = {}

    def _load_data(self, data_dir):
        """Load and preprocess data."""
        # Micro data
        micro_df = pd.read_csv(data_dir / 'kbr_micro_full.csv', sep=',', decimal='.')
        micro_df['DateParsed'] = pd.to_datetime(
            micro_df['Day'].str.split(' ').str[0],
            format='%m/%d/%y', errors='coerce'
        )
        micro_df['Period'] = micro_df['DateParsed'].dt.to_period('M').dt.to_timestamp()
        micro_pivot = micro_df.pivot_table(
            index='Period', columns='Item_code', values='MoM', aggfunc='first'
        )
        micro_pivot = micro_pivot.sort_index()
        micro_pivot = micro_pivot - 100  # To changes

        # Справочник
        sprav = pd.read_csv(
            data_dir / 'raw' / 'micro_sprav.csv',
            sep=';', decimal=',', encoding='utf-8-sig'
        )
        self.weights = dict(zip(sprav['Item_code'], sprav['Weight']))
        self.micro_names = dict(zip(sprav['Item_code'], sprav['Товар']))
        self.micro_to_subcomp = dict(zip(
            sprav['Item_code'],
            sprav['Субкомпонент'].fillna('').astype(str)
        ))

        # Subcomponent data
        sub_df = pd.read_csv(
            data_dir / 'raw' / 'sub_mom.csv',
            sep=';', decimal=',', encoding='utf-8-sig'
        )
        sub_df['Date'] = pd.to_datetime(sub_df['Date'], format='%d.%m.%Y')
        sub_df = sub_df.set_index('Date').sort_index()
        sub_df.index = sub_df.index.to_period('M').to_timestamp()
        # Already in change format

        # Subcomp weights
        sub_sprav = pd.read_csv(
            data_dir / 'raw' / 'subcomp_sprav.csv',
            sep=';', decimal=',', encoding='utf-8-sig'
        )
        self.subcomp_weights = dict(zip(
            sub_sprav['Item_code'].astype(str),
            sub_sprav['Weight']
        ))

        return micro_pivot, sub_df

    def _categorize_items(self, micro_data):
        """Categorize items by volatility."""
        recent = micro_data[micro_data.index >= '2022-01-01']

        for item in micro_data.columns:
            if item not in self.weights:
                continue

            std = recent[item].std() if item in recent.columns else 0

            if np.isnan(std):
                self.categories[item] = 'medium'
            elif std < self.STABLE_THRESHOLD:
                self.categories[item] = 'stable'
            elif std < self.MEDIUM_THRESHOLD:
                self.categories[item] = 'medium'
            elif std < self.VOLATILE_THRESHOLD:
                self.categories[item] = 'volatile'
            else:
                self.categories[item] = 'ultra_volatile'

    def _create_features(self, series, extended=False):
        """Create features."""
        df = pd.DataFrame({'y': series})

        for lag in [1, 2, 3, 6, 12]:
            df[f'L{lag}'] = df['y'].shift(lag)

        df['D1'] = df['y'].diff(1)
        df['MA3'] = df['y'].rolling(3).mean()
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

        if extended:
            df['MA6'] = df['y'].rolling(6).mean()
            df['STD3'] = df['y'].rolling(3).std()
            df['YoY'] = df['y'] - df['y'].shift(12)
            df['is_summer'] = df.index.month.isin([6, 7, 8]).astype(int)

        return df

    def _fit_huber(self, series, extended=False):
        """Fit Huber model (robust to outliers)."""
        df = self._create_features(series, extended)
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

        model = HuberRegressor(epsilon=1.35, max_iter=500)
        model.fit(X_scaled, y)

        return {
            'type': 'huber',
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'extended': extended,
            'last_data': series.copy()
        }

    def _fit_ridge(self, series, alpha=500.0, extended=True):
        """Fit Ridge model with high regularization (for volatile items)."""
        df = self._create_features(series, extended)
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

        model = Ridge(alpha=alpha, random_state=self.random_state)
        model.fit(X_scaled, y)

        return {
            'type': 'ridge',
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'extended': extended,
            'last_data': series.copy()
        }

    def fit(self, df, target_col='Все товары и услуги'):
        """Fit optimized microcomponent model."""
        data_dir = Path(__file__).parent.parent.parent / 'data'
        micro_data, subcomp_data = self._load_data(data_dir)
        self.macro_df = df.copy()

        # Categorize items
        self._categorize_items(micro_data)

        # Count by category
        cat_counts = {}
        for cat in ['stable', 'medium', 'volatile', 'ultra_volatile']:
            items = [k for k, v in self.categories.items() if v == cat]
            weight = sum(self.weights.get(k, 0) for k in items) * 100
            cat_counts[cat] = (len(items), weight)

        # Fit models
        for item_code in micro_data.columns:
            if item_code not in self.weights:
                continue

            cat = self.categories.get(item_code, 'medium')
            series = micro_data[item_code].dropna()

            if len(series) < 36:
                continue

            # Skip ultra-volatile if using subcomp fallback
            if cat == 'ultra_volatile' and self.use_subcomp_fallback:
                continue

            # Choose model based on category
            if cat in ['stable', 'medium']:
                result = self._fit_huber(series, extended=False)
            elif cat == 'volatile':
                result = self._fit_ridge(series, alpha=500.0, extended=True)
            else:  # ultra_volatile without fallback
                result = self._fit_ridge(series, alpha=1000.0, extended=True)

            if result:
                self.micro_models[item_code] = result

        # Fit subcomp models for fallback
        if self.use_subcomp_fallback:
            for subcomp in subcomp_data.columns:
                if subcomp not in self.subcomp_weights:
                    continue

                series = subcomp_data[subcomp].dropna()
                if len(series) < 24:
                    continue

                result = self._fit_ridge(series, alpha=100.0, extended=False)
                if result:
                    self.subcomp_models[subcomp] = result

        self._is_fitted = True

        # Print stats
        print(f"MicroOptimizedForecaster fitted:")
        for cat, (count, weight) in cat_counts.items():
            fitted = sum(1 for k, v in self.categories.items()
                        if v == cat and k in self.micro_models)
            print(f"  {cat}: {fitted}/{count} models ({weight:.1f}% weight)")
        if self.use_subcomp_fallback:
            print(f"  Subcomp fallback: {len(self.subcomp_models)} models")

        return self

    def _predict_single(self, model_data, target_date):
        """Predict using a single model."""
        series = model_data['last_data']
        extended = model_data.get('extended', False)
        df = self._create_features(series, extended)

        pred_date = target_date - pd.DateOffset(months=self.horizon)
        if pred_date not in df.index:
            pred_date = df.index[-1]

        feature_cols = model_data['feature_cols']
        X = df.loc[[pred_date], feature_cols].values

        if np.any(np.isnan(X)):
            X = np.nan_to_num(X, nan=0)

        X_scaled = model_data['scaler'].transform(X)
        return model_data['model'].predict(X_scaled)[0]

    def predict(self, df, target_date):
        """Predict aggregated MoM."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        predictions = {}

        # Predict from micro models
        for item_code, model_data in self.micro_models.items():
            try:
                pred = self._predict_single(model_data, target_date)
                predictions[item_code] = pred
            except:
                continue

        # Add subcomp fallback for ultra-volatile items
        if self.use_subcomp_fallback:
            ultra_volatile_items = [k for k, v in self.categories.items()
                                   if v == 'ultra_volatile']

            for item_code in ultra_volatile_items:
                if item_code in predictions:
                    continue

                # Use subcomp prediction
                subcomp = self.micro_to_subcomp.get(item_code, '')
                if subcomp and subcomp in self.subcomp_models:
                    try:
                        pred = self._predict_single(
                            self.subcomp_models[subcomp], target_date
                        )
                        predictions[item_code] = pred
                    except:
                        pass

        if not predictions:
            return {'prediction': 100.0}

        # Weighted aggregation
        total_weight = sum(self.weights.get(c, 0) for c in predictions.keys())
        if total_weight == 0:
            return {'prediction': 100.0}

        agg_pred = sum(
            self.weights.get(c, 0) / total_weight * predictions[c]
            for c in predictions.keys()
        )

        # Coverage
        covered_weight = sum(self.weights.get(c, 0) for c in predictions.keys()) * 100

        return {
            'prediction': 100 + agg_pred,
            'coverage': covered_weight,
            'n_items': len(predictions)
        }

    def forecast(self, horizon=None):
        """Generate forecast trajectory."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")

        h = horizon or self.horizon
        forecasts = []
        last_date = self.macro_df.index[-1]

        for i in range(h):
            target_date = last_date + pd.DateOffset(months=i+1)
            pred = self.predict(None, target_date)
            forecasts.append(pred['prediction'] - 100)

        return np.array(forecasts)

    def get_stats(self):
        """Get model statistics."""
        if not self._is_fitted:
            return {}

        stats = {
            'total_models': len(self.micro_models),
            'by_category': {}
        }

        for cat in ['stable', 'medium', 'volatile', 'ultra_volatile']:
            items = [k for k, v in self.categories.items() if v == cat]
            fitted = [k for k in items if k in self.micro_models]
            stats['by_category'][cat] = {
                'total': len(items),
                'fitted': len(fitted),
                'weight': sum(self.weights.get(k, 0) for k in items) * 100
            }

        if self.use_subcomp_fallback:
            stats['subcomp_fallback'] = len(self.subcomp_models)

        return stats
