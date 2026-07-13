"""
ЛММР Hybrid - гибридная модель Claude
=====================================

SA данные из sa_fl.csv + все 5 shock dummies из R-кода ЦБ.
"""

from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler

from .base import BaseForecaster
from .registry import ModelRegistry
from ..sa_data_loader import get_sa_with_total


@ModelRegistry.register("lmmr_hybrid")
class LMMRHybridForecaster(BaseForecaster):
    """ЛММР Hybrid: SA из sa_fl.csv + shock dummies."""

    name = "lmmr_hybrid"
    MIN_TRAIN_SIZE = 36
    OUTLIER_YEARS = [2010]

    SHOCK_DUMMIES = [
        'is_shock_dec2014', 'is_shock_jan2015',
        'is_shock_dec2014_jan2015', 'is_tariff_jul',
        'is_shock_mar2022', 'is_shock_apr2022'
    ]

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.model = None
        self.scaler = None
        self.sa_series = None
        self.seasonal_factors = None
        self._features = []

    def _load_sa_data(self, df: pd.DataFrame) -> pd.Series:
        """Загрузка SA из sa_fl.csv."""
        try:
            sa_df = get_sa_with_total()
            sa_total = sa_df['Все товары и услуги']

            common = df.index.intersection(sa_total.index)
            fact = df.loc[common, 'Все товары и услуги']
            sa = sa_total.loc[common]
            seasonal = fact / sa

            sdf = pd.DataFrame({'s': seasonal, 'm': seasonal.index.month, 'y': seasonal.index.year})
            sdf_clean = sdf[~sdf['y'].isin([2022])]
            self.seasonal_factors = sdf_clean.groupby('m')['s'].mean().to_dict()

            return sa_total
        except Exception as e:
            print(f"SA fallback: {e}")
            self.seasonal_factors = {m: 1.0 for m in range(1, 13)}
            return df['Все товары и услуги']

    def _add_dummies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shock dummies из R-кода."""
        r = df.copy()
        r['is_shock_dec2014'] = ((df.index.year == 2014) & (df.index.month == 12)).astype(int)
        r['is_shock_jan2015'] = ((df.index.year == 2015) & (df.index.month == 1)).astype(int)
        r['is_shock_dec2014_jan2015'] = (r['is_shock_dec2014'] | r['is_shock_jan2015']).astype(int)
        r['is_tariff_jul'] = (df.index.month == 7).astype(int)
        r['is_shock_mar2022'] = ((df.index.year == 2022) & (df.index.month == 3)).astype(int)
        r['is_shock_apr2022'] = ((df.index.year == 2022) & (df.index.month == 4)).astype(int)
        return r

    def _prepare(self, df: pd.DataFrame, sa: pd.Series) -> pd.DataFrame:
        """Признаки."""
        r = df.copy()
        sa_a = sa.reindex(df.index)
        r['y_sa_lag1'] = sa_a.shift(1)
        r['y_sa_lag2'] = sa_a.shift(2)

        if 'usd_nom_i' in df.columns:
            r['usd_lag1'] = df['usd_nom_i'].shift(1)
        if 'brent' in df.columns:
            r['brent_lag1'] = df['brent'].shift(1)

        r = self._add_dummies(r)
        return r

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'LMMRHybridForecaster':
        """Обучение."""
        self._validate_data(df, target_col)
        self.sa_series = self._load_sa_data(df)

        df_p = self._prepare(df, self.sa_series)

        self._features = ['y_sa_lag1', 'y_sa_lag2']
        for f in ['usd_lag1', 'brent_lag1']:
            if f in df_p.columns and df_p[f].notna().sum() > 10:
                self._features.append(f)
        self._features.extend(self.SHOCK_DUMMIES)

        df_p['year'] = df_p.index.year
        train = df_p[~df_p['year'].isin(self.OUTLIER_YEARS)].dropna(subset=self._features)

        X = train[self._features].values
        sa_a = self.sa_series.reindex(train.index)
        y = sa_a.values

        mask = ~np.isnan(y)
        X, y = X[mask], y[mask]

        self.scaler = RobustScaler()
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(self.scaler.fit_transform(X), y)

        self._is_fitted = True
        self._last_train_date = df.index.max()
        return self

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """Прогноз."""
        self._check_fitted()
        df_p = self._prepare(df, self.sa_series)

        if target_date not in df_p.index:
            df_p.loc[target_date] = np.nan
            df_p = self._prepare(df, self.sa_series)

        row = df_p.loc[[target_date], self._features]
        for c in row.columns:
            if row[c].isna().any():
                row[c] = df_p[c].dropna().iloc[-1] if not df_p[c].dropna().empty else 0

        sa_pred = self.model.predict(self.scaler.transform(row.values))[0]
        seasonal = self.seasonal_factors.get(target_date.month, 1.0)
        mom = sa_pred * seasonal

        return {'date': target_date, 'prediction': mom, 'model': self.name,
                'sa_prediction': sa_pred, 'seasonal_factor': seasonal}

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """Прогноз на горизонт."""
        self._check_fitted()
        last_sa = self.sa_series.dropna().iloc[-1]
        return np.array([last_sa * self.seasonal_factors.get((self._last_train_date.month + h) % 12 + 1, 1.0)
                         for h in range(horizon)])

    def backtest(self, df: pd.DataFrame, start_date: str = '2023-01-01',
                 target_col: str = 'Все товары и услуги') -> pd.DataFrame:
        """Бэктест."""
        start = pd.Timestamp(start_date)
        results = []

        for td in df.dropna(subset=[target_col]).index:
            if td < start:
                continue
            train = df[df.index < td]
            if len(train.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
                continue
            try:
                m = LMMRHybridForecaster(alpha=self.alpha)
                m.fit(train, target_col)
                pred = m.predict(df[df.index <= td], td)
                results.append({'date': td, 'actual': df.loc[td, target_col],
                               'prediction': pred['prediction'],
                               'error': df.loc[td, target_col] - pred['prediction']})
            except:
                continue

        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.DataFrame:
        """Важность признаков."""
        self._check_fitted()
        return pd.DataFrame({
            'feature': self._features,
            'coefficient': self.model.coef_,
            'abs_importance': np.abs(self.model.coef_)
        }).sort_values('abs_importance', ascending=False)
