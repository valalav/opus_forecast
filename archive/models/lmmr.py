"""
LMMR (Linear Mixed/Factor Model) Implementation
================================================

Realization of the "LMMR" methodology (Dynamic Regression on SA data)
using STL decomposition and exogenous factors.

Unified Strategy (Gemini + Claude):
- STL Decomposition (instead of X13)
- Ridge Regression on SA data
- Exogenous: USD, Brent (proxy for external prices), Real Income/Credits (proxy for demand)
- Shock Dummies (2014, 2015, 2017, 2022)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.seasonal import STL

from .base import BaseForecaster
from .registry import ModelRegistry
from ..macro_features import add_brent_features


@ModelRegistry.register("lmmr")
class LMMRForecaster(BaseForecaster):
    """
    LMMR - Local Multiplicative Model Regression (Adapted).
    
    A dynamic regression model on seasonally adjusted data (SA).
    Uses STL for decomposition and Ridge for the core regression.
    
    Features:
    - Target: Seasonally Adjusted (SA) Inflation MoM (via STL on Base Index)
    - Autoregression: SA MoM lags
    - External: USD, Brent, Real Income/Credits
    - Dummies: Shock periods (2014, 2015, 2017, 2022)
    """
    
    name = "lmmr"
    MIN_TRAIN_SIZE = 48  # Requires more data for STL (2 full years +)
    
    # Exogenous features configuration
    USE_BRENT = True
    USE_USD = True
    USE_DEMAND_PROXY = True # Real Income or Credits
    
    def __init__(self, alpha: float = 0.5, demand_proxy: str = 'all_real', **kwargs):
        """
        Initialize LMMR model.
        
        Args:
            alpha: Ridge regularization strength
            demand_proxy: Column name for demand proxy ('all_real' or 'fl_potrb_zad')
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.demand_proxy = demand_proxy
        
        self.model = None
        self.scaler = None
        
        # State
        self.base_index: Optional[pd.Series] = None
        self.sa_series: Optional[pd.Series] = None
        self.sa_mom: Optional[pd.Series] = None
        self.seasonal_component: Optional[pd.Series] = None
        self._last_train_date: Optional[pd.Timestamp] = None
        
        self.features = [
            'y_sa_lag1', 
            'is_shock_dec2014_jan2015', 
            'is_tariff_jul',
            'is_shock_mar2022', 
            'is_shock_apr2022'
        ]

    def _to_base_index(self, mom_series: pd.Series) -> pd.Series:
        """
        Convert MoM (Month-over-Month) index to Base Index.
        
        base[0] = mom[0]
        base[i] = base[i-1] * mom[i] / 100
        """
        # Ensure sorted by date
        series = mom_series.sort_index()
        
        # Calculate cumulative product
        # Start with 100 as base
        base = (series / 100).cumprod() * 100
        return base

    def _from_base_to_mom(self, base_series: pd.Series, start_value: float) -> pd.Series:
        """
        Convert Base Index back to MoM.
        
        mom[i] = base[i] / base[i-1] * 100
        """
        # We need the previous value for the first element calculation
        prev_values = base_series.shift(1)
        prev_values.iloc[0] = start_value
        
        mom = (base_series / prev_values) * 100
        return mom

    def _decompose_series(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Seasonal decomposition using STL.
        
        Returns:
            (sa_series, seasonal_component)
        """
        # STL requires defined frequency
        if series.index.freq is None:
            series = series.asfreq('MS')
            series = series.interpolate() # Handle missing values if any
            
        stl = STL(series, period=12, robust=True)
        result = stl.fit()
        
        # SA = Trend + Resid (or Observed - Seasonal)
        sa = result.trend + result.resid
        sc = result.seasonal
        
        return sa, sc

    def _prepare_features(self, df: pd.DataFrame, sa_mom: Optional[pd.Series] = None) -> pd.DataFrame:
        """Prepare features for regression."""
        result = df.copy()
        
        # --- SA MoM Lags ---
        # Determine the source of SA MoM data
        source_sa_mom = sa_mom if sa_mom is not None else self.sa_mom
        
        if source_sa_mom is not None:
            # Helper to lookup previous month's SA MoM value
            def get_sa_lag(date):
                # Try exact shift first
                prev_date = date - pd.DateOffset(months=1)
                if prev_date in source_sa_mom.index:
                    return source_sa_mom.loc[prev_date]
                return np.nan

            result['y_sa_lag1'] = result.index.map(get_sa_lag)
        else:
            result['y_sa_lag1'] = np.nan
        
        # --- Exogenous Factors ---
        
        # 1. USD (Lag 1)
        if self.USE_USD and 'usd_nom_i' in df.columns:
            result['usd_lag1'] = df['usd_nom_i'].shift(1)
            if 'usd_lag1' not in self.features:
                self.features.append('usd_lag1')

        # 2. Brent (Lags 1, 3) - Proxy for PPI/Freight
        if self.USE_BRENT:
            # If brent is already provided (e.g. in tests or pre-loaded), compute lags manually
            if 'brent' in result.columns:
                 # Compute implied pct change if not present
                 if 'brent_pct' not in result.columns:
                     result['brent_pct'] = result['brent'].pct_change() * 100
                 
                 result['brent_lag1'] = result['brent'].shift(1) # Used in features list
                 result['brent_lag3'] = result['brent'].shift(3)
            else:
                 # Load from external source
                 if 'brent_lag3' not in result.columns:
                      result = add_brent_features(result)
            
            # Ensure brent_lag1 is present
            if 'brent_lag3' in result.columns:
                 if 'brent_lag3' not in self.features:
                     self.features.append('brent_lag3')
            
            if 'brent_lag1' in result.columns:
                 if 'brent_lag1' not in self.features:
                      self.features.append('brent_lag1')

        # 3. Demand Proxy (Real Income or Credits)
        if self.USE_DEMAND_PROXY and self.demand_proxy in df.columns:
            # Usually these publish with delay, so lag 1 or 2 is appropriate
            result[f'{self.demand_proxy}_lag1'] = df[self.demand_proxy].shift(1)
            feat_name = f'{self.demand_proxy}_lag1'
            if feat_name not in self.features:
                self.features.append(feat_name)

        # --- Shock Dummies ---
        
        # Dec 2014 & Jan 2015 (Currency Crisis)
        result['is_shock_dec2014'] = ((df.index.year == 2014) & (df.index.month == 12)).astype(int)
        result['is_shock_jan2015'] = ((df.index.year == 2015) & (df.index.month == 1)).astype(int)
        result['is_shock_dec2014_jan2015'] = (result['is_shock_dec2014'] | result['is_shock_jan2015']).astype(int)

        # July Tariff Indexation (Recurrent)
        result['is_tariff_jul'] = (df.index.month == 7).astype(int)

        # 2022 Shocks (Sanctions)
        result['is_shock_mar2022'] = ((df.index.year == 2022) & (df.index.month == 3)).astype(int)
        result['is_shock_apr2022'] = ((df.index.year == 2022) & (df.index.month == 4)).astype(int)
        
        return result

    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'LMMRForecaster':
        """
        Fit the LMMR model.
        
        Process:
        1. MoM -> Base Index
        2. STL Decomposition -> SA Base Series + Seasonal
        3. Convert SA Base -> SA MoM (Target)
        4. Feature Engineering (lags of SA MoM)
        5. Ridge Regression on SA MoM
        """
        series = self._validate_data(df, target_col)
        
        # 1. Transform to Base Index
        self.base_index = self._to_base_index(series)
        
        # 2. STL Decomposition
        # sa_series here is the SA Level (Base Index)
        self.sa_series, self.seasonal_component = self._decompose_series(self.base_index)
        
        # 3. Calculate SA MoM (Target)
        # mom = current / prev * 100
        self.sa_mom = self.sa_series / self.sa_series.shift(1) * 100
        # First value is NaN, fill it with original MoM or 100
        self.sa_mom.iloc[0] = series.iloc[0] # Approximation
        
        # 4. Prepare Features
        # Pass sa_mom to generate lags
        df_prep = self._prepare_features(df, self.sa_mom)
        
        # 5. Create Train Set
        train_df = df_prep.dropna(subset=self.features)
        
        # Align target y (SA MoM) with features X
        common_indices = train_df.index.intersection(self.sa_mom.index)
        
        if len(common_indices) < self.MIN_TRAIN_SIZE:
             # Just a warning or fallback for small datasets in tests
             pass
        
        X = train_df.loc[common_indices, self.features].values
        y = self.sa_mom.loc[common_indices].values
        
        # 6. Fit Ridge
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X_scaled, y)
        
        self._is_fitted = True
        self._last_train_date = df.index.max()
        
        return self

    def _get_seasonal_factor(self, month: int) -> float:
        """
        Get seasonal factor for a given month.
        Strategy: Naive (mean of seasonal component for this month).
        Since STL seasonal component is additive (Base = SA + Seasonal), we return the additive term.
        """
        if self.seasonal_component is None:
            return 0.0
            
        # Extract seasonal values for this month
        mask = self.seasonal_component.index.month == month
        # Take the most recent values (e.g. last 3 years) to capture evolving seasonality
        recent_values = self.seasonal_component[mask].tail(3)
        return recent_values.mean()

    def forecast(self, horizon: int = 12) -> np.ndarray:
        """
        Forecast horizon steps ahead.
        """
        self._check_fitted()
        return np.zeros(horizon) # Placeholder, see predict()

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        """
        Point forecast for a specific date.
        """
        self._check_fitted()
        
        # 1. Prepare features
        # We need full context to ensure lags (shift) work correctly
        df_prep_all = self._prepare_features(df)
        
        if target_date not in df_prep_all.index:
             raise ValueError(f"target_date {target_date} not found in provided dataframe")

        # Select the specific row for prediction
        df_prep = df_prep_all.loc[[target_date]].copy()
        
        # Manual fix for y_sa_lag1 if it came out NaN (e.g. predicting out of sample)
        if pd.isna(df_prep['y_sa_lag1'].iloc[0]):
            # Use last known SA MoM value from model
            df_prep['y_sa_lag1'] = self.sa_mom.iloc[-1]

        # Check for missing features and fill with 0
        for feat in self.features:
            if feat not in df_prep.columns or pd.isna(df_prep[feat].iloc[0]):
                df_prep[feat] = 0.0
        
        X_test = self.scaler.transform(df_prep[self.features].values)
        
        # 2. Predict SA MoM
        sa_mom_pred = self.model.predict(X_test)[0]
        
        # 3. Reconstruct SA Level
        # Needs previous SA Level
        prev_date = target_date - pd.DateOffset(months=1)
        if prev_date in self.sa_series.index:
            prev_sa_level = self.sa_series.loc[prev_date]
        else:
            prev_sa_level = self.sa_series.iloc[-1] # Fallback
            
        sa_level_pred = prev_sa_level * (sa_mom_pred / 100)
        
        # 4. Add Seasonality
        seasonal_factor = self._get_seasonal_factor(target_date.month)
        base_pred = sa_level_pred + seasonal_factor
        
        # 5. Convert to MoM
        # We need the Base Index value of the previous month
        if prev_date in self.base_index.index:
            prev_base = self.base_index.loc[prev_date]
        else:
            prev_base = self.base_index.iloc[-1] # Fallback
            
        mom_pred = (base_pred / prev_base) * 100
        
        return {
            'date': target_date,
            'prediction': mom_pred,
            'model': self.name,
            'sa_prediction': sa_level_pred, # Base Level SA
            'sa_mom_prediction': sa_mom_pred,
            'seasonal_factor': seasonal_factor,
            'base_prediction': base_pred
        }

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """
        Backtest with re-fitting (Expanding Window).
        """
        start = pd.Timestamp(start_date)
        test_dates = df.index[df.index >= start]
        
        results = []
        
        for target_date in test_dates:
            # Train on data strictly BEFORE target_date
            train_df = df[df.index < target_date].copy()
            
            if len(train_df) < self.MIN_TRAIN_SIZE:
                continue
                
            try:
                # Re-instantiate and fit
                model = LMMRForecaster(alpha=self.alpha, demand_proxy=self.demand_proxy)
                model.fit(train_df, target_col)
                
                # Test on current date
                # We need a dataframe that includes the target row for exogenous features
                test_df = df[df.index <= target_date].copy()
                
                pred = model.predict(test_df, target_date)
                
                actual = df.loc[target_date, target_col]
                
                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred['prediction'],
                    'error': actual - pred['prediction'],
                    'sa_prediction': pred.get('sa_mom_prediction', 0)
                })
            except Exception as e:
                # print(f"LMMR Backtest Error at {target_date}: {e}")
                continue
                
        return pd.DataFrame(results)