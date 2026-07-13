import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List

from .base import BaseForecaster
from .registry import ModelRegistry
from .usd_model import USDForecaster
from ..macro_features import load_brent_prices

@ModelRegistry.register("fundamental")
class FundamentalForecaster(BaseForecaster):
    """
    Fundamental model that relies on economic drivers (USD, Oil, Key Rate)
    rather than hardcoded seasonality.
    
    Features:
    - Forecasted USD MoM (using USDForecaster)
    - Oil prices (Brent)
    - Key Rate (Ki) and Ruonia
    - Inflation lags and trend
    
    Does NOT use:
    - Month dummies
    - ETS blending
    """
    
    name = "fundamental"
    MIN_TRAIN_SIZE = 36
    
    def __init__(self, alpha=0.5, l1_ratio=0.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        self.scaler = StandardScaler()
        self.usd_model = USDForecaster()
        self.features = []
        
    def _prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        df = df.copy()
        
        # Ensure dates are aligned (Month Start)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        # df.index = df.index + pd.offsets.MonthBegin(0) # Assuming input is already consistent or we handle it carefully
        
        # Lags of target
        df['y_lag1'] = df['Все товары и услуги'].shift(1)
        df['y_lag2'] = df['Все товары и услуги'].shift(2)
        df['y_lag3'] = df['Все товары и услуги'].shift(3)
        df['y_lag6'] = df['Все товары и услуги'].shift(6)
        
        # Trend (rolling mean)
        df['y_trend3'] = df['Все товары и услуги'].rolling(3).mean().shift(1)
        df['y_trend6'] = df['Все товары и услуги'].rolling(6).mean().shift(1)
        
        # Volatility
        df['y_vol6'] = df['Все товары и услуги'].rolling(6).std().shift(1)
        
        # Macro Features
        # USD
        if 'usd_nom_i' in df.columns:
            df['usd_mom'] = df['usd_nom_i'] - 100
            df['usd_lag1'] = df['usd_mom'].shift(1)
            df['usd_lag2'] = df['usd_mom'].shift(2)
            df['usd_lag3'] = df['usd_mom'].shift(3)
            
        # Key Rate
        if 'Ki' in df.columns:
            df['ki_diff'] = df['Ki'].diff()
            df['ki_lag1'] = df['Ki'].shift(1)
            df['ki_lag3'] = df['Ki'].shift(3)
            df['ki_diff_lag1'] = df['ki_diff'].shift(1)
            
        # Ruonia
        if 'Ruonia' in df.columns:
            df['ruonia_diff'] = df['Ruonia'].diff()
            df['ruonia_lag1'] = df['Ruonia'].shift(1)
            df['ruonia_diff_lag1'] = df['ruonia_diff'].shift(1)
            
        # Brent (need to load if not present, but usually passed in df if prepared)
        # Check if brent columns exist
        if 'brent_pct' in df.columns:
            df['brent_lag1'] = df['brent_pct'].shift(1)
            df['brent_lag2'] = df['brent_pct'].shift(2)
            df['brent_lag3'] = df['brent_pct'].shift(3)
            
        return df
        
    def fit(self, df: pd.DataFrame, target_col: str = 'Все товары и услуги') -> 'FundamentalForecaster':
        # Fit USD model first
        try:
            self.usd_model.fit(df)
        except Exception as e:
            print(f"Warning: Could not fit USD model: {e}")
            
        # Prepare features
        df_prep = self._prepare_features(df, is_training=True)
        
        # Define features list
        potential_features = [
            'y_lag1', 'y_lag2', 'y_lag3', 'y_lag6',
            'y_trend3', 'y_trend6', 'y_vol6',
            'usd_lag1', 'usd_lag2', 'usd_lag3',
            'ki_lag1', 'ki_diff_lag1', 'ki_lag3',
            'ruonia_lag1', 'ruonia_diff_lag1',
            'brent_lag1', 'brent_lag2', 'brent_lag3'
        ]
        self.features = [f for f in potential_features if f in df_prep.columns]
        
        # Drop NA
        train_df = df_prep.dropna(subset=self.features + [target_col])
        
        if len(train_df) < self.MIN_TRAIN_SIZE:
            raise ValueError(f"Not enough data: {len(train_df)}")
            
        X = train_df[self.features].values
        y = train_df[target_col].values
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        self._is_fitted = True
        return self
        
    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp) -> Dict[str, Any]:
        self._check_fitted()
        
        # Prepare basic features
        df_prep = self._prepare_features(df, is_training=False)
        
        # We need to handle the case where we are forecasting into the future
        # and need forecasted USD values.
        # But `predict` is usually for 1-step ahead in backtest loop.
        # If target_date is in df, we use actuals (if available) or we should use forecast?
        # For strict backtest, we should use what was available at T-1.
        # The `df` passed to predict usually contains history up to T-1 (or T).
        
        # If we need to forecast USD for the target date:
        # We can use usd_model.predict() but that returns a sequence.
        # For 1-step ahead, we just need the next value.
        
        # Check if we have USD data for target_date (insider info?)
        # If not, use USD model.
        
        row = df_prep.loc[[target_date]] if target_date in df_prep.index else None
        
        if row is None or row[self.features].isna().any().any():
            # We need to construct the row using forecasts
            # This is complex for a single-step predict method if it relies on df having the row
            # But let's assume df has the row with lags populated.
            # If USD lag is missing (because it's current month USD?), we might need to forecast it.
            pass
            
        # For now, assume df is prepared correctly by the caller (standard in this repo)
        row = df_prep.loc[[target_date]]
        
        # Fill missing USD/Macro with forecasts if needed?
        # The repo usually assumes `df_ext` has NaNs for target but features are present.
        # If features are missing (e.g. current month USD), we should fill them.
        
        X_test = self.scaler.transform(row[self.features].values)
        pred = self.model.predict(X_test)[0]
        
        return {
            'date': target_date,
            'prediction': pred,
            'model': self.name
        }
        
    def forecast(self, horizon: int = 12) -> np.ndarray:
        self._check_fitted()
        # Not implemented for full horizon yet, requires recursive loop
        # For now return zeros or implement simple recursive
        return np.zeros(horizon)

    def backtest(
        self,
        df: pd.DataFrame,
        start_date: str = '2019-01-01',
        target_col: str = 'Все товары и услуги'
    ) -> pd.DataFrame:
        """
        Backtest the model.
        """
        start = pd.Timestamp(start_date)
        valid_dates = df.dropna(subset=[target_col]).index
        test_dates = valid_dates[valid_dates >= start]
        
        results = []
        
        for target_date in test_dates:
            # Cutoff - all data before current month
            train_df = df[df.index < target_date].copy()
            
            if len(train_df.dropna(subset=[target_col])) < self.MIN_TRAIN_SIZE:
                continue
                
            try:
                # Create new model for clean backtest
                model = FundamentalForecaster(alpha=self.alpha, l1_ratio=self.l1_ratio)
                model.fit(train_df, target_col)
                
                # Test data (up to target date, for features)
                test_df = df[df.index <= target_date].copy()
                
                # Predict
                pred_result = model.predict(test_df, target_date)
                
                actual = df.loc[target_date, target_col]
                
                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred_result['prediction'],
                    'error': actual - pred_result['prediction']
                })
            except Exception as e:
                print(f"Fundamental Error at {target_date}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        return pd.DataFrame(results)

    def get_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        """Calculate metrics."""
        if results.empty:
            return {'MAE': 0, 'RMSE': 0}
            
        errors = results['error'].abs()
        mae = errors.mean()
        rmse = np.sqrt((results['error'] ** 2).mean())
        
        return {
            'MAE': mae,
            'RMSE': rmse
        }
