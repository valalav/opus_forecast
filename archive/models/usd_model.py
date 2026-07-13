import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional

from ..macro_features import load_brent_prices

class USDForecaster:
    """
    Forecasting model for USD/RUB exchange rate (MoM change).
    Uses ElasticNet with Oil prices (Brent) and Key Rate (Ki) as features.
    """
    
    def __init__(self):
        self.model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        self.scaler = StandardScaler()
        self.features = []
        self._is_fitted = False
        self.last_date = None
        self.last_row = None
        
    def _prepare_data(self, df: pd.DataFrame, brent_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare data for modeling.
        Aligns dates to Month Start.
        """
        df = df.copy()
        
        # Normalize dates to Month Start
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Save original index to restore later if needed, but work with MS
        df['orig_date'] = df.index
        df.index = df.index + pd.offsets.MonthBegin(0)
        
        # Load Brent if not provided
        if brent_df is None:
            try:
                brent_df = load_brent_prices()
                # Ensure Brent is MS
                if not isinstance(brent_df.index, pd.DatetimeIndex):
                    brent_df.index = pd.to_datetime(brent_df.index)
                brent_df.index = brent_df.index + pd.offsets.MonthBegin(0)
            except Exception:
                print("Warning: Could not load Brent data")
                brent_df = pd.DataFrame()
        
        # Merge Brent
        if not brent_df.empty:
            # Drop existing brent columns to avoid overlap
            cols_to_drop = [c for c in ['brent', 'brent_pct'] if c in df.columns]
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                
            df = df.join(brent_df[['brent', 'brent_pct']], how='left')
        else:
            if 'brent' not in df.columns:
                df['brent'] = np.nan
            if 'brent_pct' not in df.columns:
                df['brent_pct'] = 0.0
            
        # Target: USD MoM (assuming usd_nom_i is index)
        if 'usd_nom_i' in df.columns:
            df['usd_mom'] = df['usd_nom_i'] - 100
        else:
            raise ValueError("Column 'usd_nom_i' not found")
            
        # Features construction
        # Lags of target
        for i in [1, 2, 3, 6]:
            df[f'usd_lag{i}'] = df['usd_mom'].shift(i)
            
        # Momentum
        df['usd_ma3'] = df['usd_mom'].rolling(3).mean().shift(1)
        
        # Oil features
        if 'brent_pct' in df.columns:
            # Fill NA in brent_pct (e.g. forward fill or 0)
            df['brent_pct'] = df['brent_pct'].fillna(0)
            for i in [1, 2, 3]:
                df[f'brent_lag{i}'] = df['brent_pct'].shift(i)
                
        # Key Rate features
        if 'Ki' in df.columns:
            # Handle missing Ki (fill with previous or median)
            df['Ki'] = df['Ki'].fillna(method='ffill')
            
            df['ki_diff'] = df['Ki'].diff()
            for i in [1, 3]:
                df[f'ki_lag{i}'] = df['Ki'].shift(i)
                df[f'ki_diff_lag{i}'] = df['ki_diff'].shift(i)
                
        # Ruonia features
        if 'Ruonia' in df.columns:
            df['Ruonia'] = df['Ruonia'].fillna(method='ffill')
            df['spread'] = df['Ki'] - df['Ruonia']
            for i in [1, 3]:
                df[f'ruonia_lag{i}'] = df['Ruonia'].shift(i)
                df[f'spread_lag{i}'] = df['spread'].shift(i)
                
        return df
        
    def fit(self, df: pd.DataFrame):
        """Fit the model."""
        data = self._prepare_data(df)
        
        # Define features based on available columns
        potential_features = [
            'usd_lag1', 'usd_lag2', 'usd_lag3', 'usd_lag6', 'usd_ma3',
            'brent_lag1', 'brent_lag2', 'brent_lag3',
            'ki_lag1', 'ki_diff_lag1', 'ruonia_lag1', 'spread_lag1'
        ]
        self.features = [f for f in potential_features if f in data.columns]
        
        # Drop NA for training
        train_data = data.dropna(subset=self.features + ['usd_mom'])
        
        if len(train_data) < 12:
            raise ValueError("Not enough data to fit USD model")
            
        X = train_data[self.features].values
        y = train_data['usd_mom'].values
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        self._is_fitted = True
        self.last_date = data.index.max()
        self.last_row = data.iloc[-1]
        
        return self
        
    def predict(self, df: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
        """
        Forecast USD MoM for horizon.
        Uses recursive strategy.
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted")
            
        # Re-prepare data to get the latest state correctly
        data = self._prepare_data(df)
        last_date = data.index.max()
        
        # Initialize forecast loop
        current_row = data.iloc[-1].copy()
        predictions = []
        dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')
        
        # We need to update lags recursively
        # For simplicity, we'll keep macro features (Oil, Ki) constant or decay them
        # Ideally we would forecast them too, but for now we assume flat/naive for exogenous
        
        for i in range(horizon):
            # Construct feature vector
            row_features = []
            for feat in self.features:
                if feat == 'usd_lag1':
                    val = current_row['usd_mom']
                elif feat == 'usd_lag2':
                    val = current_row['usd_lag1'] # Approximation from previous step
                elif feat.startswith('usd_lag'):
                    # For deeper lags, we need history. 
                    # Simplified: keep using last known values or shift from history if we had full history vector
                    # Better: maintain a small history buffer
                    val = current_row.get(feat, 0) 
                else:
                    # Exogenous: keep constant (naive)
                    val = current_row.get(feat, 0)
                row_features.append(val)
            
            # Predict
            X_test = self.scaler.transform([row_features])
            pred_mom = self.model.predict(X_test)[0]
            
            predictions.append(pred_mom)
            
            # Update current_row for next step (recursive)
            # Shift lags
            current_row['usd_lag2'] = current_row['usd_lag1']
            current_row['usd_lag1'] = pred_mom
            current_row['usd_mom'] = pred_mom
            
        return pd.DataFrame({
            'Date': dates,
            'USD_MoM': predictions,
            'USD_Index': np.array(predictions) + 100
        })
