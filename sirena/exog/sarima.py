"""
SARIMA Forecaster for exogenous variables.
"""
import pandas as pd
import numpy as np
import warnings
from typing import Optional, Tuple, Dict, Any

try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False

from statsmodels.tsa.statespace.sarimax import SARIMAX
from .base import BaseExogForecaster

class SarimaExogForecaster(BaseExogForecaster):
    """
    SARIMA model for univariate exogenous forecasting.
    Can use auto_arima to find best parameters or use fixed parameters.
    """
    
    def __init__(self, 
                 name: str = "sarima", 
                 auto: bool = True, 
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0)):
        super().__init__(name)
        self.auto = auto
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.last_date = None
        self.last_value = None
        
    def fit(self, series: pd.Series, **kwargs) -> 'SarimaExogForecaster':
        """
        Fit SARIMA model.
        
        Args:
            series: pandas Series (must have DatetimeIndex)
        """
        # Ensure series is numeric and drop NaNs
        ts = pd.to_numeric(series, errors='coerce').dropna()
        self.last_date = ts.index.max()
        self.last_value = ts.iloc[-1]
        
        if self.auto and PMDARIMA_AVAILABLE:
            try:
                self.model = pm.auto_arima(
                    ts,
                    start_p=1, start_q=1,
                    max_p=3, max_q=3,
                    m=12 if 'seasonal' in kwargs.get('type', '') else 1,
                    start_P=0, seasonal=True, # Allow seasonality check
                    d=None, D=1, trace=False,
                    error_action='ignore',  
                    suppress_warnings=True, 
                    stepwise=True
                )
            except Exception as e:
                print(f"AutoARIMA failed for {self.name}: {e}. Fallback to fixed.")
                self._fit_fixed(ts)
        else:
            self._fit_fixed(ts)
            
        self._is_fitted = True
        return self
        
    def _fit_fixed(self, ts: pd.Series):
        """Fit with fixed parameters using statsmodels"""
        try:
            # Enforce float64 to avoid dtype issues
            ts = ts.astype(float)
            if ts.index.freq is None:
                ts.index = pd.DatetimeIndex(ts.index).to_period('M').to_timestamp()
                ts.index.freq = 'MS'
            
            # Debug data quality
            # print(f"DEBUG: ts tail: \n{ts.tail()}")
            
            mod = SARIMAX(ts, order=self.order, seasonal_order=self.seasonal_order, 
                          enforce_stationarity=False, enforce_invertibility=False)
            self.model = mod.fit(disp=False)
        except Exception as e:
            print(f"SARIMAX fixed failed: {e}")
            self.model = None

    def forecast(self, horizon: int) -> pd.Series:
        """Forecast future values."""
        self._check_fitted()
        
        dates = pd.date_range(start=self.last_date, periods=horizon+1, freq='MS')[1:]

        if self.model is None:
             # Fallback: Naive forecast (last value)
            return pd.Series([self.last_value] * horizon, index=dates)
            
        try:
            if self.auto and PMDARIMA_AVAILABLE and hasattr(self.model, 'predict'):
                preds = self.model.predict(n_periods=horizon)
            else:
                # Use get_forecast which is more robust
                # Workaround: request more steps to avoid last-step NaN issue
                preds = self.model.get_forecast(steps=horizon + 2).predicted_mean
                preds = preds.iloc[:horizon]
            
            return pd.Series(preds.values, index=dates)
        except Exception as e:
            print(f"Forecast failed for {self.name}: {e}")
            return pd.Series([self.last_value] * horizon, index=dates)
