import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings('ignore')

class SirenaARIMA:
    """
    Baseline models for Inflation Forecasting:
    1. AR(1): Simple AutoRegression
    2. SARIMA: Seasonal ARIMA
    """
    
    def __init__(self, order=(1,0,0), seasonal_order=(1,0,1,12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fit_res = None
        
    def fit_ar1(self, series):
        """Fit AR(1) model."""
        self.model = ARIMA(series, order=(1,0,0))
        self.fit_res = self.model.fit()
        return self.fit_res
    
    def fit_sarima(self, series):
        """Fit SARIMA model."""
        # Using (1,0,1)x(1,0,1,12) as default for monthly inflation
        self.model = SARIMAX(series, order=self.order, seasonal_order=self.seasonal_order,
                             enforce_stationarity=False, enforce_invertibility=False)
        self.fit_res = self.model.fit(disp=False)
        return self.fit_res
        
    def forecast(self, steps=12):
        """Generate forecast."""
        if self.fit_res is None:
            raise ValueError("Model not fitted.")
        
        forecast = self.fit_res.get_forecast(steps=steps)
        mean_forecast = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05) # 95% CI
        
        return {
            'mean': mean_forecast,
            'lower': conf_int.iloc[:, 0],
            'upper': conf_int.iloc[:, 1],
            'aic': self.fit_res.aic
        }

if __name__ == "__main__":
    # Test
    print("SirenaARIMA class ready.")