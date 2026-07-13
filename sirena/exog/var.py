"""
VAR Forecaster for multivariate exogenous forecasting.
"""
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from .base import BaseExogForecaster

class VarExogForecaster(BaseExogForecaster):
    """
    Vector Autoregression (VAR) for joint forecasting of multiple macro variables.
    Good for capturing interdependencies (e.g. Oil -> USD -> Inflation Components).
    """
    
    def __init__(self, name: str = "var", lags: int = 4):
        super().__init__(name)
        self.lags = lags
        self.model = None
        self.lag_order = 0
        self.last_train_data = None
        self.last_date = None
        
    def fit(self, df: pd.DataFrame, **kwargs) -> 'VarExogForecaster':
        """
        Fit VAR model.
        
        Args:
            df: pandas DataFrame with multiple columns (variables)
        """
        df_clean = df.dropna()
        self.last_date = df_clean.index.max()
        self.last_train_data = df_clean.values
        
        try:
            model = VAR(df_clean)
            # Find best lag if not specified or check validity
            if self.lags > 0:
                self.model = model.fit(self.lags)
            else:
                self.model = model.fit(maxlags=12, ic='aic')
            
            self.lag_order = self.model.k_ar
            self._is_fitted = True
        except Exception as e:
            print(f"VAR fit failed: {e}")
            self.model = None
            
        return self
        
    def forecast(self, horizon: int) -> pd.DataFrame:
        """
        Forecast future values for all variables.
        
        Returns:
            DataFrame with forecasts
        """
        self._check_fitted()
        
        dates = pd.date_range(start=self.last_date, periods=horizon+1, freq='MS')[1:]
        
        if self.model is None:
            # Fallback: Naive (last row repeated)
            last_row = self.last_train_data[-1]
            params = np.tile(last_row, (horizon, 1))
            return pd.DataFrame(params, index=dates, columns=self.model.names if self.model else None)
            
        try:
            # Forecast
            forecast_input = self.last_train_data[-self.lag_order:]
            fc = self.model.forecast(y=forecast_input, steps=horizon)
            
            return pd.DataFrame(fc, index=dates, columns=self.model.names)
        except Exception as e:
            print(f"VAR forecast failed: {e}")
            return pd.DataFrame()
            
    # Alias to match base signature (although fit takes DataFrame)
    # This is a slightly different signature than base Series, so we might need a separate BaseMultivariate
    # But for now we stick to duck typing or expected usage.
