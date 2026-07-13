import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

class SirenaUSD:
    def __init__(self):
        self.model = None
        self.last_date = None
        self.last_val = None
        self.history = None

    def fit(self, df):
        """
        Fit ARIMA model on USD MoM data.
        df should have 'usd_nom_i' column.
        """
        # Prepare data
        if 'usd_nom_i' not in df.columns:
            raise ValueError("DataFrame must contain 'usd_nom_i'")
            
        # Convert to MoM % (centered around 0)
        self.ts = df['usd_nom_i'] - 100
        self.ts = self.ts.dropna()
        self.last_date = self.ts.index.max()
        self.history = self.ts.values
        
        # Fit ARIMA(1,0,0) or (1,0,1) - simple mean reversion for returns
        # Exchange rate returns often modeled as Random Walk (ARIMA(0,0,0))
        # But we might have some mean reversion or momentum
        
        # Let's try ARIMA(1,0,0)
        self.model = ARIMA(self.ts, order=(1, 0, 0))
        self.model_fit = self.model.fit()
        
        return self

    def predict(self, horizon=12, scenario='base'):
        """
        Predict USD MoM change.
        scenarios: 'base' (model), 'flat' (0 change), 'growth' (+1% monthly)
        """
        if self.model_fit is None:
            raise ValueError("Model not fitted")
            
        # Base forecast
        fc = self.model_fit.forecast(steps=horizon)
        
        # Scenarios
        if scenario == 'flat':
            fc[:] = 0.0
        elif scenario == 'growth':
            fc[:] = 1.0 # 1% monthly depreciation of RUB
        elif scenario == 'strong_growth':
            fc[:] = 2.0
        elif scenario == 'appreciation':
            fc[:] = -1.0
            
        dates = pd.date_range(start=self.last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')
        
        return pd.DataFrame({
            'Date': dates,
            'USD_MoM': fc,
            'USD_Index': fc + 100
        })

    def get_summary(self):
        return self.model_fit.summary()
