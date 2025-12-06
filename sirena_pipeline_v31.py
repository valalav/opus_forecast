import pandas as pd
import numpy as np
from sirena_kbr_v2_4_auto import SirenaKBR_v24
from sirena_bvar import BayesianVAR
from sirena_arima import SirenaARIMA
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings('ignore')

class SirenaEnsemble:
    """
    Ensemble Model v3.1:
    Combines Ridge (Trend/Macro) + BVAR (Probabilistic) + ETS/SARIMA (Seasonality)
    """
    
    def __init__(self):
        self.ridge = SirenaKBR_v24()
        self.bvar = None
        self.sarima = SirenaARIMA(order=(1,0,1), seasonal_order=(1,0,1,12))
        
        # Ensemble Weights (Tuned via backtest)
        self.weights = {
            'Ridge': 0.60,
            'BVAR': 0.30,
            'SARIMA': 0.10
        }
        
    def prepare_data(self, ridge_file, bvar_file):
        # Load Data
        self.df_ridge = pd.read_csv(ridge_file, sep=';', decimal=',')
        self.df_bvar = pd.read_csv(bvar_file, sep=';', decimal=',')
        
        # Robust cleaning for Ridge
        if 'MoM' in self.df_ridge.columns:
             if self.df_ridge['MoM'].dtype == object:
                 self.df_ridge['MoM'] = self.df_ridge['MoM'].astype(str).str.replace(',', '.')
             self.df_ridge['MoM'] = pd.to_numeric(self.df_ridge['MoM'], errors='coerce')

        # Robust cleaning for BVAR
        cols_bvar = ['mom', 'Prod', 'Nonprod', 'Serv', 'usd_nom_i', 'Ruonia']
        for col in cols_bvar:
            if col in self.df_bvar.columns:
                if self.df_bvar[col].dtype == object:
                    self.df_bvar[col] = self.df_bvar[col].astype(str).str.replace(',', '.')
                self.df_bvar[col] = pd.to_numeric(self.df_bvar[col], errors='coerce')
        
        # Fix Dates
        for df in [self.df_ridge, self.df_bvar]:
            if 'Day' in df.columns:
                df['Date'] = pd.to_datetime(df['Day'], format='%d.%m.%Y', errors='coerce')
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
            if df['Date'].isna().any():
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # Fallback
        
        # Pivot Ridge
        if 'Товар' in self.df_ridge.columns:
            self.df_ridge = self.df_ridge.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
        else:
            self.df_ridge = self.df_ridge.set_index('Date')
            
        # BVAR Vars
        self.df_bvar = self.df_bvar.set_index('Date')
        
        self.df_ridge = self.df_ridge.sort_index()
        self.df_bvar = self.df_bvar.sort_index()
        
    def forecast(self, horizon=12):
        # 1. Ridge Forecast
        last_date = self.df_ridge.index.max()
        start_fc = last_date + pd.DateOffset(months=1)
        
        self.ridge.fit(self.df_ridge)
        fc_ridge = self.ridge.predict_horizon(self.df_ridge, start_fc, horizon)
        path_ridge = fc_ridge['MoM_Index'].values
        
        # 2. BVAR Forecast
        bvar_data = pd.DataFrame()
        # Using names from inflation_data.csv
        bvar_data['CPI'] = self.df_bvar['mom'] - 100
        bvar_data['Food'] = self.df_bvar['Prod'] - 100
        bvar_data['USD'] = self.df_bvar['usd_nom_i'] - 100
        bvar_data['RUONIA'] = self.df_bvar['Ruonia']
        bvar_data = bvar_data.dropna()
        
        self.bvar = BayesianVAR(bvar_data, ['CPI', 'Food', 'USD', 'RUONIA'], lags=2)
        self.bvar.fit(lambda1=0.5)
        fc_bvar = self.bvar.forecast(h=horizon)
        path_bvar = fc_bvar['median'][:, 0] + 100 # Convert back to Index
        
        # 3. SARIMA Forecast
        ts = self.df_ridge['Все товары и услуги'] - 100
        self.sarima.fit_sarima(ts)
        fc_sarima = self.sarima.forecast(steps=horizon)
        path_sarima = fc_sarima['mean'].values + 100
        
        # 4. Ensemble
        dates = pd.date_range(start=start_fc, periods=horizon, freq='MS')
        
        ensemble_path = (
            path_ridge * self.weights['Ridge'] +
            path_bvar * self.weights['BVAR'] +
            path_sarima * self.weights['SARIMA']
        )
        
        return pd.DataFrame({
            'Date': dates,
            'Ensemble': ensemble_path - 100, # MoM %
            'Ridge': path_ridge - 100,
            'BVAR': path_bvar - 100,
            'SARIMA': path_sarima - 100
        })

if __name__ == "__main__":
    model = SirenaEnsemble()
    model.prepare_data('data/infl_kbr.csv', 'data/inflation_data.csv')
    fc = model.forecast(12)
    print("ENSEMBLE FORECAST:")
    print(fc)
    fc.to_csv('ensemble_forecast.csv', index=False)