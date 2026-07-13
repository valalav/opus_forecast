import pandas as pd
import numpy as np
import sys
import os

# Add root to path
sys.path.append(os.getcwd())

def test_new_models():
    print("Testing New Models...")
    try:
        from sirena.models.fundamental import FundamentalForecaster
        from sirena.models.usd_model import USDForecaster
        
        df = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')
        # Fix data loading
        if 'MoM' in df.columns:
            df = df.rename(columns={'MoM': 'Все товары и услуги'})
        if df['Все товары и услуги'].dtype == object:
            df['Все товары и услуги'] = df['Все товары и услуги'].astype(str).str.replace(',', '.').astype(float)
        df['Date'] = pd.to_datetime(df['Day'], dayfirst=True)
        df = df.set_index('Date')
        
        # Load macro
        macro = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
        macro['Date'] = pd.to_datetime(macro['Date'], dayfirst=True) + pd.offsets.MonthBegin(0)
        macro = macro.set_index('Date')
        
        # Join
        df = df.join(macro[['Ki', 'Ruonia', 'usd_nom_i']], how='left')
        
        # Test USD Model
        usd = USDForecaster()
        usd.fit(df)
        pred_usd = usd.predict(horizon=3)
        print(f"USD Forecast OK: {pred_usd['USD_MoM'].values}")
        
        # Test Fundamental Model
        fund = FundamentalForecaster()
        fund.fit(df)
        pred_fund = fund.predict(df, df.index[-1])
        print(f"Fundamental Forecast OK: {pred_fund}")
        
    except Exception as e:
        print(f"FAILED New Models: {e}")
        import traceback
        traceback.print_exc()

def test_dashboard_model():
    print("\nTesting Dashboard Model (SirenaKBR_v24)...")
    try:
        from dashboard import SirenaKBR_v24, load_data
        
        df, df_weekly = load_data()
        if df is None:
            raise ValueError("Could not load data for dashboard")
            
        model = SirenaKBR_v24()
        if df_weekly is not None:
            model.set_weekly_data(df_weekly)
            
        model.fit(df)
        
        last_date = df.index.max()
        start_date = last_date + pd.DateOffset(months=1)
        
        forecast = model.predict_horizon(df, start_date, horizon=3)
        print("Dashboard Forecast OK")
        print(forecast[['Date', 'MoM']].head())
        
    except Exception as e:
        print(f"FAILED Dashboard Model: {e}")
        import traceback
        traceback.print_exc()

def test_legacy_imports():
    print("\nTesting Legacy Imports...")
    try:
        from sirena.legacy.sirena_arima import SirenaARIMA
        from sirena.legacy.sirena_ets import SirenaETS
        print("Legacy imports OK")
    except Exception as e:
        print(f"FAILED Legacy Imports: {e}")

if __name__ == "__main__":
    test_new_models()
    test_dashboard_model()
    test_legacy_imports()
