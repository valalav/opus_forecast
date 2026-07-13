
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sirena.models.bvar import BVARForecaster

def load_data():
    df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
    df = df.rename(columns={
        'mom': 'Все товары и услуги',
        'Prod': 'Продовольственные товары',
        'Nonprod': 'Непродовольственные товары',
        'Serv': 'Услуги'
    })
    
    # Clean numeric
    cols = ['Все товары и услуги', 'Продовольственные товары', 'Непродовольственные товары', 'Услуги']
    for col in cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
    df = df.set_index('Date').sort_index()
    return df

def run_diagnosis():
    df = load_data()
    print(f"Loaded data: {df.shape}")
    
    # Train on full history
    train = df.dropna()
    print(f"Training on {len(train)} points. Last date: {train.index.max()}")
    
    configs = [
        {"name": "Default (0.2)", "lambda1": 0.2, "auto_lambda": False},
        {"name": "Loose (1.0)", "lambda1": 1.0, "auto_lambda": False},
        {"name": "Super Loose (5.0)", "lambda1": 5.0, "auto_lambda": False},
        {"name": "Auto Lambda", "lambda1": 0.2, "auto_lambda": True},
    ]
    
    horizon = 12
    last_val = train['Все товары и услуги'].iloc[-1]
    
    print(f"\nLast actual value: {last_val}")
    print("-" * 60)
    print(f"{'Model':<20} | {'h=1':<8} | {'h=6':<8} | {'h=12':<8} | {'Trend'}")
    print("-" * 60)
    
    for cfg in configs:
        try:
            model = BVARForecaster(
                lags=1,
                lambda1=cfg['lambda1'],
                auto_lambda=cfg['auto_lambda'],
                n_draws=2000
            )
            model.fit(train)
            
            fc = model.forecast_full(horizon)
            mean_fc = fc['mean']
            
            # Convert if needed (raw data is 100+mom?)
            # The model subtracts 100 internally if mean > 50, but returns raw output?
            # Let's check logic. _prepare converts to mom. forecast uses raw_data which is mom.
            # So output is mom.
            
            h1 = mean_fc[0]
            h6 = mean_fc[5]
            h12 = mean_fc[11]
            
            # Check if array
            if np.ndim(h1) > 0:
                h1_val = h1[0]
                h6_val = h6[0]
                h12_val = h12[0]
            else:
                h1_val, h6_val, h12_val = h1, h6, h12

            # Trend description
            trend = "Flat"
            if h12_val > h1_val + 0.1: trend = "Up"
            elif h12_val < h1_val - 0.1: trend = "Down"
            
            # Auto param result
            extra = ""
            if cfg['auto_lambda']:
                extra = f"(Best: {model.optimal_lambda1:.3f})"
                
            print(f"{cfg['name']:<20} | {h1_val:.4f}   | {h6_val:.4f}   | {h12_val:.4f}   | {trend} {extra}")
            
        except Exception as e:
            print(f"{cfg['name']:<20} | ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_diagnosis()
