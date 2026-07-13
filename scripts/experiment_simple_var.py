
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from packaging import version

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

def run_rolling_backtest():
    df = load_data()
    data = df[['Все товары и услуги', 'Продовольственные товары', 'Непродовольственные товары', 'Услуги']].dropna()
    
    # Check if data is Index (100+) or MoM. Looks like Index from previous output (100.6)
    # Convert to MoM % for error calculation if needed, but MAE is on raw scale. 
    # If raw scale is Index, MAE 0.4 means 0.4 p.p. which is huge if MoM is 0.5%.
    # Wait, 100.4 vs 100.8 -> error 0.4.
    # Yes, standard metric is MAE on MoM %.
    # If data is Index (100.5), subtracting 100 gives MoM %.
    
    print(f"Data range: {data.index.min()} to {data.index.max()}")
    
    # Test period: 2023-2025 (last 36 months)
    test_start = pd.Timestamp('2023-01-01')
    test_dates = data.index[data.index >= test_start]
    
    horizon = 12
    lags_list = [1, 3, 6, 12]
    
    results = {l: [] for l in lags_list}
    
    for cutoff in test_dates:
        # Train up to cutoff (exclusive or inclusive? usually predict FROM cutoff)
        # Let's say we stand at cutoff-1 and predict cutoff...cutoff+h
        # Train on data < cutoff
        train = data[data.index < cutoff]
        
        target_date = cutoff # h=1 target
        # Actually we want h=1..12 metrics avg? Or usually just h=1, h=12 accuracy?
        # Let's measure h=1 accuracy for simplicity first, or h=12 trajectory.
        # User asked "Why no adequate VAR?". Usually implies stability.
        
        # Let's measure MAE at h=1
        if len(train) < 24: continue
        
        for lags in lags_list:
            try:
                model = VAR(train)
                # Check degrees of freedom
                # k=4, p=lags. Params = 4 + 4*4*p = 4 + 16p.
                # Observations = T. Need T > 16p + 4 ideally.
                if len(train) <= 16 * lags + 4:
                    # results[lags].append(np.nan)
                    continue
                    
                fit = model.fit(lags)
                lag_order = fit.k_ar
                fc = fit.forecast(y=train.values[-lag_order:], steps=horizon)
                
                # Compare h=1 forecast
                pred_h1 = fc[0, 0]
                actual_h1 = data.loc[cutoff, 'Все товары и услуги']
                
                # Calculate error
                results[lags].append(abs(pred_h1 - actual_h1))
                
            except Exception:
                continue
                
    print("\nRolling Backtest Results (MAE h=1):")
    for lags, errors in results.items():
        if len(errors) > 0:
            print(f"VAR({lags}): {np.mean(errors):.4f} (n={len(errors)})")
        else:
            print(f"VAR({lags}): Insufficient data")

if __name__ == "__main__":
    run_rolling_backtest()
