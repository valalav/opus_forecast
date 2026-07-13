#!/usr/bin/env python3
"""
Backtest script for Exogenous Variables (USD, Brent, Ruonia).
"""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from sirena.exog import SarimaExogForecaster, VarExogForecaster

def load_data():
    """Load inflation_data.csv"""
    df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
    
    cols_to_fix = ['mom', 'usd_nom_i', 'Ruonia', 'brent']
    for col in cols_to_fix:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    # Normalize to Month Start
    df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
    df = df.set_index('Date').sort_index()
    return df

def run_backtest(variable_name: str, horizon: int = 12, start_year: int = 2020):
    """Run backtest for a specific variable."""
    df = load_data()
    
    if variable_name not in df.columns:
        print(f"Error: {variable_name} not found in data")
        return

    ts = df[variable_name].dropna()
    
    # Define test dates (last 24 months)
    start_date = pd.Timestamp(f'{start_year}-01-01')
    test_dates = ts.index[ts.index >= start_date]
    
    print(f"\nBacktesting {variable_name} (Horizon={horizon}, {len(test_dates)} points)")
    print("-" * 60)
    print(f"{'Cutoff':<10} {'Horizon':<8} {'Actual':>10} {'Forecast':>10} {'Error':>10}")
    print("-" * 60)
    
    metrics = []
    
    for cutoff in test_dates:
        # Train data: up to cutoff (inclusive? No, usually train up to cutoff, forecast after)
        # In this loop, 'cutoff' is the DATE WE STAND AT. We forecast cutoff+1...cutoff+h
        
        train = ts[ts.index <= cutoff]
        
        if len(train) < 36:
            print(f"DEBUG: Skipping {cutoff}, train len {len(train)}")
            continue
            
        # Target date for evaluation (let's check h-step ahead)
        target_date = cutoff + pd.DateOffset(months=horizon)
        if target_date not in ts.index:
            continue
            
        actual = ts.loc[target_date]
        
        try:
            print(f"DEBUG: Fitting on {len(train)} pts. Last: {train.index[-1]}")
            # Model (SARIMA) - Use AR(1) for robustness test
            model = SarimaExogForecaster(order=(1,0,0), seasonal_order=(0,0,0,0), auto=False)
            model.fit(train)
            
            # Forecast
            fc = model.forecast(horizon)
            pred = fc.iloc[-1]
            
            print(f"DEBUG: Forecast: {fc.values}")
            
            error = actual - pred
            metrics.append({'cutoff': cutoff, 'actual': actual, 'pred': pred, 'error': error})
            
            print(f"{cutoff.strftime('%Y-%m'):<10} {horizon:<8} {actual:10.2f} {pred:10.2f} {error:10.2f}")
            
        except Exception as e:
            print(f"Error at {cutoff}: {e}")
            
    if metrics:
        df_m = pd.DataFrame(metrics)
        mae = df_m['error'].abs().mean()
        rmse = np.sqrt((df_m['error']**2).mean())
        print("-" * 60)
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        return mae, rmse
    return None, None

def run_var_backtest(horizon: int = 12):
    """Run VAR backtest for multiple variables."""
    df = load_data()
    vars = ['usd_nom_i', 'Ruonia', 'fl_potrb_zad', 'fl_dep', 'all_real'] # Added household finance vars
    
    # Check what exists
    vars = [v for v in vars if v in df.columns]
    
    print(f"\nVAR Backtest for {vars}")
    
    df_var = df[vars].dropna()
    start_date = pd.Timestamp('2020-01-01')
    test_dates = df_var.index[df_var.index >= start_date]
    
    metrics = {v: [] for v in vars}
    
    for cutoff in test_dates:
        train = df_var[df_var.index <= cutoff]
        
        target_date = cutoff + pd.DateOffset(months=horizon)
        if target_date not in df_var.index:
            continue
            
        try:
            model = VarExogForecaster(lags=4)
            # VAR fit might fail if data is constant or short
            model.fit(train)
            
            fc = model.forecast(horizon)
            
            for v in vars:
                actual = df_var.loc[target_date, v]
                pred = fc.loc[target_date, v]
                metrics[v].append(actual - pred)
                
        except Exception as e:
            print(f"VAR Error {cutoff}: {e}")
            pass
            
    print("\nVAR Results (MAE):")
    for v in vars:
        if metrics[v]:
            mae = np.mean(np.abs(metrics[v]))
            print(f"{v}: {mae:.4f}")

if __name__ == "__main__":
    print("=== EXOGENOUS BACKTEST ===")
    
    # Check USD
    run_backtest('usd_nom_i', horizon=1)
    run_backtest('usd_nom_i', horizon=6)
    
    # Check VAR
    run_var_backtest(horizon=6)
