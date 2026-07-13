import pandas as pd
import numpy as np
import json
import os

def load_data():
    # Load Historical Data
    df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
    # Clean and parse
    df['mom'] = pd.to_numeric(df['mom'].astype(str).str.replace(',', '.'), errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    
    # Calculate MoM %
    df['val'] = df['mom'] - 100
    
    # Load Forecasts
    with open('data/precomputed_forecasts.json', 'r') as f:
        forecasts = json.load(f)
        
    return df, forecasts

def analyze():
    df, forecasts = load_data()
    
    # 1. Historical Analysis (Jan & Feb)
    jan_hist = df[df['Date'].dt.month == 1]['val']
    feb_hist = df[df['Date'].dt.month == 2]['val']
    
    print("=== HISTORICAL STATS (2010-2025) ===")
    print(f"JANS (n={len(jan_hist)}):")
    print(f"  Mean: {jan_hist.mean():.2f}%")
    print(f"  Median: {jan_hist.median():.2f}%")
    print(f"  Max: {jan_hist.max():.2f}% ({df.loc[jan_hist.idxmax(), 'Date'].year})")
    print(f"  Min: {jan_hist.min():.2f}%")
    print(f"  Std: {jan_hist.std():.2f}")
    
    print(f"\nFEBS (n={len(feb_hist)}):")
    print(f"  Mean: {feb_hist.mean():.2f}%")
    print(f"  Median: {feb_hist.median():.2f}%")
    print(f"  Max: {feb_hist.max():.2f}%")
    print(f"  Min: {feb_hist.min():.2f}%")
    
    # 2. Current Situation (Jan 2026)
    print("\n=== JAN 2026 ESTIMATES ===")
    nowcast_jan = forecasts['forecasts']['Nowcast'][0]
    ensemble_jan = forecasts['forecasts']['Ensemble'][0]
    print(f"Nowcast (Weekly-based): {nowcast_jan:.2f}%")
    print(f"Ensemble (Model-based): {ensemble_jan:.2f}%")
    
    # Deviation
    diff = nowcast_jan - jan_hist.mean()
    print(f"Deviation from Jan Mean: {diff:+.2f} p.p.")
    
    # 3. February Outlook
    print("\n=== FEB 2026 FORECAST ===")
    ensemble_feb = forecasts['forecasts']['Ensemble'][1]
    print(f"Ensemble Forecast: {ensemble_feb:.2f}%")
    print(f"Feb Hist Mean: {feb_hist.mean():.2f}%")
    
    # Trajectory
    print("\n=== TRAJECTORY ANALYSIS ===")
    change = ensemble_feb - nowcast_jan
    print(f"Jan -> Feb Change: {change:+.2f} p.p.")

if __name__ == "__main__":
    analyze()
