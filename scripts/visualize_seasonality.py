#!/usr/bin/env python3
"""
Visualize Seasonality of Inflation Components.
Generates HTML charts showing seasonal components.
"""
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
import os
from pathlib import Path

# Create output dir
OUTPUT_DIR = Path('assets/charts')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load data."""
    df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
    
    # Fix columns
    cols = ['mom', 'Prod', 'Nonprod', 'Serv']
    for col in cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    # Normalize to MS
    df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
    df = df.set_index('Date').sort_index()
    return df

def generate_seasonality_chart():
    """Generate seasonality chart."""
    print("Generating seasonality charts...")
    df = load_data()
    
    # Filter for valid range (e.g. from 2016 to avoid old volatility)
    df = df[df.index >= '2016-01-01']
    
    components = {
        'Inflation (MoM)': 'mom',
        'Food (Prod)': 'Prod',
        'Non-Food (Nonprod)': 'Nonprod',
        'Services (Serv)': 'Serv'
    }
    
    fig = make_subplots(
        rows=4, cols=1, 
        subplot_titles=list(components.keys()),
        vertical_spacing=0.05
    )
    
    for i, (name, col) in enumerate(components.items(), 1):
        if col not in df.columns:
            continue
            
        ts = df[col].dropna()
        if len(ts) < 24:
            continue
            
        # Decompose
        # Period=12 for monthly data
        res = seasonal_decompose(ts, model='additive', period=12)
        
        # Plot Seasonal Component
        seasonal = res.seasonal
        
        # Color based on value
        colors = ['red' if v > 0 else 'blue' for v in seasonal]
        
        fig.add_trace(
            go.Bar(
                x=seasonal.index, 
                y=seasonal,
                name=f'{name} Seasonality',
                marker_color=colors
            ),
            row=i, col=1
        )
        
    fig.update_layout(
        title='Seasonal Components of Inflation (Additive Decomposition)',
        height=1000,
        showlegend=False
    )
    
    output_path = OUTPUT_DIR / 'seasonality.html'
    fig.write_html(str(output_path))
    print(f"Saved to {output_path}")
    
    return str(output_path)

if __name__ == "__main__":
    generate_seasonality_chart()
