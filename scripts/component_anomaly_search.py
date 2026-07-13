import pandas as pd
import numpy as np

def analyze_component_signals():
    # Load Component Data
    try:
        # Load long format data
        df_long = pd.read_csv('data/subcomponent_monthly_fixed.csv') 
        df_long['Date'] = pd.to_datetime(df_long['Date'])
        
        # Pivot to Wide Format (Date x Subcomponent)
        df = df_long.pivot_table(index='Date', columns='Субкомпонент', values='MoM_pct')
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Load Aggregate Anomalies (Target)
    backtest = pd.read_csv('docs/long_backtest_results.csv')
    backtest['Date'] = pd.to_datetime(backtest['Date'])
    backtest = backtest.set_index('Date')
    
    # Calculate Anomaly Magnitude (Next Month)
    # We want to see if components Today predict Anomaly Tomorrow
    # Fill NA with 0 if forecast not available yet (for older history), but we need matching dates
    backtest['Next_Anomaly_Mag'] = backtest['Ensemble_Error'].shift(-1).abs()
    
    # Ensure alignment
    common_dates = df.index.intersection(backtest.index)
    if len(common_dates) == 0:
        print("No common dates found between components and backtest results.")
        return
        
    df = df.loc[common_dates]
    target = backtest.loc[common_dates, 'Next_Anomaly_Mag']
    
    # Clean target (remove NaNs at end)
    valid_mask = target.notna()
    df = df[valid_mask]
    target = target[valid_mask]
    
    print(f"Analyzing {len(target)} months of history for {len(df.columns)} components...")
    
    results = []
    
    # 1. Volatility Scanner
    # Does high volatility in a component today predict chaos tomorrow?
    print("Scanning for Volatility Precursors...")
    for col in df.columns:
        # Metric: Absolute Change (Volatility)
        volatility = df[col].abs()
        
        # Metric: Deviation from Trend (Z-score approx)
        rolling_mean = df[col].rolling(12, min_periods=3).mean()
        rolling_std = df[col].rolling(12, min_periods=3).std()
        z_score = ((df[col] - rolling_mean) / rolling_std).abs()
        
        # Correlation with Next Anomaly
        corr_vol = volatility.corr(target)
        corr_z = z_score.corr(target)
        
        # Correlation of simple value (maybe deflation predicts chaos?)
        corr_val = df[col].corr(target)
        
        if abs(corr_vol) > 0.3 or abs(corr_z) > 0.3 or abs(corr_val) > 0.3:
            results.append({
                'Component': col,
                'Signal_Type': 'Component_Metrics',
                'Corr_Value': corr_val,
                'Corr_Volatility': corr_vol,
                'Corr_Z_Score': corr_z
            })

    # 2. Diffusion Index
    # % of components rising > X%
    print("Calculating Diffusion Indices...")
    thresholds = [0.5, 1.0, 1.5, -0.5]
    for t in thresholds:
        if t > 0:
            diffusion = (df > t).sum(axis=1) / len(df.columns)
            name = f'Diffusion_Index_>{t}%'
        else:
            diffusion = (df < t).sum(axis=1) / len(df.columns)
            name = f'Diffusion_Index_<{t}%'
            
        corr_diff = diffusion.corr(target)
        results.append({
            'Component': name,
            'Signal_Type': 'Diffusion',
            'Corr_Value': corr_diff,
            'Corr_Volatility': np.nan,
            'Corr_Z_Score': np.nan
        })

    # Output Results
    res_df = pd.DataFrame(results)
    
    # Sort by strongest correlation found
    res_df['Max_Corr'] = res_df[['Corr_Value', 'Corr_Volatility', 'Corr_Z_Score']].abs().max(axis=1)
    res_df = res_df.sort_values('Max_Corr', ascending=False)
    
    print("\n=== COMPONENT SIGNAL ANALYSIS (Top 15) ===")
    print(res_df[['Component', 'Signal_Type', 'Corr_Value', 'Corr_Volatility', 'Corr_Z_Score']].head(15))
    
    # Save for review
    res_df.to_csv('docs/component_anomaly_signals.csv', index=False)
    print("\nSaved detailed signals to docs/component_anomaly_signals.csv")

if __name__ == "__main__":
    analyze_component_signals()