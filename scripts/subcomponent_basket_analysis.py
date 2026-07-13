import pandas as pd
import numpy as np

def analyze_subcomponents():
    print("Loading Subcomponent Data...")
    try:
        # Load long format data
        df_long = pd.read_csv('data/subcomponent_monthly_fixed.csv') 
        df_long['Date'] = pd.to_datetime(df_long['Date'])
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Calculate Contribution
    df_long['Contribution'] = df_long['MoM_pct'] * df_long['weight']
    
    # Pivot
    pivot_mom = df_long.pivot_table(index='Date', columns='Субкомпонент', values='MoM_pct')
    pivot_contrib = df_long.pivot_table(index='Date', columns='Субкомпонент', values='Contribution')
    pivot_weight = df_long.pivot_table(index='Date', columns='Субкомпонент', values='weight')
    
    # Load Target (Ensemble Error)
    backtest = pd.read_csv('docs/long_backtest_results.csv')
    backtest['Date'] = pd.to_datetime(backtest['Date'])
    target = backtest.set_index('Date')['Ensemble_Error'].abs()
    
    # Next Month Anomaly
    target_next = target.shift(-1)
    
    # Align
    common_dates = pivot_mom.index.intersection(target_next.index)
    if len(common_dates) < 12:
        print("Not enough overlap with backtest.")
        # Fallback: Use Total CPI Volatility as proxy for Anomaly if backtest is short?
        # But we want to answer the specific question.
        # Let's use the 14 months we have.
    
    pivot_mom = pivot_mom.loc[common_dates]
    pivot_contrib = pivot_contrib.loc[common_dates]
    target_next = target_next.loc[common_dates]
    
    results = []
    
    print(f"Analyzing {len(common_dates)} months...")
    
    for col in pivot_mom.columns:
        mom = pivot_mom[col]
        contrib = pivot_contrib[col]
        weight = pivot_weight[col].mean()
        
        # 1. Anomaly Prediction (Risk)
        # Does high contribution today predict Anomaly tomorrow?
        # We use absolute contribution magnitude
        corr_risk = contrib.abs().corr(target_next)
        
        # 2. Mean Reversion
        # Does high MoM today predict low MoM tomorrow?
        mom_next = mom.shift(-1)
        corr_rev = mom.corr(mom_next)
        
        # 3. Weighted Reversion Impact
        # Does high Contribution today predict Negative Contribution tomorrow?
        contrib_next = contrib.shift(-1)
        corr_rev_w = contrib.corr(contrib_next)
        
        results.append({
            'Component': col,
            'Weight': weight,
            'Corr_Risk': corr_risk,
            'Corr_Mean_Reversion': corr_rev,
            'Corr_Weighted_Reversion': corr_rev_w
        })
        
    res_df = pd.DataFrame(results).sort_values('Corr_Risk', ascending=False)
    
    print("\n=== TOP PREDICTORS OF ANOMALY (Weighted Impact) ===")
    print(res_df[['Component', 'Weight', 'Corr_Risk']].head(10))
    
    print("\n=== STRONGEST MEAN REVERSION (Negative Correlation) ===")
    print(res_df.sort_values('Corr_Mean_Reversion')[['Component', 'Weight', 'Corr_Mean_Reversion']].head(10))
    
    res_df.to_csv('docs/subcomponent_basket_signals.csv', index=False)

if __name__ == "__main__":
    analyze_subcomponents()
