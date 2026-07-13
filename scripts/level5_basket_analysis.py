import pandas as pd
import numpy as np

def analyze_level5_basket():
    print("Loading Item Names...")
    try:
        names_df = pd.read_csv('data/items_names.csv')
        names_map = names_df.drop_duplicates('Item_code').set_index('Item_code')['Item_name'].to_dict()
    except Exception as e:
        print(f"Error loading names: {e}")
        return

    print("Loading Level 5 Data (KBR Monthly)...")
    try:
        # Load the file verified to have monthly data
        micro = pd.read_csv('data/kbr_full_monthly.csv')
        
        if 'Region_code' in micro.columns:
            micro = micro[micro['Region_code'] == 7]
            
        micro['Date'] = pd.to_datetime(micro['Date'])
        # Handle 'MoM' which is index (e.g. 101.5)
        micro['MoM'] = pd.to_numeric(micro['MoM'], errors='coerce')
        micro = micro.dropna(subset=['MoM'])
        
        print(f"Loaded {len(micro)} rows.")
        print(f"Date range: {micro['Date'].min()} to {micro['Date'].max()}")
        
    except Exception as e:
        print(f"Error loading micro data: {e}")
        return

    print("Loading Weights (Region 7)...")
    try:
        # Using the same logic as before, which seemed to find weights
        chunks = []
        for chunk in pd.read_csv('data/access_weights.csv', chunksize=50000):
            filtered = chunk[chunk['Region_code'] == 7]
            if not filtered.empty:
                chunks.append(filtered)
        weights = pd.concat(chunks)
        weights['Day'] = pd.to_datetime(weights['Day'], format='%d/%m/%y %H:%M:%S')
        weights['Year'] = weights['Day'].dt.year
        
        # Keep unique weights per Year-Item
        w_subset = weights[['Year', 'Item_code', 'Weight_vertical']].drop_duplicates()
        print(f"Loaded weights for {len(w_subset)} Year-Item pairs.")
        
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    print("Merging Data...")
    micro['Year'] = micro['Date'].dt.year
    
    # Inner merge to ensure we have weights
    merged = pd.merge(micro, w_subset, on=['Year', 'Item_code'], how='inner')
    
    # Calculate Impact (Contribution)
    # MoM is index (100.5), we need pct (0.5)
    merged['MoM_pct'] = merged['MoM'] - 100.0
    merged['Contribution'] = merged['MoM_pct'] * merged['Weight_vertical']
    
    print(f"Merged rows: {len(merged)}")
    
    # Load Target (Ensemble Error)
    print("Loading Target...")
    backtest = pd.read_csv('docs/long_backtest_results.csv')
    backtest['Date'] = pd.to_datetime(backtest['Date'])
    target_series = backtest.set_index('Date')['Ensemble_Error'].abs()
    
    # Target is NEXT Month's Error
    target_next = target_series.shift(-1)
    
    # Pivot for Analysis
    print("Pivoting...")
    pivot_contrib = merged.pivot_table(index='Date', columns='Item_code', values='Contribution', aggfunc='sum')
    pivot_mom = merged.pivot_table(index='Date', columns='Item_code', values='MoM_pct', aggfunc='mean')
    
    # Align
    common_dates = pivot_contrib.index.intersection(target_next.index)
    print(f"Common Analysis Months: {len(common_dates)}")
    
    if len(common_dates) < 12:
        print("Warning: Low overlap.")
    
    pivot_contrib = pivot_contrib.loc[common_dates]
    pivot_mom = pivot_mom.loc[common_dates]
    target = target_next.loc[common_dates]
    
    results = []
    
    print("Calculating Metrics...")
    for item_code in pivot_contrib.columns:
        # Filter aggregates (heuristic: usually codes < 200 are groups, but not always)
        # Let's rely on names later or list top ones.
        # Actually, "Топливо моторное" is ~10-20? No, subcomponents have names.
        # Micro items usually have higher codes.
        
        contrib = pivot_contrib[item_code]
        mom = pivot_mom[item_code]
        
        # Filter sparse items
        if contrib.notna().sum() < 12: continue
        
        # 1. Risk Correlation (Contribution -> Next Error)
        corr_risk = contrib.abs().corr(target)
        
        # 2. Mean Reversion (MoM -> Next MoM)
        mom_next = mom.shift(-1)
        # We need overlap for this calculation
        valid_rev = mom.notna() & mom_next.notna()
        if valid_rev.sum() < 12:
            corr_rev = np.nan
        else:
            corr_rev = mom.corr(mom_next)
            
        item_name = names_map.get(item_code, f"Code_{item_code}")
        avg_weight = merged[merged['Item_code'] == item_code]['Weight_vertical'].mean()
        
        results.append({
            'Code': item_code,
            'Name': item_name,
            'Weight': avg_weight,
            'Corr_Risk': corr_risk, # Higher = Warning Signal
            'Corr_Reversion': corr_rev # Negative = Mean Reversion
        })
        
    res_df = pd.DataFrame(results)
    
    # Filter out likely aggregates based on Weight or Code?
    # Items with Weight > 0.05 (5%) are likely aggregates.
    res_df = res_df[res_df['Weight'] < 0.05]
    
    print("\n=== TOP LEVEL 5 PREDICTORS (Risk of Chaos) ===")
    print(res_df.sort_values('Corr_Risk', ascending=False).head(15)[['Name', 'Weight', 'Corr_Risk']])
    
    print("\n=== TOP LEVEL 5 MEAN REVERSION (Bouncing Prices) ===")
    print(res_df.sort_values('Corr_Reversion', ascending=True).head(15)[['Name', 'Weight', 'Corr_Reversion']])
    
    res_df.to_csv('docs/level5_basket_analysis.csv', index=False)
    print("\nSaved to docs/level5_basket_analysis.csv")

if __name__ == "__main__":
    analyze_level5_basket()
