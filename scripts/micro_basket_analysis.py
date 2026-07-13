import pandas as pd
import numpy as np

def analyze_micro_basket():
    print("Loading Item Names...")
    try:
        names_df = pd.read_csv('data/items_names.csv')
        names_map = names_df.drop_duplicates('Item_code').set_index('Item_code')['Item_name'].to_dict()
    except Exception as e:
        print(f"Error loading names: {e}")
        return

    print("Loading Micro Data (KBR)...")
    try:
        micro = pd.read_csv('data/kbr_micro_full.csv')
        if 'Region_code' in micro.columns:
            micro = micro[micro['Region_code'] == 7]
        micro['Date'] = pd.to_datetime(micro['Date'])
        micro['MoM'] = pd.to_numeric(micro['MoM'], errors='coerce')
        micro = micro.dropna(subset=['MoM'])
        print(f"Micro dates: {micro['Date'].min()} to {micro['Date'].max()}")
    except Exception as e:
        print(f"Error loading micro data: {e}")
        return

    print("Loading Weights (Region 7)...")
    try:
        chunks = []
        for chunk in pd.read_csv('data/access_weights.csv', chunksize=50000):
            filtered = chunk[chunk['Region_code'] == 7]
            if not filtered.empty:
                chunks.append(filtered)
        weights = pd.concat(chunks)
        weights['Day'] = pd.to_datetime(weights['Day'], format='%d/%m/%y %H:%M:%S')
        print(f"Weight years: {weights['Day'].dt.year.unique()}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    print("Merging Data...")
    micro['Year'] = micro['Date'].dt.year
    weights['Year'] = weights['Day'].dt.year
    w_subset = weights[['Year', 'Item_code', 'Weight_vertical']].drop_duplicates()
    
    merged = pd.merge(micro, w_subset, on=['Year', 'Item_code'], how='inner')
    merged['MoM_pct'] = merged['MoM'] - 100.0
    merged['Impact'] = merged['MoM_pct'] * merged['Weight_vertical']
    
    print(f"Merged dates: {merged['Date'].min()} to {merged['Date'].max()}")
    
    # Load Target (Anomalies)
    backtest = pd.read_csv('docs/long_backtest_results.csv')
    backtest['Date'] = pd.to_datetime(backtest['Date'])
    target_series = backtest.set_index('Date')['Ensemble_Error'].abs()
    target_next = target_series.shift(-1)
    
    print(f"Target dates (raw): {target_series.index.min()} to {target_series.index.max()}")
    
    print("Pivoting...")
    impact_pivot = merged.pivot_table(index='Date', columns='Item_code', values='Impact', aggfunc='sum')
    mom_pivot = merged.pivot_table(index='Date', columns='Item_code', values='MoM_pct', aggfunc='mean')
    
    # Align
    common_dates = impact_pivot.index.intersection(target_next.index)
    
    print(f"Common Months: {len(common_dates)}")
    print(f"Sample Common Dates: {common_dates[:5]}")
    
    if len(common_dates) < 2:
        return

    impact_pivot = impact_pivot.loc[common_dates]
    mom_pivot = mom_pivot.loc[common_dates]
    target = target_next.loc[common_dates]
    
    results = []
    
    print("Calculating Correlations...")
    for item_code in impact_pivot.columns:
        # Filter aggregates: Usually codes < 200 are groups
        if item_code < 200: continue 
        
        impact_series = impact_pivot[item_code]
        mom_series = mom_pivot[item_code]
        
        # Check valid data points
        valid_points = impact_series.notna() & target.notna()
        # Lower threshold to debug
        if valid_points.sum() < 6: continue
        
        corr_risk = impact_series.corr(target)
        
        mom_next = mom_series.shift(-1)
        valid_reversion = mom_series.notna() & mom_next.notna()
        if valid_reversion.sum() < 6:
            corr_reversion = np.nan
        else:
            corr_reversion = mom_series.corr(mom_next)
        
        avg_weight = merged[merged['Item_code'] == item_code]['Weight_vertical'].mean()
        item_name = names_map.get(item_code, f"Unknown_{item_code}")
        
        results.append({
            'Item_Code': item_code,
            'Item_Name': item_name,
            'Avg_Weight': avg_weight,
            'Corr_Risk': corr_risk,
            'Corr_Reversion': corr_reversion,
            'N_Points': valid_points.sum()
        })
            
    res_df = pd.DataFrame(results).sort_values('Corr_Risk', ascending=False)
    
    print("\n=== TOP PREDICTORS (Risk of Anomaly) ===")
    print(res_df.sort_values('Corr_Risk', key=abs, ascending=False).head(15))
    
    print("\n=== TOP MEAN REVERSION CANDIDATES ===")
    print(res_df.sort_values('Corr_Reversion', ascending=True).head(10))
    
    res_df.to_csv('docs/micro_basket_signals_v2.csv', index=False)

if __name__ == "__main__":
    analyze_micro_basket()
