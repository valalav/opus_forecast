import pandas as pd
import numpy as np

def analyze_weights_pareto():
    print("Loading Weights...")
    try:
        # Load weights for Region 7 (KBR)
        chunks = []
        for chunk in pd.read_csv('data/access_weights.csv', chunksize=50000):
            filtered = chunk[chunk['Region_code'] == 7]
            if not filtered.empty:
                chunks.append(filtered)
        
        if not chunks:
            print("No weights found for Region 7.")
            return

        weights = pd.concat(chunks)
        weights['Date'] = pd.to_datetime(weights['Day'], format='%d/%m/%y %H:%M:%S')
        weights['Year'] = weights['Date'].dt.year
        
        # Use latest full year (e.g. 2024 or 2025)
        # Check max year
        max_year = weights['Year'].max()
        print(f"Using weights for year: {max_year}")
        
        w_curr = weights[weights['Year'] == max_year].copy()
        
        # Filter duplicates (keep unique Item_code, taking the first occurrence if duplicates exist)
        # Ideally, weights should be unique per Item_code for a given year/region.
        w_curr = w_curr.drop_duplicates(subset=['Item_code'])
        
        # Load Names
        names = pd.read_csv('data/items_names.csv')
        names_map = names.drop_duplicates('Item_code').set_index('Item_code')['Item_name'].to_dict()
        
        # Map Names
        w_curr['Name'] = w_curr['Item_code'].map(names_map)
        
        # Filter out aggregates?
        # Aggregates usually have weight ~1.0 or large sums.
        # But 'Weight_vertical' usually sums to 1 (or 1000).
        # Let's check the sum.
        total_w = w_curr['Weight_vertical'].sum()
        print(f"Total Weight Sum (Raw): {total_w}")
        
        # If sum is huge (e.g. includes sub-aggregates + items), we need to filter.
        # Typically, we want to filter out aggregates to analyze the 'micro' tail.
        # Heuristic: Sort by weight. Level 1 (All items) = 1. Level 2 (Food) ~ 0.4.
        # We want to exclude these top aggregates to analyze the 'micro' tail.
        
        # Exclude items with weight > 0.1 (10%) as they are likely aggregates
        w_micro = w_curr[w_curr['Weight_vertical'] < 0.1].copy()
        print(f"Filtered out aggregates > 10%. Remaining items: {len(w_micro)}")
        
        # Re-normalize weights to sum to 100% of the micro basket?
        # Or just look at their contribution to Total CPI.
        
        # Sort by Weight
        w_micro = w_micro.sort_values('Weight_vertical', ascending=False)
        
        # Calculate Cumulative Weight based on TOTAL original sum (approx 1.0)
        # To see how much of the Total CPI they explain.
        w_micro['Share_in_CPI'] = w_micro['Weight_vertical'] # Assuming sum is near 1
        w_micro['Cum_Share'] = w_micro['Share_in_CPI'].cumsum()
        
        print("\n=== PARETO ANALYSIS (Concentration) ===")
        
        # Find cutoffs
        top50_share = w_micro.head(50)['Share_in_CPI'].sum()
        top100_share = w_micro.head(100)['Share_in_CPI'].sum()
        top200_share = w_micro.head(200)['Share_in_CPI'].sum()
        
        count_50pct = len(w_micro[w_micro['Cum_Share'] <= 0.5])
        count_80pct = len(w_micro[w_micro['Cum_Share'] <= 0.8])
        
        print(f"Top 50 items control:  {top50_share:.1%} of inflation.")
        print(f"Top 100 items control: {top100_share:.1%} of inflation.")
        print(f"Top 200 items control: {top200_share:.1%} of inflation.")
        print("-" * 30)
        print(f"Items needed to explain 50% of CPI: {count_50pct}")
        print(f"Items needed to explain 80% of CPI: {count_80pct}")
        
        print("\n=== TOP 20 HEAVYWEIGHTS (Micro) ===")
        print(w_micro[['Name', 'Weight_vertical', 'Cum_Share']].head(20))
        
        # Save Top List
        w_micro.to_csv('docs/pareto_weights_analysis.csv', index=False)
        print("\nSaved analysis to docs/pareto_weights_analysis.csv")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_weights_pareto()
