import pandas as pd
import numpy as np

def analyze_pareto():
    # 1. Load Data
    
    # Weights Source: micro_sprav.csv (Verified weights)
    # Format: Item_code;Tv;Component;Subcomponent;Weight
    # Weight uses comma decimal
    try:
        weights = pd.read_csv('data/micro_sprav.csv', sep=';', decimal=',')
        # Clean names
        weights['Name_Clean'] = weights['Товар'].str.lower().str.strip()
    except Exception as e:
        print(f"Error loading micro_sprav: {e}")
        return

    # Price Source: kbr_weekly_prices
    try:
        prices = pd.read_csv('data/kbr_weekly_prices_2008_2026.csv', header=None, sep=',', on_bad_lines='skip')
        if len(prices.columns) < 4:
             prices = pd.read_csv('data/kbr_weekly_prices_2008_2026.csv', header=None, sep=';', on_bad_lines='skip')
        prices.columns = ['Date', 'Code', 'Item', 'Price', 'PrevPrice', 'Change']
    except Exception as e:
        print(f"Error loading prices: {e}")
        return

    # 2. Calculate Growth (2024-2025)
    prices['Date'] = pd.to_datetime(prices['Date'], errors='coerce')
    prices['Price'] = pd.to_numeric(prices['Price'], errors='coerce')
    period_prices = prices[(prices['Date'] >= '2024-01-01') & (prices['Date'] <= '2025-12-31')].dropna(subset=['Price'])
    
    item_stats = []
    # Using Item Name from weekly prices as key
    for name in period_prices['Item'].unique():
        subset = period_prices[period_prices['Item'] == name].sort_values('Date')
        if subset.empty: continue
        
        start_p = subset.iloc[0]['Price']
        end_p = subset.iloc[-1]['Price']
        if start_p == 0: continue
        
        growth = (end_p / start_p) - 1
        item_stats.append({'Item': name, 'Growth': growth, 'Item_Clean': name.lower().strip()})
        
    growth_df = pd.DataFrame(item_stats)
    
    # 3. Fuzzy Merge on Names
    # We need to map `growth_df['Item_Clean']` to `weights['Name_Clean']`
    # Exact match first
    merged = pd.merge(growth_df, weights, left_on='Item_Clean', right_on='Name_Clean', how='left')
    
    # For unmapped, try fuzzy or partial
    # (Simplified for script: assume high overlap on key items)
    
    # Fill missing weights with 0 (or handle them)
    # The 'Weight' column in micro_sprav is the share (0.0158 = 1.58%)
    merged = merged.dropna(subset=['Weight'])
    
    # Calculate Impact
    # Contribution to Inflation Index ~ Weight * Growth
    merged['Weighted_Impact'] = merged['Weight'] * merged['Growth']
    
    # Sort
    merged = merged.sort_values('Weighted_Impact', ascending=False)
    
    # Cumulative
    total_impact = merged['Weighted_Impact'][merged['Weighted_Impact'] > 0].sum()
    merged['Pct_of_Total'] = (merged['Weighted_Impact'] / total_impact) * 100
    merged['Cumulative_Pct'] = merged['Pct_of_Total'].cumsum()
    
    # Filter
    pareto_set = merged[merged['Cumulative_Pct'] <= 85]
    
    print(f"=== PARETO 80% DRIVERS (2024-2025) ===")
    print(f"{'Item':<40} | {'Growth':<8} | {'Weight':<8} | {'Impact':<8} | {'Cumul%':<8}")
    print("-" * 85)
    
    for idx, row in pareto_set.iterrows():
        name = row['Item'][:40]
        growth = f"{row['Growth']*100:.1f}%"
        weight = f"{row['Weight']:.4f}"
        impact = f"{row['Pct_of_Total']:.1f}%"
        cumul = f"{row['Cumulative_Pct']:.1f}%"
        print(f"{name:<40} | {growth:<8} | {weight:<8} | {impact:<8} | {cumul:<8}")
    
    # Sort by Impact
    merged = merged.sort_values('Weighted_Impact', ascending=False)
    
    # Calculate Cumulative %
    total_impact = merged['Weighted_Impact'].sum()
    merged['Pct_of_Total'] = (merged['Weighted_Impact'] / total_impact) * 100
    merged['Cumulative_Pct'] = merged['Pct_of_Total'].cumsum()
    
    # Filter Top 80%
    pareto_set = merged[merged['Cumulative_Pct'] <= 85] # slightly over 80 to capture the boundary item
    
    print(f"=== PARETO 80% DRIVERS (2024-2025) ===")
    print(f"{'Item':<40} | {'Growth':<8} | {'Weight':<8} | {'Impact':<8} | {'Cumul%':<8}")
    print("-" * 80)
    
    for idx, row in pareto_set.iterrows():
        name = row['Item'][:40]
        growth = f"{row['Growth']*100:.1f}%"
        weight = f"{row['Weight_gross']:.4f}"
        impact = f"{row['Pct_of_Total']:.1f}%"
        cumul = f"{row['Cumulative_Pct']:.1f}%"
        print(f"{name:<40} | {growth:<8} | {weight:<8} | {impact:<8} | {cumul:<8}")

if __name__ == "__main__":
    analyze_pareto()
