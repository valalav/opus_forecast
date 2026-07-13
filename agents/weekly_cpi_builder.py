import pandas as pd
import numpy as np
import datetime

def load_and_prep_weights(weights_path):
    print(f"Loading weights from {weights_path}...")
    df_w = pd.read_csv(weights_path)
    
    # Parse dates
    df_w['Date'] = pd.to_datetime(df_w['Day'], errors='coerce')
    df_w['Year'] = df_w['Date'].dt.year
    
    # We need 'Item_code' and 'Weight_gross' (or vertical?)
    # Usually Rosstat weights are per 1000 or fractions summing to 1.
    # Let's check sums per year.
    sums = df_w.groupby('Year')['Weight_gross'].sum()
    print("Weight sums per year (should be close to 1 or 1000):")
    print(sums.head())
    
    # Keep only relevant columns
    # We will join on Item_code and Year
    weights = df_w[['Year', 'Item_code', 'Weight_gross']].copy()
    weights.rename(columns={'Item_code': 'product_code', 'Weight_gross': 'weight'}, inplace=True)
    
    # Ensure int types for merge
    weights = weights.dropna(subset=['product_code'])
    weights['product_code'] = weights['product_code'].astype(int)
    
    # Group by (Year, product_code) to handle duplicates
    weights = weights.groupby(['Year', 'product_code'])['weight'].mean().reset_index()
    
    # Expand weights to cover full range 2008-2026?
    # Strategy: 
    # - Pivot to wide form (index=product_code, columns=Year)
    # - Reindex columns to include 2008-2026
    # - Ffill/Bfill
    # - Melt back
    
    print("Expanding weights coverage...")
    pivot_w = weights.pivot(index='product_code', columns='Year', values='weight')
    
    # Add missing years
    all_years = range(2008, 2027)
    pivot_w = pivot_w.reindex(columns=all_years)
    
    # Fill gaps (forward fill from 2016->2026, backfill 2016->2008)
    pivot_w = pivot_w.ffill(axis=1).bfill(axis=1)
    
    # Melt back
    weights_expanded = pivot_w.reset_index().melt(id_vars='product_code', var_name='Year', value_name='weight')
    weights_expanded['Year'] = weights_expanded['Year'].astype(int)
    
    return weights_expanded

def build_weekly_cpi():
    PRICES_PATH = "data/kbr_weekly_prices_2008_2026.csv"
    WEIGHTS_PATH = "data/access_weights.csv"
    OFFICIAL_CPI_PATH = "data/infl_kbr.csv"
    OUTPUT_PATH = "data/kbr_weekly_cpi_2008_2026.csv"
    REPORT_PATH = "weekly_cpi_report.md"
    
    # 1. Load Prices
    print("Loading prices...")
    df_p = pd.read_csv(PRICES_PATH)
    df_p['date'] = pd.to_datetime(df_p['date'])
    df_p['year'] = df_p['date'].dt.year
    df_p['month'] = df_p['date'].dt.month
    
    # 2. Load Weights 
    df_w = load_and_prep_weights(WEIGHTS_PATH)
    
    # 3. Merge Weights
    # Weights are usually published at start of year for that year.
    # So 2020 prices use 2020 weights.
    print("Merging weights...")
    df_merged = pd.merge(df_p, df_w, left_on=['year', 'product_code'], right_on=['Year', 'product_code'], how='left')
    
    # Analyze coverage
    missing_weights = df_merged[df_merged['weight'].isna()]
    print(f"Rows with missing weights: {len(missing_weights)} / {len(df_merged)}")
    
    if len(missing_weights) > 0:
        print("Initial missing weights detected. Attempting forward/backward fill for weights...")
        # Sort by product and time to fill weights across years if missing
        df_merged.sort_values(['product_code', 'date'], inplace=True)
        df_merged['weight'] = df_merged.groupby('product_code')['weight'].ffill().bfill()
        
    # Check again
    df_clean = df_merged.dropna(subset=['weight', 'wow_growth'])
    print(f"Clean rows for calculation: {len(df_clean)}")
    
    # 4. Calculate Aggregate Weekly Inflation
    # Formula: Sum(w_i * growth_i) / Sum(w_i)
    # Note: wow_growth is in % (e.g. 0.5 for 0.5%).
    
    print("Calculating Weekly CPI...")
    weekly_cpi = df_clean.groupby('date').apply(
        lambda x: np.average(x['wow_growth'], weights=x['weight'])
    ).reset_index(name='weekly_inflation_pct')
    
    weekly_cpi['weekly_index'] = 1 + weekly_cpi['weekly_inflation_pct'] / 100
    
    # 5. Build Cumulative Index (Base = Start)
    weekly_cpi = weekly_cpi.sort_values('date')
    weekly_cpi['cumulative_cpi'] = weekly_cpi['weekly_index'].cumprod() * 100
    
    # Save Weekly CPI
    weekly_cpi.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH}")
    
    # 6. Validate Against Official Monthly
    print("Validating...")
    # Load with simple options first, then clean
    df_off = pd.read_csv(OFFICIAL_CPI_PATH, sep=';', decimal=',')
    
    # Clean MoM column - ensure it's numeric
    if df_off['MoM'].dtype == 'object':
        df_off['MoM'] = df_off['MoM'].astype(str).str.replace(',', '.', regex=False)
        
    df_off['MoM'] = pd.to_numeric(df_off['MoM'], errors='coerce')
    
    # Filter and setup index
    df_off = df_off[df_off['Товар'] == 'Все товары и услуги'].copy()
    
    # Normalize Date
    df_off['Date'] = pd.to_datetime(df_off['Date'], errors='coerce')
    if df_off['Date'].isna().any() and 'Day' in df_off.columns:
         df_off['Date'] = pd.to_datetime(df_off['Day'], format='%d.%m.%Y', dayfirst=True)
         
    df_off.set_index('Date', inplace=True)
    df_off.sort_index(inplace=True)
    
    # Aggregate Weekly to Monthly for comparison
    weekly_cpi.set_index('date', inplace=True)
    monthly_agg = weekly_cpi['weekly_index'].resample('MS').prod() * 100
    
    # Join
    comparison = pd.DataFrame({
        'My_Weekly_Agg': monthly_agg,
        'Official_MoM': df_off['MoM']
    }).dropna()
    
    # Correlation
    if len(comparison) > 0:
        corr = comparison.corr().iloc[0, 1]
        mae = np.mean(np.abs(comparison['My_Weekly_Agg'] - comparison['Official_MoM']))
        
        print(f"Correlation with Official CPI: {corr:.4f}")
        print(f"MAE: {mae:.4f}")
    else:
        print("No overlapping data for validation.")
        corr, mae = 0, 0
    
    # Generate Report
    with open(REPORT_PATH, 'w') as f:
        f.write("# Weekly CPI Construction Report\n\n")
        f.write(f"- **Correlation:** {corr:.4f}\n")
        f.write(f"- **MAE:** {mae:.4f} p.p.\n")
        f.write("\n## Comparison (Last 10 Months)\n")
        f.write("\n## Comparison (Last 10 Months)\n")
        # f.write(comparison.tail(10).to_markdown())
        f.write(comparison.tail(10).to_string())
        f.write("\n\n## Analysis\n")
        if corr > 0.8:
            f.write("✅ **High Correlation**: The derived weekly index is a strong proxy for official inflation.\n")
        elif corr > 0.5:
            f.write("⚠️ **Moderate Correlation**: Trend is captured but weekly volatility might introduce noise.\n")
        else:
            f.write("❌ **Low Correlation**: Weighting or data coverage issues likely.\n")

if __name__ == "__main__":
    build_weekly_cpi()
