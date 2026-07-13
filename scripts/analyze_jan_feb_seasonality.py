import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# Setup directories
CHARTS_DIR = Path('assets/charts')
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',', parse_dates=['Date'], dayfirst=True, index_col='Date')

if 'mom' in df.columns:
    df['inflation'] = df['mom'] - 100
else:
    raise ValueError(f"Column 'mom' not found. Available: {df.columns.tolist()}")

df = df.dropna(subset=['inflation'])
df = df.sort_index()

# ---------------------------------------------------------
# 1. Monthly Statistics (Jan & Feb)
# ---------------------------------------------------------
print("\n" + "="*50)
print("SEASONALITY ANALYSIS: JANUARY & FEBRUARY")
print("="*50)

for month, name in [(1, 'January'), (2, 'February')]:
    m_data = df[df.index.month == month].copy()
    m_data['year'] = m_data.index.year
    
    # Identify outliers (Z-score > 2)
    mean = m_data['inflation'].mean()
    std = m_data['inflation'].std()
    m_data['z_score'] = (m_data['inflation'] - mean) / std
    m_data['is_outlier'] = m_data['z_score'].abs() > 2.0
    
    clean_data = m_data[~m_data['is_outlier']]
    
    print(f"\n[{name.upper()}]")
    print(f"Count: {len(m_data)} years ({m_data.index.min().year}-{m_data.index.max().year})")
    print(f"Mean (All):     {mean:.2f}%")
    print(f"Median (All):   {m_data['inflation'].median():.2f}%")
    print(f"robust Mean (Trimmed): {clean_data['inflation'].mean():.2f}% (excl. outliers)")
    print(f"Min: {m_data['inflation'].min():.2f}% ({m_data.loc[m_data['inflation'].idxmin(), 'year']})")
    print(f"Max: {m_data['inflation'].max():.2f}% ({m_data.loc[m_data['inflation'].idxmin(), 'year']})") # Fix logic bug: idxmax needed
    print(f"Outliers (>2std): {m_data[m_data['is_outlier']]['year'].tolist()}")
    
    # Last 5 years
    last_5 = m_data.sort_index().tail(5)
    print("\nLast 5 years:")
    for date, row in last_5.iterrows():
        print(f"  {date.year}: {row['inflation']:.2f}%")

# ---------------------------------------------------------
# 2. Rebound Analysis (Nov+Dec vs Jan)
# ---------------------------------------------------------
print("\n" + "="*50)
print("REBOUND ANALYSIS (Nov/Dec impact on Jan)")
print("="*50)

rebound_data = []
# Iterate years to find pairs (Nov/Dec T-1) -> (Jan T)
years = df.index.year.unique().sort_values()
for year in years:
    # Need Nov, Dec of year-1 and Jan of year
    try:
        prev_nov = df.loc[f"{year-1}-11", 'inflation'].values[0]
        prev_dec = df.loc[f"{year-1}-12", 'inflation'].values[0]
        curr_jan = df.loc[f"{year}-01", 'inflation'].values[0]
        
        sum_prev_end = prev_nov + prev_dec
        rebound_data.append({
            'year': year,
            'prev_sum': sum_prev_end,
            'jan': curr_jan
        })
    except:
        continue

rdf = pd.DataFrame(rebound_data)
corr = rdf['prev_sum'].corr(rdf['jan'])
print(f"\nCorrelation (Nov+Dec) vs Jan: {corr:.2f}")

# Check specific case: Low Nov+Dec
threshold_low = rdf['prev_sum'].quantile(0.25)
low_prev_years = rdf[rdf['prev_sum'] <= threshold_low]
print(f"\nYears with LOW Nov+Dec (<= {threshold_low:.2f}%):")
print(low_prev_years[['year', 'prev_sum', 'jan']].to_string(index=False))
print(f"Average Jan in these years: {low_prev_years['jan'].mean():.2f}%")

# Check 2025 specifically (using provided context)
# The file likely goes up to Dec 2025 or Jan 2026?
# precompute_forecasts said "Last data point: 2025-12"
# So we can check Nov 2025 + Dec 2025
try:
    nov_25 = df.loc['2025-11', 'inflation'].values[0]
    dec_25 = df.loc['2025-12', 'inflation'].values[0]
    sum_25 = nov_25 + dec_25
    print(f"\nNov+Dec 2025 Sum: {sum_25:.2f}%")
    
    # Compare to low years
    if sum_25 <= threshold_low:
        print("-> 2025 ended with LOW inflation accumulation.")
    else:
        print("-> 2025 ended with Normal/High inflation accumulation.")
        
except Exception as e:
    print(f"Could not check 2025: {e}")

# Save plot for user
fig = go.Figure()
fig.add_trace(go.Scatter(x=rdf['prev_sum'], y=rdf['jan'], mode='markers+text', 
                         text=rdf['year'], textposition='top center',
                         name='Years'))
fig.update_layout(title="Correlation: (Nov+Dec) Inflation vs January Inflation",
                  xaxis_title="Sum of prev Nov+Dec (%)",
                  yaxis_title="January Inflation (%)")
fig.write_html(CHARTS_DIR / 'seasonality_jan_feb_compare.html')
print(f"\nChart saved to {CHARTS_DIR / 'seasonality_jan_feb_compare.html'}")
