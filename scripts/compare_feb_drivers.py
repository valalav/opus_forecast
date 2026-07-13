import pandas as pd
import numpy as np

TARGET_ITEMS = [
    "Огурцы свежие, кг",
    "Помидоры свежие, кг",
    "Картофель, кг",
    "Яйца куриные, 10 шт.",
    "Лук репчатый, кг",
    "Бананы, кг"
]

def analyze_drivers():
    # Read New Source
    try:
        df = pd.read_csv('data/kbr_weekly_prices_2008_2026.csv', header=None, sep=',', on_bad_lines='skip')
    except:
        df = pd.read_csv('data/kbr_weekly_prices_2008_2026.csv', header=None, sep=';', on_bad_lines='skip')
        
    if len(df.columns) >= 6:
        df = df.iloc[:, :6]
    df.columns = ['Date', 'Code', 'Item', 'Price', 'PrevPrice', 'Change']
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df.dropna(subset=['Price', 'Date'])
    df['Item'] = df['Item'].astype(str).str.strip()
    
    # Filter for rough matching
    mask = df['Item'].apply(lambda x: any(t.split(',')[0] in x for t in TARGET_ITEMS))
    filtered = df[mask].copy()
    
    # Analyze Feb 2025 dynamics (Week-on-Week)
    # Feb 2025 weeks: ~2025-02-03 to 2025-02-24
    feb_2025 = filtered[(filtered['Date'] >= '2025-01-27') & (filtered['Date'] <= '2025-02-24')].sort_values(['Item', 'Date'])
    
    print("=== FEB 2025 DRIVERS (Weekly Growth) ===")
    for item in TARGET_ITEMS:
        item_match = next((x for x in filtered['Item'].unique() if item.split(',')[0] in x), None)
        if not item_match: continue
        
        subset = feb_2025[feb_2025['Item'] == item_match]
        if subset.empty: continue
        
        # Calculate total growth in Feb 2025
        start_p = subset.iloc[0]['Price']
        end_p = subset.iloc[-1]['Price']
        growth = (end_p / start_p - 1) * 100
        print(f"{item_match[:20]:<20} | Start: {start_p:.1f} | End: {end_p:.1f} | Growth: {growth:+.1f}%")

    print("\n=== JAN 2026 CONTEXT (Weekly Growth) ===")
    # Jan 2026 recent dynamics
    jan_2026 = filtered[(filtered['Date'] >= '2025-12-29') & (filtered['Date'] <= '2026-01-26')].sort_values(['Item', 'Date'])
    
    for item in TARGET_ITEMS:
        item_match = next((x for x in filtered['Item'].unique() if item.split(',')[0] in x), None)
        if not item_match: continue
        
        subset = jan_2026[jan_2026['Item'] == item_match]
        if subset.empty: continue
        
        start_p = subset.iloc[0]['Price']
        end_p = subset.iloc[-1]['Price']
        growth = (end_p / start_p - 1) * 100
        print(f"{item_match[:20]:<20} | Start: {start_p:.1f} | End: {end_p:.1f} | Growth: {growth:+.1f}%")

if __name__ == "__main__":
    analyze_drivers()
