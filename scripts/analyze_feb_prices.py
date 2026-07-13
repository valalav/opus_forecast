import pandas as pd
import numpy as np
import datetime

# Target items (Volatile & High Weight)
TARGET_ITEMS = [
    "Огурцы свежие, кг",
    "Помидоры свежие, кг",
    "Картофель, кг",
    "Яйца куриные, 10 шт.",
    "Сахар-песок, кг",
    "Куры охлажденные и мороженые, кг",
    "Бананы, кг", # Check availability
    "Лук репчатый, кг",
    "Морковь, кг"
]

def load_data():
    # 1. Weekly Prices (New Source)
    # Format: Date, Code, Name, Price, PrevPrice, Change
    # Sample: 2026-01-26,9796,"Name",2749.31,2749.31,0.0
    
    # Read with flexible engine
    # Use low_memory=False to avoid mixed type warnings initially, but we handle conversion below
    try:
        df_w = pd.read_csv('data/kbr_weekly_prices_2008_2026.csv', header=None, sep=',', on_bad_lines='skip')
    except Exception as e:
        # Fallback if comma fails (unlikely given tail, but safe)
        df_w = pd.read_csv('data/kbr_weekly_prices_2008_2026.csv', header=None, sep=';', on_bad_lines='skip')
    
    # Assign columns based on inspection
    # Ensure we have enough columns
    if len(df_w.columns) >= 6:
        df_w = df_w.iloc[:, :6]
    df_w.columns = ['Date', 'Code', 'Item', 'Price', 'PrevPrice', 'Change']
    
    # Parse Date
    df_w['Date'] = pd.to_datetime(df_w['Date'], errors='coerce')
    df_w['Year'] = df_w['Date'].dt.year
    df_w['WeekNum'] = df_w['Date'].dt.isocalendar().week
    
    # Force Numeric Price
    df_w['Price'] = pd.to_numeric(df_w['Price'], errors='coerce')
    
    # Drop invalid rows
    df_w = df_w.dropna(subset=['Date', 'Price'])
    
    # Normalize Item names
    df_w['Item'] = df_w['Item'].astype(str).str.strip()
    
    # Filter targets
    # Note: Target names must match exactly. Let's check for partial matches if needed.
    # The new file might have slightly different naming (e.g. "Огурцы свежие" vs "Огурцы свежие, кг")
    # We will filter by partial string match to be safe
    
    mask = df_w['Item'].apply(lambda x: any(t.split(',')[0] in x for t in TARGET_ITEMS))
    df_w = df_w[mask]
    
    return df_w

def analyze_prices(df):
    results = {}
    
    # Ensure targets are found
    # Map input target names to actual names in DF
    found_items = df['Item'].unique()
    
    for target in TARGET_ITEMS:
        # Fuzzy match target to available items
        match = next((x for x in found_items if target.split(',')[0] in x), None)
        if not match:
            print(f"Warning: {target} not found in data")
            continue
            
        item_data = df[df['Item'] == match].sort_values('Date')
        
        # 1. Historical Feb 2024 & 2025
        # Feb is month 2
        p_2024 = item_data[(item_data['Date'].dt.year == 2024) & (item_data['Date'].dt.month == 2)]['Price'].mean()
        p_2025 = item_data[(item_data['Date'].dt.year == 2025) & (item_data['Date'].dt.month == 2)]['Price'].mean()
        
        # 2. Spot Price (Latest Jan 2026)
        # Check specifically for Jan 2026 data
        jan_2026_data = item_data[(item_data['Date'].dt.year == 2026) & (item_data['Date'].dt.month == 1)]
        
        if not jan_2026_data.empty:
            last_entry = jan_2026_data.iloc[-1]
            p_last = last_entry['Price']
            last_date = last_entry['Date'].strftime('%Y-%m-%d')
        else:
            # Fallback to absolute last if Jan 2026 missing (shouldn't happen with new file)
            last_entry = item_data.iloc[-1]
            p_last = last_entry['Price']
            last_date = last_entry['Date'].strftime('%Y-%m-%d')
            
        # 3. Calculate YoY Trend (Feb-to-Feb potential)
        trend_24_25 = (p_2025 / p_2024 - 1) * 100 if pd.notna(p_2024) and p_2024 != 0 else 0
        
        # 4. Model Feb 2026
        # Seasonality: Factor = Feb 2025 / Jan 2025 Mean
        jan_2025 = item_data[(item_data['Date'].dt.year == 2025) & (item_data['Date'].dt.month == 1)]['Price'].mean()
        feb_mult = p_2025 / jan_2025 if pd.notna(jan_2025) and jan_2025 != 0 else 1.02
        
        p_feb_2026_est = p_last * feb_mult
        
        results[target] = {
            'Feb_2024': p_2024,
            'Feb_2025': p_2025,
            'Trend_YoY': trend_24_25,
            'Spot_Price': p_last,
            'Spot_Date': last_date,
            'Est_Feb_2026': p_feb_2026_est,
            'Seasonal_Mult': feb_mult
        }
        
    return results

def print_report(results):
    print("Item | Feb 2024 | Feb 2025 | YoY% | Spot (Date) | Est Feb 2026 (Corridor)")
    print("-" * 80)
    for item, data in results.items():
        if np.isnan(data['Feb_2024']): continue
        
        spot_str = f"{data['Spot_Price']:.1f} ({data['Spot_Date']})"
        est_str = f"{data['Est_Feb_2026']:.1f}"
        
        print(f"{item[:20]:<20} | {data['Feb_2024']:<8.1f} | {data['Feb_2025']:<8.1f} | {data['Trend_YoY']:>5.1f}% | {spot_str:<15} | {est_str}")

if __name__ == "__main__":
    df = load_data()
    res = analyze_prices(df)
    print_report(res)
