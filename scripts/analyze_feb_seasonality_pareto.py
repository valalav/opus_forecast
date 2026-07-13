import pandas as pd
import numpy as np

PARETO_ITEMS = [
    "Говядина (кроме бескостного мяса), кг",
    "Масло сливочное, кг",
    "Картофель, кг",
    "Бензин автомобильный марки АИ-95, л",
    "Бензин автомобильный марки АИ-92, л",
    "Сметана, кг",
    "Куры охлажденные и мороженые, кг",
    "Творог, кг"
]

def analyze_seasonality():
    try:
        # Load Weekly Data
        try:
            df = pd.read_csv('data/kbr_weekly_prices_2008_2026.csv', header=None, sep=',', on_bad_lines='skip')
        except:
            df = pd.read_csv('data/kbr_weekly_prices_2008_2026.csv', header=None, sep=';', on_bad_lines='skip')
            
        if len(df.columns) >= 6:
            df = df.iloc[:, :6]
        df.columns = ['Date', 'Code', 'Item', 'Price', 'PrevPrice', 'Change']
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Item'] = df['Item'].astype(str).str.strip()
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Date', 'Price'])
        
        print("=== FEB SEASONALITY (Historical Average MoM) ===")
        print(f"{'Item':<35} | {'Avg Feb Chg':<12} | {'Risk Level':<10}")
        print("-" * 65)
        
        for target in PARETO_ITEMS:
            # Fuzzy match
            match = next((x for x in df['Item'].unique() if target.split(',')[0] in x), None)
            if not match: continue
            
            subset = df[df['Item'] == match].sort_values('Date')
            
            # Calculate Feb changes for 2020-2025
            feb_changes = []
            for year in range(2020, 2026):
                # Start of Feb
                start_date = f"{year}-02-01"
                end_date = f"{year}-02-28"
                if year % 4 == 0: end_date = f"{year}-02-29"
                
                # Get price closest to start and end
                month_data = subset[(subset['Date'] >= f"{year}-01-25") & (subset['Date'] <= end_date)]
                if month_data.empty: continue
                
                # Simple approximation: Last of Feb / Last of Jan
                # Actually, let's just take the month's data points
                jan_end = subset[subset['Date'] <= f"{year}-02-01"].tail(1)
                feb_end = subset[subset['Date'] <= end_date].tail(1)
                
                if not jan_end.empty and not feb_end.empty:
                    p1 = jan_end.iloc[0]['Price']
                    p2 = feb_end.iloc[0]['Price']
                    if p1 > 0:
                        chg = (p2/p1 - 1) * 100
                        feb_changes.append(chg)
            
            if feb_changes:
                avg_chg = np.mean(feb_changes)
                risk = "HIGH" if avg_chg > 1.0 else "MED" if avg_chg > 0.5 else "LOW"
                print(f"{match[:35]:<35} | {avg_chg:>+9.2f}% | {risk:<10}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    analyze_seasonality()
