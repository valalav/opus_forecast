import pandas as pd

def check_dates():
    try:
        df = pd.read_csv('data/kbr_full_monthly.csv')
        print("Loaded data/kbr_full_monthly.csv")
        print(f"Total rows: {len(df)}")
        
        if 'Date' in df.columns:
            # Check unique dates
            dates = df['Date'].unique()
            print(f"Unique dates count: {len(dates)}")
            print(f"First 5 dates: {sorted(dates)[:5]}")
            print(f"Last 5 dates: {sorted(dates)[-5:]}")
            
            # Check a specific year (2024)
            d24 = [d for d in dates if '2024' in str(d)]
            print(f"Dates in 2024: {len(d24)}")
            print(f"Sample 2024: {sorted(d24)}")
            
        else:
            print("No 'Date' column found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_dates()
