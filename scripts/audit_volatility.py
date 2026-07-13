import pandas as pd
import numpy as np

def audit_volatility():
    print("Loading Aggregate Inflation Data...")
    try:
        # Load verified source
        df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')
        
        # Ensure MoM is numeric
        if df_raw['MoM'].dtype == object:
             df_raw['MoM'] = df_raw['MoM'].astype(str).str.replace(',', '.')
        df_raw['MoM'] = pd.to_numeric(df_raw['MoM'], errors='coerce')
        
        # Drop rows with NaN MoM
        df_raw = df_raw.dropna(subset=['MoM'])
        
        # Check structure
        if 'Товар' in df_raw.columns:
            print("Detected Long Format. Pivoting...")
            # Pivot
            df = df_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='mean')
            df.index = pd.to_datetime(df.index, dayfirst=True)
            df = df.sort_index()
            
            # Identify columns
            cols_map = {
                'Все товары и услуги': 'CPI',
                'Продовольственные товары': 'Food',
                'Непродовольственные товары': 'NonFood',
                'Услуги': 'Services'
            }
            df = df.rename(columns=cols_map)
        else:
            print("Unknown format.")
            return

        # Calculate MoM %
        mean_val = df['CPI'].mean()
        if mean_val > 50:
            df['MoM'] = df['CPI'] - 100.0
        else:
            df['MoM'] = df['CPI']
            
        print(f"Data Loaded: {len(df)} months ({df.index.min().date()} to {df.index.max().date()})")
        
        # 1. Annual Volatility (Std Dev)
        print("\n=== ANNUAL VOLATILITY (Sigma) ===")
        annual_stats = df.resample('Y')['MoM'].agg(['std', 'mean', 'count'])
        annual_stats['std'] = annual_stats['std'].round(3)
        annual_stats['mean'] = annual_stats['mean'].round(3)
        
        # Display last 5 years
        print(annual_stats.tail(5))
        
        # Compare 2024 vs 2025
        sigma_24 = annual_stats.loc['2024-12-31', 'std'] if '2024-12-31' in annual_stats.index else np.nan
        sigma_25 = annual_stats.loc['2025-12-31', 'std'] if '2025-12-31' in annual_stats.index else np.nan
        
        if pd.notna(sigma_24) and pd.notna(sigma_25):
            growth = (sigma_25 / sigma_24 - 1) * 100
            print(f"\nGrowth 2025 vs 2024: {growth:+.1f}%")
            print(f"Sigma 2024: {sigma_24}")
            print(f"Sigma 2025: {sigma_25}")
        
        # 2. Quarterly Volatility
        print("\n=== QUARTERLY VOLATILITY ===")
        q_stats = df.resample('Q')['MoM'].std()
        
        # Check Q3 2025 vs history
        q3_25 = q_stats.get('2025-09-30')
        if pd.notna(q3_25):
            rank = (q_stats >= q3_25).sum()
            total_q = len(q_stats)
            print(f"Q3 2025 Sigma: {q3_25:.3f}")
            print(f"Rank: {rank} of {total_q} (1 = Most Volatile)")
            
            # Top 5 Volatile Quarters
            print("\nTop 5 Most Volatile Quarters in History:")
            print(q_stats.sort_values(ascending=False).head(5))

        # 3. Anomaly Count (Deviation from Seasonality)
        print("\n=== ANOMALY COUNT (Deviation > 0.5 sigma) ===")
        baseline = df[df.index.year < 2024]
        seasonal_means = baseline.groupby(baseline.index.month)['MoM'].mean()
        
        # Check 2025
        curr_year = df[df.index.year == 2025].copy()
        curr_year['Norm'] = curr_year.index.month.map(seasonal_means)
        curr_year['Diff'] = curr_year['MoM'] - curr_year['Norm']
        
        anomalies_fixed = curr_year[curr_year['Diff'].abs() > 0.5]
        
        print(f"Anomalies (Diff > 0.5 p.p.): {len(anomalies_fixed)} of {len(curr_year)}")
        print(anomalies_fixed[['MoM', 'Norm', 'Diff']])
        
        # 4. Check Food Volatility (True Check)
        if 'Food' in df.columns:
             df['Food_MoM'] = df['Food'] - 100.0
             food_std_24 = df.loc['2024', 'Food_MoM'].std()
             food_std_25 = df.loc['2025', 'Food_MoM'].std()
             print(f"\nFood Volatility 2024: {food_std_24:.3f}")
             print(f"Food Volatility 2025: {food_std_25:.3f}")
             if pd.notna(food_std_24) and pd.notna(food_std_25):
                 print(f"Food Volatility Growth: {(food_std_25/food_std_24 - 1)*100:+.1f}%")
             else:
                 print("Food Volatility Data incomplete.")

        # Save results
        with open('docs/audit_volatility_results.txt', 'w') as f:
            f.write(f"Sigma 2024: {sigma_24}\n")
            f.write(f"Sigma 2025: {sigma_25}\n")
            f.write(f"Growth: {growth:.1f}%\n")
            f.write(f"Anomalies: {len(anomalies_fixed)}\n")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    audit_volatility()
