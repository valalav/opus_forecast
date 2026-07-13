import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class SirenaRegions:
    """
    Анализ межрегиональных связей инфляции.
    """
    
    def __init__(self):
        self.data = None
        self.regions_map = {}
        self.items_map = {
            1: 'CPI',
            2: 'NonFood',
            3: 'Food',
            4: 'Services',
            33: 'FruitsVeg'
        }
        
    def load_data(self):
        # 1. Load Regions
        regs = pd.read_csv('data/regions_names.csv')
        self.regions_map = dict(zip(regs['Region_code'], regs['Region_names']))
        
        # 2. Load Indices
        df = pd.read_csv('data/all_regions_indices.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df['Item'] = df['Item_code'].map(self.items_map)
        
        # Filter out unknown items
        df = df.dropna(subset=['Item'])
        self.data = df
        
    def get_correlation_matrix(self, target_region_code=7, item='CPI', lag=0):
        """
        Calculate correlation of Target Region vs All Other Regions.
        If lag > 0: Corr(Target(t), Other(t-lag)).
        """
        # Pivot: Index=Date, Columns=Region_Name, Values=MoM
        subset = self.data[self.data['Item'] == item]
        
        pivot = subset.pivot_table(index='Date', columns='Region_code', values='MoM')
        
        if target_region_code not in pivot.columns:
            print(f"Region {target_region_code} not found in data.")
            return None
            
        target_series = pivot[target_region_code]
        
        results = []
        for col in pivot.columns:
            if col == target_region_code: continue
            
            other_series = pivot[col]
            if lag > 0:
                other_series = other_series.shift(lag)
                
            # Correlate
            # Use common index
            common = pd.concat([target_series, other_series], axis=1).dropna()
            if len(common) < 24: continue # Need minimum history
            
            corr = common.corr().iloc[0, 1]
            
            results.append({
                'Region_Code': col,
                'Region_Name': self.regions_map.get(col, f"Unknown {col}"),
                'Correlation': corr,
                'Lag': lag,
                'Points': len(common)
            })
            
        return pd.DataFrame(results).sort_values('Correlation', ascending=False)

    def find_best_predictors(self, target_region=7):
        print(f"Поиск лучших предикторов для региона {self.regions_map.get(target_region, target_region)}...")
        
        summary = []
        
        for item in self.items_map.values():
            # Check Lag 0 (Sync) and Lag 1 (Leading)
            corr0 = self.get_correlation_matrix(target_region, item, lag=0)
            corr1 = self.get_correlation_matrix(target_region, item, lag=1)
            
            if corr0 is not None and not corr0.empty:
                best0 = corr0.iloc[0]
                summary.append({
                    'Item': item,
                    'Type': 'Simultaneous',
                    'Region': best0['Region_Name'],
                    'Corr': best0['Correlation']
                })
                
            if corr1 is not None and not corr1.empty:
                best1 = corr1.iloc[0]
                summary.append({
                    'Item': item,
                    'Type': 'Leading (1 mon)',
                    'Region': best1['Region_Name'],
                    'Corr': best1['Correlation']
                })
                
        return pd.DataFrame(summary)

if __name__ == "__main__":
    sr = SirenaRegions()
    sr.load_data()
    res = sr.find_best_predictors(7) # KBR
    print(res)
    
    # Detailed look at CPI leaders
    print("\nТоп-5 опережающих регионов (ИПЦ):")
    cpi_lag = sr.get_correlation_matrix(7, 'CPI', lag=1)
    print(cpi_lag.head(5))
    cpi_lag.to_csv('regional_correlations_cpi_lag1.csv', index=False)
