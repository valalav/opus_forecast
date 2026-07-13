import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

class SirenaMicroRegions:
    """
    Анализ опережающих регионов-предикторов для конкретных товаров.
    """
    
    def __init__(self):
        self.data = None
        self.regions_map = {}
        self.items_map = {
            10: 'Meat',
            1100: 'Milk',
            21: 'FruitsVeg',
            42: 'Fuel',
            1700: 'Sweets',
            4700: 'Shoes',
            9400: 'Utilities',
            7400: 'Construction'
        }
        
    def load_data(self):
        regs = pd.read_csv('data/regions_names.csv')
        self.regions_map = dict(zip(regs['Region_code'], regs['Region_names']))
        
        df = pd.read_csv('data/all_regions_micro.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df['Item'] = df['Item_code'].map(self.items_map)
        self.data = df
        
    def find_micro_predictors(self, target_region=7, start_date=None):
        """Find best leading region for each item."""
        summary = []
        
        for item_code, item_name in self.items_map.items():
            subset = self.data[self.data['Item'] == item_name]
            
            if start_date:
                subset = subset[subset['Date'] >= pd.Timestamp(start_date)]
                
            if subset.empty: continue
            
            pivot = subset.pivot_table(index='Date', columns='Region_code', values='MoM')
            
            if target_region not in pivot.columns: continue
            target_series = pivot[target_region]
            
            best_corr = -1
            best_reg = None
            
            # Scan all regions
            for col in pivot.columns:
                if col == target_region: continue
                
                # Check Lag 1
                other_series = pivot[col].shift(1)
                common = pd.concat([target_series, other_series], axis=1).dropna()
                
                min_len = 12 if start_date else 24
                if len(common) > min_len:
                    corr = common.corr().iloc[0, 1]
                    if corr > best_corr:
                        best_corr = corr
                        best_reg = col
            
            if best_reg is not None:
                summary.append({
                    'Товар': item_name,
                    'Регион-Лидер': self.regions_map.get(best_reg, str(best_reg)),
                    'Корреляция (Лаг 1)': best_corr
                })
                
        return pd.DataFrame(summary).sort_values('Корреляция (Лаг 1)', ascending=False)

if __name__ == "__main__":
    sm = SirenaMicroRegions()
    sm.load_data()
    print("All time:")
    print(sm.find_micro_predictors(7))
    print("\nSince 2023:")
    print(sm.find_micro_predictors(7, start_date='2023-01-01'))
