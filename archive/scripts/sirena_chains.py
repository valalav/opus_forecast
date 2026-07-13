import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class SirenaChains:
    """
    Анализ производственных цепочек (Cost-Push Effect).
    Проверяет гипотезу: Цена Сырья (t) -> Цена Продукта (t+k).
    """
    
    def __init__(self):
        self.df = None
        self.pairs = [
            # (Сырье_Code, Продукт_Code, Название)
            # Коды из items_names (примерные, уточним по данным)
            # 1100: Молоко -> 1200: Сыр (если есть)
            # 42: Топливо -> 9200: Транспорт (если есть) или 4: Услуги
            # 1600: Сахар -> 1700: Кондитерские
            # 10: Мясо -> 201: Колбаса? (Нужно проверить коды в all_regions_micro)
        ]
        
    def load_data(self):
        print("Загрузка данных КБР (Region 7)...")
        
        # Load extracted micro data
        try:
            df = pd.read_csv('data/all_regions_micro.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Filter KBR
            self.df = df[df['Region_code'] == 7].copy()
            
            # Check available items
            available_items = self.df['Item_code'].unique()
            print(f"Доступные товары: {available_items}")
            
            # Define pairs based on available items in micro data
            # Micro data has: {10, 1100, 21, 42, 1700, 4700, 9400, 7400}
            # 10: Meat
            # 1100: Milk
            # 21: FruitVeg
            # 42: Fuel
            # 1700: Sweets
            # 4700: Shoes
            # 9400: Utilities
            # 7400: Construction
            
            # Hypothesis 1: Fuel -> Utilities (Transport inside Utilities?)
            # Hypothesis 2: Milk -> Sweets (maybe?)
            # Hypothesis 3: Meat -> ? (No Sausage code in micro export yet)
            
            # Let's try correlations between these available groups
            self.pairs = [
                (42, 9400, 'Топливо -> ЖКУ'),
                (42, 10, 'Топливо -> Мясо (логистика)'),
                (42, 21, 'Топливо -> Плодоовощи (логистика)'),
                (1100, 1700, 'Молоко -> Кондитерские')
            ]
            
        except Exception as e:
            print(f"Ошибка: {e}")

    def analyze_chains(self):
        print("\n--- Анализ производственных цепочек (Lag 1-3) ---")
        results = []
        
        pivot = self.df.pivot_table(index='Date', columns='Item_code', values='MoM')
        
        for raw_code, prod_code, name in self.pairs:
            if raw_code not in pivot.columns or prod_code not in pivot.columns:
                continue
                
            raw = pivot[raw_code]
            prod = pivot[prod_code]
            
            # Check lags 1, 2, 3
            best_corr = -1
            best_lag = 0
            
            for lag in [0, 1, 2, 3]:
                raw_shifted = raw.shift(lag)
                common = pd.concat([raw_shifted, prod], axis=1).dropna()
                if len(common) > 24:
                    corr = common.corr().iloc[0, 1]
                    if abs(corr) > abs(best_corr): # Magnitude
                        best_corr = corr
                        best_lag = lag
            
            results.append({
                'Цепочка': name,
                'Корреляция': best_corr,
                'Лучший лаг': best_lag
            })
            
        res_df = pd.DataFrame(results).sort_values('Корреляция', ascending=False)
        print(res_df)
        return res_df

if __name__ == "__main__":
    chains = SirenaChains()
    chains.load_data()
    chains.analyze_chains()
