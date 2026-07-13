import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class RegionalDeepDive:
    def __init__(self):
        self.df = None
        self.items = None
        self.regions = None
        
    def load_data(self):
        print("Загрузка данных...")
        # Load Micro Data (All regions, key items)
        self.df = pd.read_csv('data/all_regions_micro.csv')
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Load Item Names mapping if needed, but Item_code is in df
        # Let's map codes to names for readable output
        # 10: Meat, 1100: Milk, 21: FruitsVeg, 42: Fuel, 9400: Utilities
        self.item_map = {
            10: 'Мясопродукты',
            1100: 'Молоко',
            21: 'Плодоовощи',
            42: 'Топливо',
            9400: 'ЖКУ',
            1700: 'Кондитерские',
            7400: 'Стройматериалы'
        }
        self.df['Item_Name'] = self.df['Item_code'].map(self.item_map)
        self.df = self.df.dropna(subset=['Item_Name'])
        
        # Regions map
        regs = pd.read_csv('data/regions_names.csv')
        self.reg_map = dict(zip(regs['Region_code'], regs['Region_names']))
        
        # Codes
        self.kbr_code = 7
        # Find Stavropol code
        # Search in regions names
        self.stav_code = regs[regs['Region_names'].str.contains("Ставрополь", case=False)]['Region_code'].iloc[0]
        print(f"КБР: {self.kbr_code}, Ставрополь: {self.stav_code}")

    def compare_baskets(self):
        """
        Compare price dynamics of same items in KBR vs Stavropol.
        """
        print("\n--- Корреляция товаров (КБР vs Ставрополь) ---")
        
        results = []
        
        for item in self.df['Item_Name'].unique():
            # Extract series
            kbr = self.df[(self.df['Region_code'] == self.kbr_code) & (self.df['Item_Name'] == item)].set_index('Date')['MoM']
            stav = self.df[(self.df['Region_code'] == self.stav_code) & (self.df['Item_Name'] == item)].set_index('Date')['MoM']
            
            # Sync index
            common = pd.concat([kbr, stav], axis=1).dropna()
            common.columns = ['KBR', 'Stavropol']
            
            if len(common) > 24:
                # Full Correlation
                corr = common.corr().iloc[0, 1]
                
                # Recent Correlation (Last 24 months)
                recent = common.iloc[-24:]
                corr_recent = recent.corr().iloc[0, 1]
                
                # Lead-Lag? (Stavropol leading KBR)
                # Corr(KBR_t, Stav_t-1)
                lag_df = pd.concat([kbr, stav.shift(1)], axis=1).dropna()
                corr_lag = lag_df.corr().iloc[0, 1]
                
                results.append({
                    'Товар': item,
                    'Corr (Full)': corr,
                    'Corr (24m)': corr_recent,
                    'Lag-1 Corr': corr_lag,
                    'Status': 'Sync' if corr > corr_lag else 'Lagged'
                })
                
        res_df = pd.DataFrame(results).sort_values('Corr (24m)', ascending=False)
        print(res_df)
        return res_df

    def rolling_correlation_analysis(self, item_name='Плодоовощи'):
        """
        Analyze how correlation evolved over time.
        """
        print(f"\n--- Скользящая корреляция: {item_name} ---")
        
        kbr = self.df[(self.df['Region_code'] == self.kbr_code) & (self.df['Item_Name'] == item_name)].set_index('Date')['MoM']
        stav = self.df[(self.df['Region_code'] == self.stav_code) & (self.df['Item_Name'] == item_name)].set_index('Date')['MoM']
        
        common = pd.concat([kbr, stav], axis=1).dropna()
        common.columns = ['KBR', 'Stavropol']
        
        # Rolling 24 months correlation
        rolling_corr = common['KBR'].rolling(window=24).corr(common['Stavropol'])
        
        # Plot
        plt.figure(figsize=(12, 5))
        plt.plot(rolling_corr.index, rolling_corr, label=f'Корреляция {item_name} (24 мес окно)', color='purple')
        plt.axhline(0, color='black', lw=1)
        plt.axhline(0.7, color='green', ls='--', label='Сильная связь')
        plt.title(f"Устойчивость связи КБР-Ставрополь ({item_name})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('rolling_corr_stavropol.png')
        print("График сохранен: rolling_corr_stavropol.png")
        
        # Analyze breaks
        # Find periods where correlation dropped significantly
        drops = rolling_corr[rolling_corr < 0.3]
        if not drops.empty:
            print("Периоды разрыва связи (Corr < 0.3):")
            # Group by year
            print(drops.groupby(drops.index.year).count())

if __name__ == "__main__":
    dive = RegionalDeepDive()
    dive.load_data()
    dive.compare_baskets()
    dive.rolling_correlation_analysis('Плодоовощи')
    dive.rolling_correlation_analysis('Топливо')
