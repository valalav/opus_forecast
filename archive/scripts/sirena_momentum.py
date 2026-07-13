import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class SirenaMomentum:
    """
    Анализ внутримесячного импульса (Momentum).
    Гипотеза: Ускорение инфляции в конце месяца (Неделя 4 - Неделя 1) предсказывает рост в следующем месяце.
    """
    
    def __init__(self):
        self.weekly = None
        self.monthly = None
        
    def load_data(self):
        print("Загрузка данных...")
        # Load Weekly (Long format from extract, or Wide from Pipeline)
        # Let's use raw file and pivot manually to be safe
        try:
            # We need Wide format: Index=Item, Cols=Weeks
            raw_weekly = pd.read_csv('data/weekly_prices.csv', sep=';', decimal=',')
            self.weekly = raw_weekly.pivot_table(
                index='Товары', 
                columns='Сведено', 
                values='Значение',
                aggfunc='mean'
            )
            
            # Load Monthly Targets (Deviation from Seasonality)
            df_monthly = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')
            
            # Clean numeric
            if 'MoM' in df_monthly.columns:
                 if df_monthly['MoM'].dtype == object:
                     df_monthly['MoM'] = df_monthly['MoM'].astype(str).str.replace(',', '.')
                 df_monthly['MoM'] = pd.to_numeric(df_monthly['MoM'], errors='coerce')

            # Fix dates
            if 'Day' in df_monthly.columns:
                 df_monthly['Date'] = pd.to_datetime(df_monthly['Day'], format='%d.%m.%Y', errors='coerce')
            
            if 'Товар' in df_monthly.columns:
                 pivot_m = df_monthly.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
            else:
                 pivot_m = df_monthly.set_index('Date')
                 
            # Calculate Deviation (Target)
            # De-seasonalize?
            # Simple MoM is fine for first check.
            self.monthly = pivot_m['Все товары и услуги'] - 100
            
        except Exception as e:
            print(f"Ошибка: {e}")

    def analyze_momentum(self):
        print("\n--- Анализ недельного импульса (Acceleration) ---")
        
        # Calculate Momentum for each month
        # Momentum = Avg Price (Week 4) / Avg Price (Week 1) - 1
        # Need to group weeks by month
        
        weeks = self.weekly.columns
        week_map = {}
        for w in weeks:
            try:
                y_str, w_str = str(w).split('_')
                # ISO week to Month
                from datetime import datetime
                d = datetime.fromisocalendar(int(y_str), int(w_str), 1)
                month_key = f"{d.year}-{d.month:02d}"
                if month_key not in week_map: week_map[month_key] = []
                week_map[month_key].append(w)
            except: pass
            
        results = []
        
        for m_key, w_list in week_map.items():
            if len(w_list) < 2: continue
            w_list.sort()
            
            first_week = w_list[0]
            last_week = w_list[-1]
            
            # Basket acceleration
            # Sum of prices in last week vs first week
            # Better: Median growth of items
            
            p_first = self.weekly[first_week]
            p_last = self.weekly[last_week]
            
            # Ratio
            ratios = p_last / p_first
            acceleration = (ratios.median() - 1) * 100
            
            # Target: Next Month's Inflation
            # Current month is m_key. Target is m_key + 1 month.
            date_curr = pd.to_datetime(f"{m_key}-01")
            date_next = date_curr + pd.DateOffset(months=1)
            
            if date_next in self.monthly.index:
                target = self.monthly.loc[date_next]
                
                results.append({
                    'Month': m_key,
                    'Acceleration': acceleration,
                    'Next_CPI': target
                })
                
        res_df = pd.DataFrame(results)
        if res_df.empty:
            print("Нет данных для анализа.")
            return
            
        # Correlation
        corr = res_df[['Acceleration', 'Next_CPI']].corr().iloc[0, 1]
        print(f"Корреляция (Ускорение внутри месяца -> ИПЦ след. месяца): {corr:.2f}")
        
        if corr > 0.3:
            print("✅ Внутримесячное ускорение предсказывает будущую инфляцию.")
        else:
            print("❌ Эффект импульса слабый/шумный.")
            
        return res_df

if __name__ == "__main__":
    sm = SirenaMomentum()
    sm.load_data()
    sm.analyze_momentum()
