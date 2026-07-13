import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

class SirenaBeta:
    """
    Анализ федерального импульса (Beta-коэффициент).
    Проверяет гипотезу: Инфляция КБР = alpha + beta * Инфляция РФ.
    """
    
    def __init__(self):
        self.data = None
        
    def load_data(self):
        print("Загрузка макро-индексов...")
        # Load aggregated regional data
        try:
            df = pd.read_csv('data/all_regions_indices.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Region 0 (Russia) vs Region 7 (KBR)
            # Item 1 (CPI)
            
            rus = df[(df['Region_code'] == 0) & (df['Item_code'] == 1)][['Date', 'MoM']].set_index('Date')
            kbr = df[(df['Region_code'] == 7) & (df['Item_code'] == 1)][['Date', 'MoM']].set_index('Date')
            
            self.data = pd.concat([rus, kbr], axis=1).dropna()
            self.data.columns = ['Russia', 'KBR']
            
            # Convert to Growth %
            self.data = self.data - 100
            
        except Exception as e:
            print(f"Ошибка: {e}")

    def analyze_beta(self):
        print("\n--- Анализ чувствительности к федеральному тренду (Beta) ---")
        
        # 1. Static Beta
        model = LinearRegression()
        X = self.data[['Russia']]
        y = self.data['KBR']
        
        model.fit(X, y)
        beta_static = model.coef_[0]
        r2_static = model.score(X, y)
        
        print(f"Статическая Beta (2010-2025): {beta_static:.2f}")
        print(f"R²: {r2_static:.2f}")
        
        # 2. Rolling Beta (24 months)
        rolling_beta = []
        dates = []
        
        window = 24
        for i in range(window, len(self.data)):
            subset = self.data.iloc[i-window:i]
            model.fit(subset[['Russia']], subset['KBR'])
            rolling_beta.append(model.coef_[0])
            dates.append(self.data.index[i])
            
        beta_series = pd.Series(rolling_beta, index=dates)
        
        print(f"Текущая Beta (последние 24 мес): {rolling_beta[-1]:.2f}")
        
        # 3. Lagged Impact
        # Does Russia(t-1) predict KBR(t)?
        common_lag = pd.concat([self.data['KBR'], self.data['Russia'].shift(1)], axis=1).dropna()
        common_lag.columns = ['KBR', 'Russia_Lag1']
        corr_lag = common_lag.corr().iloc[0, 1]
        
        print(f"Корреляция с лагом (РФ t-1 -> КБР t): {corr_lag:.2f}")
        
        if corr_lag > 0.5:
            print("✅ Федеральный тренд является опережающим индикатором.")
        else:
            print("❌ Лаговая связь слабая (синхронная динамика сильнее).")
            
        return beta_series

if __name__ == "__main__":
    sb = SirenaBeta()
    sb.load_data()
    sb.analyze_beta()
