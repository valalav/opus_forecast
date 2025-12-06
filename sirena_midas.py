import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class SirenaMIDAS:
    """
    U-MIDAS (Unrestricted Mixed Data Sampling) Model for Nowcasting.
    
    Uses high-frequency weekly data to predict low-frequency monthly inflation.
    Approach:
    1. Aggregate weekly data into monthly 'features'.
       - Mean price growth of top volatile items.
       - Weighted average of weekly basket.
       - Lags of weekly basket (Week 1, Week 2, Week 3, Week 4).
    2. Train a regression model:
       Monthly_CPI = alpha + beta * Weekly_Features + gamma * Trend + error
    """
    
    def __init__(self):
        self.model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.weekly_features = None
        
    def prepare_data(self, weekly_df, monthly_df):
        """
        Align weekly and monthly data.
        weekly_df: Wide format (Index=[Item, Code], Cols=Weeks YYYY_WW)
        monthly_df: Index=Date, Col='mom' (or target)
        """
        # 1. Process Weekly Data
        # Identify correct ID columns
        id_cols = ['Товары', 'Rostat_code']
        if 'Наименование ' in weekly_df.columns:
             id_cols[0] = 'Наименование '
        elif 'Наименование' in weekly_df.columns:
             id_cols[0] = 'Наименование'
             
        # Transpose to Time Series: Index=Week, Cols=Items
        # Drop non-week columns like Weights if present
        drop_cols = [c for c in weekly_df.columns if c not in id_cols and not (str(c)[0].isdigit() and '_' in str(c))]
        w_work = weekly_df.drop(columns=drop_cols, errors='ignore')
        
        w_ts = w_work.set_index(id_cols).T
        
        # Reset index to make Week Code a column
        w_ts = w_ts.reset_index().rename(columns={'index': 'Week_Code'})
        
        # Convert Week Code to Date
        dates = []
        for val in w_ts['Week_Code']:
            try:
                y_str, w_str = str(val).split('_')
                d = datetime.fromisocalendar(int(y_str), int(w_str), 1)
                dates.append(d)
            except:
                dates.append(pd.NaT)
        
        w_ts['Date'] = dates
        w_ts = w_ts.dropna(subset=['Date'])
        w_ts['Year'] = w_ts['Date'].dt.year
        w_ts['Month'] = w_ts['Date'].dt.month
        
        # 2. Aggregate Weeks to Month
        # Group by Year, Month.
        # We need to aggregate the ITEM columns (which are now columns in w_ts).
        # Exclude 'Week_Code', 'Date'.
        numeric_cols = [c for c in w_ts.columns if c not in ['Week_Code', 'Date', 'Year', 'Month']]
        
        # Aggregate: Mean of weekly prices for each month
        monthly_items = w_ts.groupby(['Year', 'Month'])[numeric_cols].mean()
        
        # Robust Proxy: Median of individual item growth rates
        # This handles missing items gracefully (they become NaN in pct_change and ignored in median)
        item_growth = monthly_items.pct_change()
        monthly_agg = pd.DataFrame(index=monthly_items.index)
        monthly_agg['Weekly_Proxy_MoM'] = item_growth.median(axis=1) * 100
        
        # Create Lag features from weekly structure?
        # U-MIDAS uses fine-grained lags.
        # e.g. Week 1 growth, Week 2 growth...
        # This requires fixed number of weeks per month (4).
        # Let's stick to "Bridge" approach first: Monthly Average Proxy.
        
        # 3. Merge with Monthly Target
        monthly_agg = monthly_agg.reset_index()
        monthly_agg['Date'] = pd.to_datetime(monthly_agg[['Year', 'Month']].assign(DAY=1))
        monthly_agg = monthly_agg.set_index('Date')
        
        # Target
        target = monthly_df['Все товары и услуги'] - 100 # MoM Growth
        
        # Merge
        df = pd.concat([target, monthly_agg['Weekly_Proxy_MoM']], axis=1).dropna()
        df.columns = ['Target', 'Weekly_Proxy']
        
        # Add Lags
        df['Target_Lag1'] = df['Target'].shift(1)
        df = df.dropna()
        
        return df
    
    def fit(self, df):
        X = df[['Weekly_Proxy', 'Target_Lag1']]
        y = df['Target']
        
        X_sc = self.scaler.fit_transform(X)
        self.model.fit(X_sc, y)
        self.is_fitted = True
        
        r2 = self.model.score(X_sc, y)
        return r2
    
    def predict(self, weekly_proxy, target_lag1):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
            
        X = np.array([[weekly_proxy, target_lag1]])
        X_sc = self.scaler.transform(X)
        return self.model.predict(X_sc)[0]

if __name__ == "__main__":
    from datetime import datetime
    print("SirenaMIDAS initialized.")
