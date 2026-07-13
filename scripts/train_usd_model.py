import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_data():
    # Load inflation data (contains USD, Ki, Ruonia)
    # Handle BOM if present, use correct separator and decimal
    df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',', parse_dates=['Date'], dayfirst=True)
    
    # Strip whitespace from columns
    df.columns = df.columns.str.strip()
    
    # Normalize to Month Start
    df['Date'] = pd.to_datetime(df['Date']) + pd.offsets.MonthBegin(0)
    df = df.set_index('Date')
    
    # Load Brent data
    brent = pd.read_csv('data/brent_prices.csv', parse_dates=['Date'])
    brent['Date'] = pd.to_datetime(brent['Date']) + pd.offsets.MonthBegin(0)
    brent = brent.set_index('Date')
    
    # Merge
    df = df.join(brent[['brent', 'brent_pct']], how='left')
    
    # Target: USD MoM change (usd_nom_i - 100)
    df['usd_mom'] = df['usd_nom_i'] - 100
    
    print("Data loaded. Info:")
    print(df.info())
    print("Head:")
    print(df.head())
    print("Tail:")
    print(df.tail())
    
    return df

def create_features(df):
    df = df.copy()
    
    # Lags of target
    for i in [1, 2, 3, 6, 12]:
        df[f'usd_lag{i}'] = df['usd_mom'].shift(i)
        
    # Momentum
    df['usd_ma3'] = df['usd_mom'].rolling(3).mean().shift(1)
    df['usd_ma6'] = df['usd_mom'].rolling(6).mean().shift(1)
    
    # Oil features
    df['brent_pct'] = df['brent'].pct_change() * 100
    for i in [1, 2, 3]:
        df[f'brent_lag{i}'] = df['brent_pct'].shift(i)
        
    # Key Rate features
    # Ki is usually annual rate. Monthly change?
    df['ki_diff'] = df['Ki'].diff()
    for i in [1, 3, 6]:
        df[f'ki_lag{i}'] = df['Ki'].shift(i)
        df[f'ki_diff_lag{i}'] = df['ki_diff'].shift(i)
        
    # Ruonia features
    df['ruonia_diff'] = df['Ruonia'].diff()
    df['spread'] = df['Ki'] - df['Ruonia']
    for i in [1, 3]:
        df[f'ruonia_lag{i}'] = df['Ruonia'].shift(i)
        df[f'spread_lag{i}'] = df['spread'].shift(i)
        
    return df.dropna()

def train_evaluate(model_name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    print(f"{model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}")
    return pred

def main():
    df = load_data()
    df = create_features(df)
    
    # Split
    test_start = '2024-01-01'
    train = df[df.index < test_start]
    test = df[df.index >= test_start]
    
    features = [c for c in df.columns if 'lag' in c or 'ma' in c]
    target = 'usd_mom'
    
    print(f"Train size: {len(train)}, Test size: {len(test)}")
    print(f"Features: {features}")
    
    X_train = train[features]
    y_train = train[target]
    X_test = test[features]
    y_test = test[target]
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Models
    models = {
        'Ridge': Ridge(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'GBM': GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
        'Linear': LinearRegression()
    }
    
    results = {}
    for name, model in models.items():
        results[name] = train_evaluate(name, model, X_train_s, y_train, X_test_s, y_test)
        
    # Baseline (Last Value / Random Walk)
    # Predict 0 change (Random Walk for levels)
    # Or predict previous month's change
    pred_naive_0 = np.zeros(len(y_test))
    mae_naive_0 = mean_absolute_error(y_test, pred_naive_0)
    print(f"Naive (0 change): MAE={mae_naive_0:.4f}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual', color='black', linewidth=2)
    for name, pred in results.items():
        plt.plot(y_test.index, pred, label=name)
    plt.plot(y_test.index, pred_naive_0, label='Naive (0)', linestyle='--')
    plt.legend()
    plt.title('USD MoM Forecast Comparison (2024-2025)')
    plt.grid(True)
    plt.savefig('usd_forecast_comparison.png')
    print("Plot saved to usd_forecast_comparison.png")

if __name__ == "__main__":
    main()
