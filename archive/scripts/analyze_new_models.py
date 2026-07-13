import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sirena.models import RidgeForecaster, FundamentalForecaster
from sirena.macro_features import load_brent_prices

def load_data():
    # Load main data
    # Use index_col='Day' to avoid confusion
    df = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',', parse_dates=['Day'], dayfirst=True, index_col='Day')
    df.index.name = 'Date'
    
    # Filter for "Все товары и услуги"
    if 'Товар' in df.columns:
        df = df[df['Товар'] == 'Все товары и услуги']
        
    # Rename MoM column
    if 'MoM' in df.columns:
        df = df.rename(columns={'MoM': 'Все товары и услуги'})
        
    # Convert target to numeric (handle comma decimal if read_csv failed)
    if df['Все товары и услуги'].dtype == 'object':
        df['Все товары и услуги'] = df['Все товары и услуги'].astype(str).str.replace(',', '.').astype(float)
    
    # Normalize index
    df.index = pd.to_datetime(df.index) + pd.offsets.MonthBegin(0)
    
    # Load macro data (Ki, Ruonia)
    macro = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',', parse_dates=['Date'], dayfirst=True)
    # Strip whitespace
    macro.columns = macro.columns.str.strip()
    # Normalize dates
    macro['Date'] = pd.to_datetime(macro['Date']) + pd.offsets.MonthBegin(0)
    macro = macro.set_index('Date')
    
    # Join macro columns
    cols_to_use = ['Ki', 'Ruonia', 'usd_nom_i']
    df = df.join(macro[cols_to_use], how='left')
    
    # Load Brent
    brent = load_brent_prices()
    brent.index = brent.index + pd.offsets.MonthBegin(0)
    df = df.join(brent[['brent', 'brent_pct']], how='left')
    
    # Filter for "Все товары и услуги" only if multiple rows per date
    # But infl_kbr.csv seems to have 'Товар' column.
    if 'Товар' in df.columns:
        df = df[df['Товар'] == 'Все товары и услуги']
        
    return df

def run_backtest():
    df = load_data()
    print(f"Data loaded: {len(df)} rows. Columns: {df.columns}")
    
    # Models
    ridge = RidgeForecaster()
    fund = FundamentalForecaster()
    
    # Backtest settings
    start_date = '2023-01-01'
    target_col = 'Все товары и услуги'
    
    print(f"Starting backtest from {start_date}...")
    print(f"Data range: {df.index.min()} to {df.index.max()}")
    print(f"Target col non-nulls: {df[target_col].count()}")
    
    res_ridge = ridge.backtest(df, start_date=start_date, target_col=target_col)
    print(f"Ridge backtest done. Results: {len(res_ridge)}")
    if not res_ridge.empty:
        print(res_ridge.head())
    
    res_fund = fund.backtest(df, start_date=start_date, target_col=target_col)
    print(f"Fundamental backtest done. Results: {len(res_fund)}")
    if not res_fund.empty:
        print(res_fund.head())
    
    # Metrics
    metrics_ridge = ridge.get_metrics(res_ridge)
    metrics_fund = fund.get_metrics(res_fund)
    
    print("\nResults:")
    print(f"Ridge: MAE={metrics_ridge['MAE']:.4f}, RMSE={metrics_ridge['RMSE']:.4f}")
    print(f"Fund : MAE={metrics_fund['MAE']:.4f}, RMSE={metrics_fund['RMSE']:.4f}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(res_ridge['date'], res_ridge['actual'], label='Actual', color='black', linewidth=2)
    plt.plot(res_ridge['date'], res_ridge['prediction'], label=f"Ridge (MAE={metrics_ridge['MAE']:.3f})")
    plt.plot(res_fund['date'], res_fund['prediction'], label=f"Fundamental (MAE={metrics_fund['MAE']:.3f})")
    plt.legend()
    plt.title('Model Comparison: Ridge vs Fundamental (No Seasonality)')
    plt.grid(True)
    plt.savefig('comparison_fundamental.png')
    print("Plot saved to comparison_fundamental.png")

if __name__ == "__main__":
    run_backtest()
