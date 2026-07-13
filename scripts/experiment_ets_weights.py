
import pandas as pd
import numpy as np
from sirena.models.ridge import RidgeForecaster

def load_data():
    """Load and prepare data."""
    df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
    
    df = df.rename(columns={
        'mom': 'Все товары и услуги',
        'Prod': 'Продовольственные товары',
        'Nonprod': 'Непродовольственные товары',
        'Serv': 'Услуги'
    })

    # Fix columns
    cols = ['Все товары и услуги', 'Продовольственные товары', 'Непродовольственные товары', 'Услуги', 'Ki', 'Ruonia']
    for col in cols:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
    df = df.set_index('Date').sort_index()
    return df

def run_experiment_horizon(df, horizon, experiment_name, weights):
    """Run backtest for a specific horizon and weights."""
    print(f"\nRunning {experiment_name} (Horizon={horizon})...")
    
    model = RidgeForecaster(ets_weights=weights)
    
    # Run custom backtest loop for specific horizon
    start_date = pd.Timestamp('2023-01-01') # Shortened for speed, focus on recent
    valid_dates = df.index[df.index >= start_date]
    
    errors = []
    
    for date in valid_dates:
        # Train on data up to date-horizon
        # But wait, standard backtest uses date as target
        # Let's use model.backtest logic but for specific horizon
        
        # Actually easiest is to trust the model.forecast output if we simulate a rolling loop
        pass

    # Re-using internal backtest logic but we need multi-step verification.
    # The standard backtest() in Ridge only does 1-step forecast check in the simplified version I saw?
    # No, it calls model.predict which returns a single value.
    # Let's write a proper multi-step backtest here.
    
    results = []
    test_dates = valid_dates[::2] # Skip every other month to speed up
    
    for cutoff in test_dates:
        train = df[df.index < cutoff].copy()
        target_date = cutoff + pd.DateOffset(months=horizon-1) # Horizon th step
        
        if target_date not in df.index:
            continue
            
        try:
            model = RidgeForecaster(ets_weights=weights)
            model.fit(train)
            
            # Forecast h steps
            fc = model.forecast(horizon=horizon)
            pred = fc[horizon-1] # The value at the specific horizon
            
            actual = df.loc[target_date, 'Все товары и услуги']
            # Convert index to % if needed, or assume data is index
            # Ridge model expects Index (~100.x) output from forecast?
            # Looking at code: forecast returns "MoM в %" (e.g. 0.5 for 100.5)
            # data is 100.x
            
            actual_mom = actual - 100
            
            error = actual_mom - pred
            results.append(abs(error))
        except Exception as e:
            print(f"Err {cutoff}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    return np.mean(results)

def main():
    df = load_data()
    
    # Configs
    configs = {
        "Original": {
            1: 0.9, 2: 0.0, 3: 0.5, 4: 0.3,
            5: 0.9, 6: 0.5, 7: 0.0, 8: 0.5,
            9: 0.9, 10: 0.9, 11: 0.0, 12: 0.0
        },
        "Balanced (0.5)": {k: 0.5 for k in range(1, 13)},
        "Pure Model (0.0)": {k: 0.0 for k in range(1, 13)},
        "Heavy Seasonal (0.8)": {k: 0.8 for k in range(1, 13)},
    }
    
    horizons = [1, 2, 12]
    
    results = {}
    
    for name, weights in configs.items():
        results[name] = {}
        for h in horizons:
            mae = run_experiment_horizon(df, h, name, weights)
            results[name][h] = mae
            print(f"  > MAE (h={h}): {mae:.4f}")
            
    print("\n\n=== FINAL RESULTS (MAE) ===")
    print(f"{'Configuration':<20} | {'h=1':<8} | {'h=2':<8} | {'h=12':<8}")
    print("-" * 50)
    for name, metrics in results.items():
        print(f"{name:<20} | {metrics[1]:.4f}   | {metrics[2]:.4f}   | {metrics[12]:.4f}  ")

if __name__ == "__main__":
    main()
