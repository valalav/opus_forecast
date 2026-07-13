import pandas as pd
import numpy as np
from sirena_component_bvar import SirenaComponentBVAR
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_backtest():
    # Load Data
    df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
    cols_to_fix = ['mom', 'Prod', 'Nonprod', 'Serv', 'usd_nom_i']
    for col in cols_to_fix:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    if df['Date'].isna().any(): df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    # Define test period (last 12 months)
    test_dates = df.index[-12:]
    
    results = []
    
    print(f"Запуск бэктеста Component BVAR на {len(test_dates)} точках...")
    
    for date in test_dates:
        cutoff = date - pd.DateOffset(months=1)
        train_df = df[df.index <= cutoff].copy()
        
        # Skip if not enough data
        if len(train_df) < 24: continue
        
        # try:
        if True:
            model = SirenaComponentBVAR()
            model.fit(train_df)
            
            # Forecast 1 step
            fc = model.predict(horizon=1)
            pred_val = fc['CPI'].iloc[0] - 100 # Convert back to MoM %
            
            actual = df.loc[date, 'mom'] - 100
            
            results.append({
                'Date': date,
                'Actual': actual,
                'Predicted': pred_val,
                'Error': actual - pred_val
            })
        # except Exception as e:
        #     print(f"Error on {date}: {e}")
            
    res_df = pd.DataFrame(results)
    
    if len(res_df) > 0:
        mae = mean_absolute_error(res_df['Actual'], res_df['Predicted'])
        rmse = np.sqrt(mean_squared_error(res_df['Actual'], res_df['Predicted']))
        
        print("\n=== Результаты Component BVAR (12 мес) ===")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        
        # Сравнение с BVAR v2.0 (из отчета)
        # BVAR v2.0 (Overall 1yr): MAE ~0.368, RMSE ~0.454
        print("\nСравнение с BVAR v2.0 (Baseline):")
        print(f"MAE Delta:  {mae - 0.3684:.4f}")
        print(f"RMSE Delta: {rmse - 0.4540:.4f}")
        
        # Вывод последних 5 прогнозов
        print("\nПоследние 5 точек:")
        print(res_df.tail(5)[['Date', 'Actual', 'Predicted', 'Error']])
    else:
        print("Нет результатов")

if __name__ == "__main__":
    run_backtest()
