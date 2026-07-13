import pandas as pd
import numpy as np

def calculate_metrics(df, model_col, actual_col='Actual'):
    # Remove NaN
    df = df.dropna(subset=[model_col, actual_col])
    
    y_true = df[actual_col]
    y_pred = df[model_col]
    
    # Error
    error = y_pred - y_true
    
    # ME (Mean Error) - Bias
    me = np.mean(error)
    
    # MAE
    mae = np.mean(np.abs(error))
    
    # RMSE
    rmse = np.sqrt(np.mean(error**2))
    
    # MAPE (Handle zeros by adding epsilon or ignoring)
    # Inflation can be 0 or negative, MAPE is tricky. 
    # Let's use symmetric MAPE or just standard, warning about zeros.
    # If y_true is 0, we skip or treat as large error.
    non_zero = y_true != 0
    mape = np.mean(np.abs((error[non_zero]) / y_true[non_zero])) * 100
    
    # Directional Accuracy (Sign Match)
    # If both have same sign (or both zero)
    same_sign = (np.sign(y_true) == np.sign(y_pred))
    dir_acc = np.mean(same_sign) * 100
    
    return {
        'ME': me,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'DirAcc': dir_acc
    }

df = pd.read_csv('rolling_backtest_1yr_results.csv')

print("--- Overall Metrics (All Horizons) ---")
print("BVAR:")
print(calculate_metrics(df, 'BVAR'))
print("\nSARIMA (Alternative):")
print(calculate_metrics(df, 'SARIMA'))

print("\n--- Horizon 1 Metrics ---")
df_h1 = df[df['horizon'] == 1]
print("BVAR (H1):")
print(calculate_metrics(df_h1, 'BVAR'))
print("\nSARIMA (H1):")
print(calculate_metrics(df_h1, 'SARIMA'))
