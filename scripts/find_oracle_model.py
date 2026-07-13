import pandas as pd
import numpy as np

# Load results
df = pd.read_csv('docs/long_backtest_results.csv')

# Define Anomaly Magnitude (Abs Error)
df['Anomaly_Magnitude'] = df['Ensemble_Error'].abs()

# Find prediction columns
models = [c.replace('_pred', '') for c in df.columns if c.endswith('_pred') and 'Ensemble' not in c]

results = []

for m in models:
    # Metric 1: "Contrarian Score" (How far is this model from the herd?)
    # Calculate herd mean excluding current model (roughly ensemble pred)
    # Simplified: abs(Model_Pred - Ensemble_Pred)
    df[f'{m}_deviance'] = (df[f'{m}_pred'] - df['Ensemble_Pred']).abs()
    
    # Check correlation with CURRENT Anomaly
    corr_curr = df[f'{m}_deviance'].corr(df['Anomaly_Magnitude'])
    
    # Check correlation with NEXT Month Anomaly (Predictive power)
    df['Next_Anomaly'] = df['Anomaly_Magnitude'].shift(-1)
    corr_next = df[f'{m}_deviance'].corr(df['Next_Anomaly'])
    
    # Check if the MODEL'S own value predicts anomaly (e.g. extreme forecast = danger)
    # abs(Model_Pred) -> Anomaly?
    corr_val = df[f'{m}_pred'].abs().corr(df['Anomaly_Magnitude'])
    
    results.append({
        'Model': m,
        'Corr_Deviance_Current': corr_curr,
        'Corr_Deviance_Next': corr_next,
        'Corr_Value_Current': corr_val
    })

res_df = pd.DataFrame(results).sort_values('Corr_Deviance_Next', ascending=False)

print("=== ПОИСК МОДЕЛИ-ОРАКУЛА ===")
print("Корреляция 'Особого мнения' модели (отклонение от ансамбля) с Аномалией СЛЕДУЮЩЕГО месяца:")
print(res_df[['Model', 'Corr_Deviance_Next', 'Corr_Deviance_Current']])

print("\nВывод: Если корреляция > 0.3-0.4, то 'истерика' этой модели сегодня предсказывает шторм завтра.")

# Check LightGBM specifically
lgbm = res_df[res_df['Model'] == 'lightgbm'].iloc[0]
print(f"\nLightGBM Deviance -> Next Anomaly Corr: {lgbm['Corr_Deviance_Next']:.4f}")
