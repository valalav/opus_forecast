import pandas as pd
import numpy as np

# Load full results
df = pd.read_csv('docs/long_backtest_results.csv')

# Define Anomaly: Error > 0.5 or < -0.5
df['Is_Anomaly'] = df['Ensemble_Error'].abs() > 0.5

print("=== АНАЛИЗ ВЫЖИВАЕМОСТИ В АНОМАЛИЯХ (2024-2025) ===")
anomalies = df[df['Is_Anomaly']].copy()
print(f"Всего аномальных месяцев: {len(anomalies)} из {len(df)}")

print("\n--- Хронология Аномалий ---")
print(anomalies[['Date', 'Fact', 'Ensemble_Pred', 'Ensemble_Error', 'Best_Model']])

print("\n--- Кто спасал в кризис? (Best Model Count in Anomalies) ---")
print(anomalies['Best_Model'].value_counts())

print("\n--- Кто топил корабль? (Worst Model Analysis) ---")
# Need to find which model had max error in these months
# We have columns like 'ridge_pred', 'bvar_pred' etc. in the csv?
# Let's check columns. The run script saved 'model_pred' for all models.
model_cols = [c for c in df.columns if c.endswith('_pred') and 'Ensemble' not in c]
models = [c.replace('_pred', '') for c in model_cols]

worst_models = []
for idx, row in anomalies.iterrows():
    max_err = 0
    worst = None
    for m in models:
        pred = row[f'{m}_pred']
        if pd.notna(pred):
            err = abs(row['Fact'] - pred)
            if err > max_err:
                max_err = err
                worst = m
    worst_models.append(worst)

anomalies['Worst_Model'] = worst_models
print(pd.Series(worst_models).value_counts())

print("\n--- Средняя ошибка моделей в Аномальные месяцы ---")
# Calculate MAE for each model specifically during anomalies
mae_anomaly = {}
for m in models:
    preds = anomalies[f'{m}_pred']
    facts = anomalies['Fact']
    mae = (facts - preds).abs().mean()
    mae_anomaly[m] = mae

mae_df = pd.Series(mae_anomaly).sort_values()
print(mae_df)
