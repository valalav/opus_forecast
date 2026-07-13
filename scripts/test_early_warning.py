import pandas as pd
import numpy as np

# Load results
df = pd.read_csv('docs/long_backtest_results.csv')

# Identify prediction columns
pred_cols = [c for c in df.columns if c.endswith('_pred') and 'Ensemble' not in c]

# Calculate "Disagreement Index" (Standard Deviation of predictions for that month)
# We only take non-NaN predictions
df['Model_Dispersion'] = df[pred_cols].std(axis=1)

# Calculate Absolute Error
df['Abs_Error'] = df['Ensemble_Error'].abs()

# 1. Correlation Analysis
corr = df['Model_Dispersion'].corr(df['Abs_Error'])

print(f"Корреляция между 'Разбросом мнений моделей' и 'Ошибкой прогноза': {corr:.4f}")

# 2. Threshold Test
# If Dispersion > X, what is the chance of Anomaly (Error > 0.5)?
high_uncertainty = df[df['Model_Dispersion'] > 0.2] # 0.2 is empirical threshold
prob_anomaly = (high_uncertainty['Abs_Error'] > 0.5).mean()

print(f"\nВероятность аномалии при высоком разбросе моделей (>0.2): {prob_anomaly:.1%}")
print(f"Вероятность аномалии в среднем по выборке: {(df['Abs_Error'] > 0.5).mean():.1%}")

# 3. Check specific months
print("\nМесяцы с самым высоким разбросом мнений (Топ-5):")
print(df.sort_values('Model_Dispersion', ascending=False)[['Date', 'Model_Dispersion', 'Abs_Error', 'Fact']].head(5))