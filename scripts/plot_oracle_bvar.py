import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('docs/long_backtest_results.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Define Targets
df['Ensemble_Error'] = (df['Fact'] - df['Ensemble_Pred'])
df['Abs_Error'] = df['Ensemble_Error'].abs()
df['Next_Abs_Error'] = df['Abs_Error'].shift(-1)

# Define Signal: BVAR Deviance
df['BVAR_Deviance'] = (df['bvar_pred'] - df['Ensemble_Pred']).abs()

# Define Signal: Directional
# If BVAR > Ensemble, is it a signal that Fact will be > Ensemble?
df['BVAR_Diff'] = df['bvar_pred'] - df['Ensemble_Pred']
df['Next_Error_Signed'] = df['Ensemble_Error'].shift(-1)

# --- Visualization ---
plt.figure(figsize=(12, 10))

# 1. Time Series
plt.subplot(2, 1, 1)
plt.title("Oracle Signal: BVAR Deviance (Today) vs Ensemble Crisis (Tomorrow)", fontsize=14)
plt.plot(df['Date'], df['Next_Abs_Error'], label='Next Month Crisis (Abs Error)', color='red', linewidth=2, marker='o')
plt.plot(df['Date'], df['BVAR_Deviance'], label='BVAR "Hysteria" (Deviance from Mean)', color='blue', linestyle='--', alpha=0.7)
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylabel("Magnitude")

# 2. Scatter
plt.subplot(2, 2, 3)
plt.title("Correlation Analysis")
sns.regplot(x=df['BVAR_Deviance'], y=df['Next_Abs_Error'], color='purple')
plt.xlabel("BVAR Deviance (Today)")
plt.ylabel("Next Month Error (Tomorrow)")

# 3. Hit Rate Analysis
threshold = df['BVAR_Deviance'].mean() + 0.5 * df['BVAR_Deviance'].std()
high_signal = df[df['BVAR_Deviance'] > threshold]
low_signal = df[df['BVAR_Deviance'] <= threshold]

avg_error_high = high_signal['Next_Abs_Error'].mean()
avg_error_low = low_signal['Next_Abs_Error'].mean()

plt.subplot(2, 2, 4)
bars = plt.bar(['Low BVAR Deviance', 'High BVAR Deviance'], [avg_error_low, avg_error_high], color=['green', 'red'])
plt.title("Risk Stratification")
plt.ylabel("Average Next Month Error")
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('docs/oracle_bvar_analysis.png')
print(f"Plot saved to docs/oracle_bvar_analysis.png")

# --- Stats Output ---
print("\n=== BVAR ORACLE STATISTICS ===")
print(f"Correlation (Deviance vs Next Error): {df['BVAR_Deviance'].corr(df['Next_Abs_Error']):.4f}")
print(f"Signal Threshold (Mean + 0.5 SD): {threshold:.4f}")
print(f"Avg Error when Signal is LOW: {avg_error_low:.4f}")
print(f"Avg Error when Signal is HIGH: {avg_error_high:.4f}")
print(f"Risk Multiplier: {avg_error_high / avg_error_low:.2f}x")

# Directional Check
# Does BVAR pull the prediction in the right direction?
# We check if (BVAR - Ensemble) has same sign as (Fact - Ensemble)
same_sign = np.sign(df['BVAR_Diff']) == np.sign(df['Ensemble_Error'])
print(f"\nDirectional Accuracy (Does BVAR 'pull' correctly?): {same_sign.mean():.2%}")
