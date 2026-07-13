import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_fuel():
    print("Loading data...")
    try:
        df = pd.read_csv('data/subcomponent_monthly_fixed.csv')
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception as e:
        print(f"Error: {e}")
        return

    # Filter for Fuel
    fuel = df[df['Субкомпонент'] == 'Топливо моторное'].copy()
    fuel = fuel.sort_values('Date').set_index('Date')
    
    if fuel.empty:
        print("No fuel data found.")
        return

    # Metrics
    fuel['MoM'] = fuel['MoM_pct']
    
    # 1. Define "Spike"
    # Let's look at distribution to define a spike
    mean_mom = fuel['MoM'].mean()
    std_mom = fuel['MoM'].std()
    
    spike_threshold = mean_mom + 1.5 * std_mom
    drop_threshold = mean_mom - 1.5 * std_mom
    
    print(f"Fuel MoM Stats: Mean={mean_mom:.3f}, Std={std_mom:.3f}")
    print(f"Spike Threshold (> 1.5 SD): {spike_threshold:.3f}%")
    print(f"Drop Threshold (< -1.5 SD): {drop_threshold:.3f}%")
    
    # 2. Lag Analysis
    # Does a spike at T lead to a drop at T+1, T+2, T+3?
    fuel['Next_1m'] = fuel['MoM'].shift(-1)
    fuel['Next_2m'] = fuel['MoM'].shift(-2)
    fuel['Next_3m'] = fuel['MoM'].shift(-3)
    fuel['Cum_3m_Future'] = fuel['Next_1m'] + fuel['Next_2m'].fillna(0) + fuel['Next_3m'].fillna(0)
    
    # Identify Spikes
    spikes = fuel[fuel['MoM'] > spike_threshold]
    print(f"\nFound {len(spikes)} spikes in history.")
    
    print("\n--- Analysis of Aftermath (What happens after a spike?) ---")
    print(spikes[['MoM', 'Next_1m', 'Next_2m', 'Next_3m', 'Cum_3m_Future']])
    
    avg_response = spikes[['Next_1m', 'Next_2m', 'Next_3m']].mean()
    print("\nAverage MoM after a spike:")
    print(avg_response)
    
    # Probability of Correction (Negative MoM)
    prob_neg_1m = (spikes['Next_1m'] < 0).mean()
    prob_neg_3m = (spikes['Cum_3m_Future'] < 0).mean()
    print(f"\nProbability of price drop in next month: {prob_neg_1m:.1%}")
    print(f"Probability of cumulative drop over 3 months: {prob_neg_3m:.1%}")
    
    # 3. Correlation Check (Mean Reversion)
    # Negative correlation between MoM(t) and MoM(t+k) implies reversion
    corr_1 = fuel['MoM'].corr(fuel['Next_1m'])
    corr_2 = fuel['MoM'].corr(fuel['Next_2m'])
    corr_3 = fuel['MoM'].corr(fuel['Next_3m'])
    
    print("\n--- Serial Correlation (Mean Reversion Check) ---")
    print(f"Lag 1 Correlation: {corr_1:.3f} (Negative = Reversion, Positive = Momentum)")
    print(f"Lag 2 Correlation: {corr_2:.3f}")
    print(f"Lag 3 Correlation: {corr_3:.3f}")
    
    # 4. Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(fuel['MoM'], fuel['Next_1m'], alpha=0.6)
    plt.axhline(0, color='gray', linestyle='--')
    plt.axvline(0, color='gray', linestyle='--')
    
    # Add trendline
    z = np.polyfit(fuel['MoM'], fuel['Next_1m'].fillna(0), 1)
    p = np.poly1d(z)
    plt.plot(fuel['MoM'], p(fuel['MoM']), "r--", label=f'Trend (Slope={z[0]:.2f})')
    
    plt.title("Fuel Price Mean Reversion: Today vs Tomorrow")
    plt.xlabel("Current Month MoM (%)")
    plt.ylabel("Next Month MoM (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('docs/fuel_reversion_plot.png')
    print("Plot saved to docs/fuel_reversion_plot.png")

if __name__ == "__main__":
    analyze_fuel()
