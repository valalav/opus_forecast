import pandas as pd
import sys

def check_status():
    try:
        # Load latest results
        df = pd.read_csv('docs/long_backtest_results.csv')
        
        # Get last available row with both BVAR and Ensemble data
        # Ensure we don't pick a row where one is NaN
        valid_df = df.dropna(subset=['bvar_pred', 'Ensemble_Pred'])
        
        if valid_df.empty:
            print("Error: No valid data found in docs/long_backtest_results.csv")
            return

        last_row = valid_df.iloc[-1]
        
        # Calculate Signal
        bvar_pred = last_row['bvar_pred']
        ensemble_pred = last_row['Ensemble_Pred']
        deviance = abs(bvar_pred - ensemble_pred)
        
        # Parameters (derived from research)
        THRESHOLD = 0.1790
        RISK_MULTIPLIER = 1.55
        
        print("\n=== SYSTEM STATUS CHECK: EARLY WARNING ORACLE ===")
        print(f"Date of Analysis: {last_row['Date']}")
        print("-" * 50)
        print(f"Ensemble Forecast: {ensemble_pred:.4f}")
        print(f"BVAR Forecast:     {bvar_pred:.4f}")
        print(f"Divergence:        {deviance:.4f}")
        print("-" * 50)
        
        if deviance > THRESHOLD:
            print("🚨 ALERT STATUS: RED 🚨")
            print(f"Warning: High Model Disagreement Detected.")
            print(f"Interpretation: BVAR is deviating significantly from consensus.")
            print(f"Forecast Confidence: LOW")
            print(f"Action: Widen Confidence Intervals by {RISK_MULTIPLIER}x")
            print(f"Expected Volatility Risk: HIGH (Anomaly Probability increased)")
        else:
            print("✅ ALERT STATUS: GREEN")
            print(f"Status: Models are in alignment.")
            print(f"Forecast Confidence: NORMAL")
            print(f"Action: Standard Operating Procedure")
            
    except Exception as e:
        print(f"System Error: {e}")

if __name__ == "__main__":
    check_status()
