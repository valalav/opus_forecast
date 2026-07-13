
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_favar_experiment():
    # Load subcomponent data
    df = pd.read_csv('data/subcomponent_monthly.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Pivot to wide format: Date x Subcomponent
    # Use MoM_pct
    wide = df.pivot(index='Date', columns='Субкомпонент', values='MoM_pct')
    
    # Fill NAs (if any) with 0 or interpolation
    # Prices shouldn't have many NaNs in properly curated data, but let's check
    print(f"Wide shape before dropna: {wide.shape}")
    wide = wide.dropna(axis=1, thresh=len(wide)*0.9) # Drop cols with >10% missing
    wide = wide.fillna(0) # Fill remaining with 0 (no price change assumption)
    print(f"Wide shape after cleaning: {wide.shape}")
    
    # Load Aggregate Inflation (Target)
    target_df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
    target_df['Date'] = pd.to_datetime(target_df['Date'], format='%d.%m.%Y').dt.to_period('M').dt.to_timestamp()
    target_df = target_df.set_index('Date')
    
    # Target: All goods and services
    # Ensure numeric
    target = target_df['mom']
    if target.dtype == object:
        target = target.astype(str).str.replace(',', '.').astype(float)
        
    # Align dates
    common_idx = wide.index.intersection(target.index)
    wide = wide.loc[common_idx]
    target = target.loc[common_idx]
    
    print(f"Aligned Data Points: {len(common_idx)}")
    
    # 1. PCA Extraction
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(wide)
    
    n_factors = 3
    pca = PCA(n_components=n_factors)
    factors = pca.fit_transform(X_scaled)
    
    explained_var = pca.explained_variance_ratio_
    print(f"Explained Variance by Top {n_factors} factors: {explained_var} (Sum: {np.sum(explained_var):.2f})")
    
    # 2. Correlation with Target
    # Factor 1 should be "General Inflation"
    # Note: PCA sign is arbitrary. Factor 1 might be -Inflation.
    
    print("\nCorrelations with Target (MoM):")
    for i in range(n_factors):
        corr = np.corrcoef(factors[:, i], target)[0, 1]
        print(f"Factor {i+1}: {corr:.4f}")
        
    # 3. Granger Causality / Predictive Power
    # Does Factor predict Inflation better than simple AR?
    # Simple test: OLS of Inflation_t on Inflation_{t-1} vs Inflation_{t-1} + Factor_{t-1}
    
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    
    # Shift data for 1-step forecast
    y = target.iloc[1:]
    X_ar = target.iloc[:-1].values.reshape(-1, 1)
    X_favar = np.hstack([X_ar, factors[:-1, :]]) # AR(1) + Factors lagged
    
    # Train/Test split (Last 24 months)
    test_size = 24
    if len(y) > test_size:
        X_ar_train, X_ar_test = X_ar[:-test_size], X_ar[-test_size:]
        X_favar_train, X_favar_test = X_favar[:-test_size], X_favar[-test_size:]
        y_train, y_test = y[:-test_size], y[-test_size:]
        
        # Model 1: AR(1)
        m1 = LinearRegression().fit(X_ar_train, y_train)
        p1 = m1.predict(X_ar_test)
        mae1 = mean_absolute_error(y_test, p1)
        
        # Model 2: FAVAR-Lite (AR + Factors)
        m2 = LinearRegression().fit(X_favar_train, y_train)
        p2 = m2.predict(X_favar_test)
        mae2 = mean_absolute_error(y_test, p2)
        
        print(f"\nForecasting Performance (MAE h=1, Last {test_size} months):")
        print(f"AR(1) Baseline: {mae1:.4f}")
        print(f"FAVAR (AR+3 Factors): {mae2:.4f}")
        
        if mae2 < mae1:
            print("CONCLUSION: FAVAR improves forecasting.")
        else:
            print("CONCLUSION: FAVAR does not improve forecasting (or overfitting).")

if __name__ == "__main__":
    run_favar_experiment()
