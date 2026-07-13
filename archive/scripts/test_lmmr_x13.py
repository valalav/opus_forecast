"""
Test script for LMMR X13 model.

Tests:
1. X13-ARIMA integration
2. Model training
3. Backtest comparison with LMMR Hybrid
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())

def load_data():
    """Load and prepare data."""
    # Load inflation data
    df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')
    
    # Fix MoM column
    if 'MoM' in df_raw.columns:
        df_raw['MoM'] = pd.to_numeric(
            df_raw['MoM'].astype(str).str.replace(',', '.'),
            errors='coerce'
        )
    
    # Pivot
    if 'Товар' in df_raw.columns:
        df = df_raw.pivot(index='Date', columns='Товар', values='MoM')
    else:
        df = df_raw.set_index('Date')
    
    df.index = pd.to_datetime(df.index)
    df.index = pd.PeriodIndex(df.index, freq='M').to_timestamp()
    df = df.sort_index()
    
    # Load macro data
    try:
        macro = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
        macro['Date'] = pd.to_datetime(macro['Date'], format='%d.%m.%Y', errors='coerce')
        
        for col in ['usd_nom_i', 'Ki', 'Ruonia', 'brent']:
            if col in macro.columns:
                macro[col] = pd.to_numeric(macro[col], errors='coerce')
        
        macro['period_date'] = macro['Date'].dt.to_period('M').dt.to_timestamp()
        macro.set_index('period_date', inplace=True)
        
        for col in ['usd_nom_i', 'Ki', 'Ruonia', 'brent']:
            if col in macro.columns:
                df = df.join(pd.DataFrame(macro[col]), how='left')
    except Exception as e:
        print(f"Warning: Could not load macro data: {e}")
    
    return df

def test_x13_integration():
    """Test X13-ARIMA integration."""
    print("\n" + "="*60)
    print("TEST 1: X13-ARIMA Integration")
    print("="*60)
    
    df = load_data()
    target_col = 'Все товары и услуги'
    
    print(f"Data period: {df.index.min().date()} — {df.index.max().date()}")
    print(f"Total observations: {len(df)}")
    
    try:
        from sirena.models.lmmr_x13 import LMMRX13Forecaster
        
        # Test with small sample
        train_df = df[df.index < '2024-01-01'].copy()
        
        print(f"\nTraining LMMR X13 on {len(train_df)} observations...")
        model = LMMRX13Forecaster(alpha=0.5, use_x13=True)
        model.fit(train_df, target_col)
        
        print("\n✓ X13-ARIMA integration successful!")
        print(f"  Features used: {len(model._features)}")
        print(f"  X13 available: {model._x13_available}")
        
        # Test prediction
        test_date = pd.Timestamp('2024-01-01')
        pred = model.predict(df[df.index <= test_date], test_date)
        
        print(f"\nTest prediction for {test_date.date()}:")
        print(f"  Predicted: {pred['prediction']:.3f}")
        print(f"  SA MoM: {pred['sa_mom']:.3f}")
        print(f"  Seasonal factor: {pred['seasonal_factor']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ X13-ARIMA integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_backtest():
    """Run backtest and compare with LMMR Hybrid."""
    print("\n" + "="*60)
    print("TEST 2: Backtest Comparison")
    print("="*60)
    
    df = load_data()
    target_col = 'Все товары и услуги'
    
    try:
        from sirena.models.lmmr_x13 import LMMRX13Forecaster
        from sirena.models.lmmr_hybrid import LMMRHybridForecaster
        
        print("\nRunning LMMR X13 backtest...")
        model_x13 = LMMRX13Forecaster(alpha=0.5, use_x13=True)
        results_x13 = model_x13.backtest(df, start_date='2024-01-01', target_col=target_col)
        
        print("\nRunning LMMR Hybrid backtest...")
        model_hybrid = LMMRHybridForecaster(alpha=0.5)
        results_hybrid = model_hybrid.backtest(df, start_date='2024-01-01', target_col=target_col)
        
        if len(results_x13) > 0 and len(results_hybrid) > 0:
            # Ensure abs_error column exists
            if 'abs_error' not in results_hybrid.columns:
                results_hybrid['abs_error'] = results_hybrid['error'].abs()
            
            mae_x13 = results_x13['abs_error'].mean()
            mae_hybrid = results_hybrid['abs_error'].mean()
            
            print("\n" + "="*60)
            print("RESULTS")
            print("="*60)
            print(f"LMMR X13:     MAE = {mae_x13:.3f}, N = {len(results_x13)}")
            print(f"LMMR Hybrid:  MAE = {mae_hybrid:.3f}, N = {len(results_hybrid)}")
            print(f"Improvement:  {((mae_hybrid - mae_x13) / mae_hybrid * 100):+.1f}%")
            
            return True
        else:
            print("\n✗ No backtest results")
            return False
            
    except Exception as e:
        print(f"\n✗ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_importance():
    """Test feature importance."""
    print("\n" + "="*60)
    print("TEST 3: Feature Importance")
    print("="*60)
    
    df = load_data()
    target_col = 'Все товары и услуги'
    
    try:
        from sirena.models.lmmr_x13 import LMMRX13Forecaster
        
        train_df = df[df.index < '2024-01-01'].copy()
        
        model = LMMRX13Forecaster(alpha=0.5, use_x13=True)
        model.fit(train_df, target_col)
        
        importance = model.get_feature_importance()
        
        print("\nTop 10 Features:")
        print(importance.head(10).to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"\n✗ Feature importance failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("LMMR X13 MODEL TEST SUITE")
    print("="*60)
    
    results = []
    
    # Test 1: X13 Integration
    results.append(("X13 Integration", test_x13_integration()))
    
    # Test 2: Backtest
    results.append(("Backtest", test_backtest()))
    
    # Test 3: Feature Importance
    results.append(("Feature Importance", test_feature_importance()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(r[1] for r in results)
    
    if all_passed:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
