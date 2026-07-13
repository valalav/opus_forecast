"""
Focused backtest for LMMR Claude model - Key metrics only
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sirena.models.lmmr_claude import LMMRForecasterClaude
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def focus_backtest():
    """
    Run a focused backtest for key metrics
    """
    print("LMMR CLAUDE MODEL - BACKTEST RESULTS")
    print()
    
    # Load and prepare data
    df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')
    df_raw['MoM'] = pd.to_numeric(df_raw['MoM'].astype(str).str.replace(',', '.'), errors='coerce')
    df = df_raw.pivot(index='Date', columns='Товар', values='MoM')
    df.index = pd.to_datetime(df.index)
    
    # Load exogenous data and align dates
    additional_data = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
    additional_data['Date'] = pd.to_datetime(additional_data['Date'], format='%d.%m.%Y', errors='coerce')
    
    for col in ['usd_nom_i', 'Ki_i', 'Ruonia', 'fl_potrb_zad', 'all_real']:
        if col in additional_data.columns:
            additional_data[col] = pd.to_numeric(additional_data[col], errors='coerce')

    # Align date formats
    df.index = df.index.to_period('M').to_timestamp()
    additional_data['period_date'] = additional_data['Date'].dt.to_period('M').dt.to_timestamp()
    additional_data.set_index('period_date', inplace=True)
    
    # Join data
    target_col = 'Все товары и услуги'
    for col in ['usd_nom_i', 'Ki_i', 'Ruonia', 'fl_potrb_zad', 'all_real']:
        if col in additional_data.columns:
            df = df.join(pd.DataFrame(additional_data[col]), how='left')
    
    # Backtest setup
    start_date = '2023-01-01'  # Focus on recent performance
    recent_data = df[df.index >= pd.Timestamp(start_date)]
    
    print(f"Testing period: {recent_data.index[0]} to {recent_data.index[-1]}")
    print(f"Number of data points: {len(recent_data)}")
    print()
    
    results = []
    for target_date in recent_data.index:
        # Prepare training data
        train_df = df[df.index < target_date].copy()
        
        if len(train_df) < 48:  # Need sufficient data for STL
            continue
        
        try:
            model = LMMRForecasterClaude(alpha=0.5)
            model.fit(train_df, target_col)
            
            test_df = df[df.index <= target_date].copy()
            pred = model.predict(test_df, target_date)
            prediction = pred['prediction']
            actual = df.loc[target_date, target_col]
            
            results.append({
                'date': target_date,
                'actual': actual,
                'prediction': prediction,
                'error': actual - prediction,
                'abs_error': abs(actual - prediction)
            })
        except Exception:
            continue
    
    if results:
        results_df = pd.DataFrame(results)
        
        # Calculate metrics
        mae = results_df['abs_error'].mean()
        rmse = np.sqrt((results_df['error'] ** 2).mean())
        mape = (results_df['abs_error'] / results_df['actual'].abs() * 100).mean()
        direction_accuracy = np.mean(
            np.sign(results_df['actual'] - 100) == np.sign(results_df['prediction'] - 100)
        )
        
        print("RECENT PERFORMANCE (2023-2025):")
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Direction Accuracy: {direction_accuracy:.1%}")
        print(f"  Number of predictions: {len(results_df)}")
        print()
        
        # Also calculate for full period if available
        all_test_dates = df.index[df.index >= pd.Timestamp('2020-01-01')]  # Full recent period
        all_results = []
        
        for target_date in all_test_dates:
            train_df = df[df.index < target_date].copy()
            if len(train_df) < 48:
                continue
                
            try:
                model = LMMRForecasterClaude(alpha=0.5)
                model.fit(train_df, target_col)
                
                test_df = df[df.index <= target_date].copy()
                pred = model.predict(test_df, target_date)
                
                actual_val = df.loc[target_date, target_col]
                pred_val = pred['prediction']
                all_results.append({
                    'date': target_date,
                    'actual': actual_val,
                    'prediction': pred_val,
                    'error': actual_val - pred_val,
                    'abs_error': abs(actual_val - pred_val)
                })
            except Exception:
                continue
        
        if all_results:
            all_results_df = pd.DataFrame(all_results)
            full_mae = all_results_df['abs_error'].mean()
            full_direction_acc = np.mean(
                np.sign(all_results_df['actual'] - 100) == np.sign(all_results_df['prediction'] - 100)
            )
            
            print("FULL RECENT PERIOD PERFORMANCE (2020-2025):")
            print(f"  MAE: {full_mae:.3f}")
            print(f"  Direction Accuracy: {full_direction_acc:.1%}")
            print(f"  Number of predictions: {len(all_results_df)}")
            print()
    
    # Compare with baseline (persistence model: predicting previous month's value)
    baseline_errors = []
    for i in range(1, len(recent_data)):
        actual = recent_data.iloc[i][target_col]
        # Baseline: predict previous month value
        predicted = recent_data.iloc[i-1][target_col]
        baseline_errors.append(abs(actual - predicted))
    
    if baseline_errors:
        baseline_mae = np.mean(baseline_errors)
        print(f"BASELINE (Persistence Model) MAE: {baseline_mae:.3f}")
        print(f"LMMR improvement over baseline: {(baseline_mae - mae) / baseline_mae * 100:.1f}%")
        print()
    
    print("CONCLUSION:")
    if mae < 0.5:
        print("  EXCELLENT: MAE < 0.5 - Outstanding performance")
    elif mae < 1.0:
        print("  VERY GOOD: 0.5 <= MAE < 1.0 - Very good performance")
    elif mae < 1.5:
        print("  GOOD: 1.0 <= MAE < 1.5 - Good performance")
    elif mae < 2.0:
        print("  ACCEPTABLE: 1.5 <= MAE < 2.0 - Acceptable but could be improved")
    else:
        print("  NEEDS IMPROVEMENT: MAE >= 2.0 - Significant improvement needed")
    
    print()
    print("For context, SIRENA-КБР production models typically target MAE < 0.4")

if __name__ == "__main__":
    focus_backtest()