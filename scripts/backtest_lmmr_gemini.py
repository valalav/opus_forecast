"""
Script to backtest LMMR models (Gemini vs Claude)
"""

import pandas as pd
import numpy as np
from sirena.models.lmmr import LMMRForecaster
try:
    from sirena.models.lmmr_claude import LMMRForecasterClaude
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    print("Warning: LMMRForecasterClaude not found.")

def load_data():
    # Load inflation data
    # infl_kbr.csv uses dots for decimals, YYYY-MM-DD format (so dayfirst=False)
    df_infl = pd.read_csv('data/infl_kbr.csv', sep=';', decimal='.', parse_dates=['Date'], index_col='Date')
    
    # Ensure MoM is numeric
    if 'MoM' in df_infl.columns:
        df_infl['MoM'] = pd.to_numeric(df_infl['MoM'], errors='coerce')
    
    # Pivot if necessary (Long to Wide)
    if 'Товар' in df_infl.columns and 'MoM' in df_infl.columns:
        df_infl = df_infl.pivot_table(index='Date', columns='Товар', values='MoM')
    
    # Load macro data
    try:
        # inflation_data.csv uses commas for decimals
        df_macro = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',', parse_dates=['Date'], dayfirst=True, index_col='Date')
        # Merge macro data
        # Note: inflation_data.csv might have different date alignment (end of month vs start of month)
        # Assuming both are MS or can be aligned.
        # Let's inspect indices first.
        # df_infl index is usually MS (01.01.2020)
        # df_macro index is usually Month End (31.01.2010)
        
        # Convert df_macro index to Month Start to match df_infl
        df_macro.index = df_macro.index + pd.offsets.MonthBegin(0) - pd.offsets.MonthBegin(1)
        # Wait, if 31.01.2010 -> MonthBegin(0) -> 01.02.2010? No.
        # 31.01.2010. to MS is 01.01.2010?
        # Let's just normalize to Month Start
        df_macro.index = df_macro.index.to_period('M').to_timestamp()
        
        # Join
        # Use only cols not in df_infl
        cols_to_use = df_macro.columns.difference(df_infl.columns)
        df = df_infl.join(df_macro[cols_to_use], how='left')
        
    except Exception as e:
        print(f"Error loading macro data: {e}")
        df = df_infl
        
    return df

def calculate_metrics(df_results):
    mae = df_results['error'].abs().mean()
    rmse = np.sqrt((df_results['error']**2).mean())
    mape = (df_results['error'].abs() / df_results['actual']).mean() * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def main():
    print("Loading data...")
    df = load_data()
    print(f"Data loaded: {len(df)} rows. Columns: {df.columns.tolist()}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Index head: {df.index[:5]}")
    
    start_date = '2023-01-01'
    print(f"Starting backtest from {start_date}...")
    
    # 1. Backtest Gemini LMMR
    print("\n--- Backtesting LMMR (Gemini) ---")
    model_gemini = LMMRForecaster()
    results_gemini = model_gemini.backtest(df, start_date=start_date)
    
    if not results_gemini.empty:
        metrics_gemini = calculate_metrics(results_gemini)
        print("Results LMMR (Gemini):")
        print(f"MAE:  {metrics_gemini['MAE']:.4f}")
        print(f"RMSE: {metrics_gemini['RMSE']:.4f}")
    else:
        print("No results for Gemini model.")
        
    # 2. Backtest Claude LMMR
    if CLAUDE_AVAILABLE:
        print("\n--- Backtesting LMMR (Claude) ---")
        try:
            model_claude = LMMRForecasterClaude()
            results_claude = model_claude.backtest(df, start_date=start_date)
            
            if not results_claude.empty:
                metrics_claude = calculate_metrics(results_claude)
                print("Results LMMR (Claude):")
                print(f"MAE:  {metrics_claude['MAE']:.4f}")
                print(f"RMSE: {metrics_claude['RMSE']:.4f}")
            else:
                print("No results for Claude model.")
        except Exception as e:
            print(f"Error running Claude model: {e}")

if __name__ == "__main__":
    main()
