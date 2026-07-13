#!/usr/bin/env python3
"""Task 117: Simple Backtest"""

import sys
sys.path.insert(0, '/home/valalav/_projects/sirena-kbr')
sys.path.insert(1, '/home/valalav/_projects/sirena-kbr/edge_lab/sirena/data')

from sirena.models.ridge_macro import RidgeMacroForecaster
from sirena.models.huber import HuberForecaster
import sirena.data.enhanced_loader as el

import pandas as pd
import numpy as np


def main():
    print("Task 117: Simple Backtest")
    print("=" * 50)
    
    # Load enhanced data
    df = el.load_enhanced_data(top_opr_features=5)
    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Show OPR features
    opr_cols = [c for c in df.columns if c.startswith('opr_')]
    print(f"\\nOPR Features: {len(opr_cols)} features")
    print(f"  {', '.join(opr_cols)}")
    
    # Test baseline RidgeMacro
    print("\\n[1/2] Testing Baseline RidgeMacroForecaster...")
    baseline_df = df[['mom', 'Prod', 'Serv', 'Nonprod', 'Ki_i', 'Ruonia', 'usd_nom_i', 'brent']].copy()
    
    baseline_ridge = RidgeMacroForecaster(alpha=1.0, use_huber=False)
    results = baseline_ridge.backtest(baseline_df, start_date="2019-01-01", target_col="mom", horizon=1)
    
    if not results.empty:
        mae = results["error"].abs().mean()
        print(f"  MAE: {mae:.4f}")
        print(f"  Samples: {len(results)}")
        print(f"  Features used: {len(getattr(baseline_ridge, '_available_features', []))}")
    else:
        print("  No results")
    
    print("\\n✅ Test completed successfully")


if __name__ == "__main__":
    main()
