import sys
sys.path.insert(0, '/home/valalav/_projects/sirena-kbr/edge_lab/sirena')

from sirena.models.ridge_macro import RidgeMacroForecaster
from sirena.models.huber import HuberForecaster

import pandas as pd
import numpy as np
from pathlib import Path

# Inline data loader
def load_enhanced_data(top_opr_features=5):
    """Load enhanced data with OPR features."""
    priority_path = "data/regressor_priority_list.csv"
    
    if not Path(priority_path).exists():
        raise FileNotFoundError(f"Priority list not found: {priority_path}")
    
    sectoral_path = "data/test_kbr_sectoral.csv"
    
    if not Path(sectoral_path).exists():
        raise FileNotFoundError(f"Sectoral data not found: {sectoral_path}")
    
    priority_df = pd.read_csv(priority_path)
    sectoral_df = pd.read_csv(sectoral_path)
    sectoral_df['Date'] = pd.to_datetime(sectoral_df['Date'])
    
    top_features = priority_df.head(top_opr_features)
    
    opr_features = []
    
    for _, row in top_features.iterrows():
        feature_id = row['feature_id']
        
        parts = feature_id.split('::')
        indicator = parts[0]
        metric_type = parts[1] if len(parts) > 1 else 'г/г'
        sheet = int(parts[2]) if len(parts) > 2 else 102
        
        matches = sectoral_df[
            (sectoral_df['Indicator'] == indicator) &
            (sectoral_df['Metric_Type'] == metric_type) &
            (sectoral_df['Sheet'] == sheet)
        ].copy()
        
        if len(matches) == 0:
            continue
        
        clean_name = f"opr_{feature_id.replace(';', '_').replace(' ', '_').replace('::', '_').replace('/', '_')}"
        clean_name = clean_name[:60]
        
        matches = matches[['Date', 'Value']].copy()
        matches = matches.groupby('Date')['Value'].first().reset_index()
        matches = matches.rename(columns={'Value': clean_name})
        
        opr_features.append(matches)
    
    if not opr_features:
        raise ValueError("No OPR features found")
    
    result = opr_features[0]
    for feat_df in opr_features[1:]:
        result = pd.merge(result, feat_df, on='Date', how='outer')
    
    result = result.set_index('Date')
    result = result.sort_index()
    
    return result


def main():
    print("Task 117: Simple Backtest")
    print("=" * 50)
    
    df = load_enhanced_data(top_opr_features=5)
    print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")
    
    opr_cols = [c for c in df.columns if c.startswith('opr_')]
    print(f"\\nOPR Features: {len(opr_cols)} features")
    print(f"  {', '.join(opr_cols)}")
    
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
    
    print("\\n[2/2] Testing Enhanced RidgeMacroForecaster...")
    
    enhanced_df = df.copy()
    for col in opr_cols:
        enhanced_df[f'{col}_L1'] = enhanced_df[col].shift(1)
    
    enhanced_ridge = RidgeMacroForecaster(alpha=1.0, use_huber=False)
    results_enh = enhanced_ridge.backtest(enhanced_df, start_date="2019-01-01", target_col="mom", horizon=1)
    
    if not results_enh.empty:
        mae_enh = results_enh["error"].abs().mean()
        print(f"  MAE: {mae_enh:.4f}")
        print(f"  Samples: {len(results_enh)}")
        
        features = getattr(enhanced_ridge, '_available_features', [])
        opr_used = sum(1 for f in features if f.startswith('opr_'))
        print(f"  Features used: {opr_used}")
    else:
        print("  No results")
    
    print("\\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if not results.empty and not results_enh.empty:
        target_mae = 0.236
        improvement = ((target_mae - mae_enh) / target_mae) * 100
        print(f"\\nTarget MAE: {target_mae:.3f}")
        print(f"Baseline RidgeMacro MAE: {mae:.4f}")
        print(f"Enhanced RidgeMacro MAE: {mae_enh:.4f}")
        print(f"\\nImprovement: {improvement:+.2f}%")
        print(f"\\nGoal (MAE < 0.22): {'ACHIEVED' if mae_enh < 0.22 else 'NOT ACHIEVED'}")
    
    print("\\n✅ Test completed successfully")


if __name__ == "__main__":
    main()
