import sys
import os
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(os.getcwd())

from logger import get_logger
from sirena.data_loader import get_data_loader
import sirena.models 
from sirena.models.registry import ModelRegistry

# Suppress warnings
warnings.filterwarnings("ignore")

logger = get_logger("LongRollingBacktest")

class LongRollingBacktest:
    def __init__(self):
        self.loader = get_data_loader()
        self.df = self.loader.load_monthly_kbr()
        self.target_col = 'Все товары и услуги'
        
        # Полный список моделей, включая "Шоковые" версии
        self.candidate_models = [
            'ridge',                # Standard Production
            'ridge_shock_dummies',  # "Smart" Ridge with shock flags
            'bvar',                 # Macro
            'sarima', 
            'lightgbm', 
            'prophet', 
            'ets', 
            'ebm',
            'ngboost_shock_macro',  # Previous Champion (if libs exist)
            'regime_switching'      # Experimental
        ]
        
        # Filter available models
        self.models = []
        print("Checking available models:")
        for m in self.candidate_models:
            if ModelRegistry.is_registered(m):
                try:
                    # Quick instantiation check
                    cls = ModelRegistry.get_class(m)
                    _ = cls()
                    self.models.append(m)
                    print(f"  [OK] {m}")
                except Exception as e:
                    print(f"  [FAIL] {m}: {e}")
            else:
                print(f"  [MISSING] {m}")
        
        # Initial weights (equal)
        self.weights = {m: 1.0/len(self.models) for m in self.models}
        
        # Storage
        self.results = []
        self.history_errors = {m: [] for m in self.models}

    def update_weights(self):
        """Update weights based on Inverse MSE of all previous errors."""
        inv_mse_sum = 0
        inv_mses = {}
        
        for m in self.models:
            errors = np.array(self.history_errors[m])
            if len(errors) == 0:
                inv_mses[m] = 1.0 
            else:
                # Calculate MSE
                mse = np.mean(errors**2)
                if mse < 1e-6: mse = 1e-6
                
                # Inverse weighting
                inv_mses[m] = 1.0 / mse
            
            inv_mse_sum += inv_mses[m]
            
        # Normalize
        for m in self.models:
            self.weights[m] = inv_mses[m] / inv_mse_sum

    def run(self):
        # Long period: Dec 2023 to Nov 2025 (24 months)
        dates_to_predict = pd.date_range(start='2023-12-01', end='2025-11-01', freq='MS')
        
        print(f"\n{'Month':<10} | {'Fact':<6} | {'Pred':<6} | {'Error':<6} | {'Best Model':<20}")
        print("-" * 70)
        
        for pred_date in dates_to_predict:
            train_cutoff = pred_date - pd.DateOffset(months=1)
            
            if pred_date not in self.df.index:
                continue
                
            fact_val = self.df.loc[pred_date, self.target_col] - 100
            
            month_results = {
                'Date': pred_date,
                'Fact': fact_val
            }
            
            # Prepare Training Data (Strictly up to cutoff)
            train_df = self.df[self.df.index <= train_cutoff].copy()
            
            ensemble_pred = 0
            best_model = "None"
            min_error = float('inf')
            
            # Predict
            for model_name in self.models:
                pred_val = np.nan
                try:
                    model_cls = ModelRegistry.get_class(model_name)
                    model_instance = model_cls()
                    
                    # Fit
                    try:
                        model_instance.fit(train_df, self.target_col)
                    except TypeError:
                         model_instance.fit(train_df)
                    
                    # Forecast
                    fc = model_instance.forecast(horizon=1)
                    
                    # Extract scalar
                    if isinstance(fc, pd.DataFrame):
                        if 'mean' in fc.columns: val = fc['mean'].iloc[0]
                        elif 'MoM' in fc.columns: val = fc['MoM'].iloc[0]
                        elif model_name in fc.columns: val = fc[model_name].iloc[0]
                        elif model_name.upper() in fc.columns: val = fc[model_name.upper()].iloc[0]
                        else: val = fc.select_dtypes(include=np.number).iloc[0,0]
                    elif isinstance(fc, dict):
                         val = fc.get('mean', [0])[0] if 'mean' in fc else fc.get('prediction', 0)
                    elif isinstance(fc, (np.ndarray, list)):
                         val = fc[0]
                    else:
                         val = float(fc)
                    
                    # Sanity Check
                    if abs(val) > 10.0: # Looser check
                        val = np.nan
                        
                    pred_val = val

                except Exception as e:
                    print(f"FAILED {model_name}: {str(e)}")
                    pred_val = np.nan
                
                # Store prediction
                month_results[f'{model_name}_pred'] = pred_val
                month_results[f'{model_name}_weight'] = self.weights[model_name]
                
                # Error tracking
                if not np.isnan(pred_val):
                    error = fact_val - pred_val
                    self.history_errors[model_name].append(error)
                    
                    if abs(error) < min_error:
                        min_error = abs(error)
                        best_model = model_name
                else:
                    self.history_errors[model_name].append(2.0) # Penalty
            
            # Weighted Ensemble Calculation
            ensemble_sum = 0
            weight_sum = 0
            for m in self.models:
                val = month_results.get(f'{m}_pred')
                if val is not None and not np.isnan(val):
                    ensemble_sum += val * self.weights[m]
                    weight_sum += self.weights[m]
            
            if weight_sum > 0:
                ensemble_pred = ensemble_sum / weight_sum
            else:
                ensemble_pred = 0
            
            month_results['Ensemble_Pred'] = ensemble_pred
            month_results['Ensemble_Error'] = fact_val - ensemble_pred
            month_results['Best_Model'] = best_model
            
            self.results.append(month_results)
            
            print(f"{pred_date.strftime('%Y-%m') :<10} | {fact_val:6.2f} | {ensemble_pred:6.2f} | {month_results['Ensemble_Error']:6.2f} | {best_model:<20}")
            
            self.update_weights()
            
        # Save Results
        res_df = pd.DataFrame(self.results)
        res_df.to_csv('docs/long_backtest_results.csv', index=False)
        
        w_df = res_df[['Date'] + [f'{m}_weight' for m in self.models]]
        w_df.to_csv('docs/long_backtest_weights.csv', index=False)

if __name__ == "__main__":
    bt = LongRollingBacktest()
    bt.run()
