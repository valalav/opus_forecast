import pandas as pd
import numpy as np
from sirena_micro_regions import SirenaMicroRegions
from sirena_kbr_v2_4_auto import SirenaKBR_v24
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

def run_micro_backtest():
    print("="*70)
    print("БЭКТЕСТ МИКРО-ПРЕДИКТОРОВ (Micro-Regions Approach)")
    print("="*70)
    
    # 1. Setup
    sm = SirenaMicroRegions()
    sm.load_data()
    
    # Load KBR actuals (Region 7)
    # We need full index history for evaluation
    # Using data from sm.data (all_regions_micro) is sufficient for specific items
    # But for TOTAL CPI validation we need infl_kbr.csv
    
    df_agg_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')
    if 'Day' in df_agg_raw.columns:
         df_agg_raw['Date'] = pd.to_datetime(df_agg_raw['Day'], format='%d.%m.%Y', errors='coerce')
    elif 'Date' in df_agg_raw.columns:
         df_agg_raw['Date'] = pd.to_datetime(df_agg_raw['Date'], errors='coerce')
         
    if 'Товар' in df_agg_raw.columns and 'MoM' in df_agg_raw.columns:
         df_agg = df_agg_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
    else:
         df_agg = df_agg_raw.set_index('Date')
    
    df_agg = df_agg.sort_index()
    
    # 2. Identify Predictors (Train on history < 2023)
    print("Обучение: поиск предикторов на данных до 2023 года...")
    # Filter sm.data temporarily
    full_data = sm.data.copy()
    train_data = sm.data[sm.data['Date'] < '2023-01-01']
    sm.data = train_data
    
    predictors = sm.find_micro_predictors(7) # KBR
    print("Найдено предикторов:")
    print(predictors)
    
    # Restore full data
    sm.data = full_data
    
    # 3. Backtest Loop (2023-2025)
    test_dates = pd.date_range('2023-01-01', '2025-10-01', freq='MS')
    
    # Baseline Model (Ridge)
    baseline_model = SirenaKBR_v24()
    
    results = []
    
    for date in test_dates:
        cutoff = date - pd.DateOffset(months=1)
        
        # --- Baseline Prediction ---
        train_r = df_agg[df_agg.index <= cutoff].copy()
        train_r_ext = train_r.copy()
        train_r_ext.loc[date] = np.nan
        
        try:
            baseline_model.fit(train_r)
            pred_base = baseline_model.predict(train_r_ext, date)['prediction']
        except:
            pred_base = np.nan
            
        # --- Micro-Predictor Prediction ---
        # Logic: 
        # 1. Take Lag 1 growth of Predictor Region for Item X.
        # 2. Use this as forecast for Item X in KBR.
        # 3. Aggregate item forecasts into CPI (using weights).
        # 4. Fill missing items with Baseline forecast (trend).
        
        # Weights (Approx for KBR 2024)
        weights = {
            'Meat': 0.08, 'Milk': 0.06, 'FruitsVeg': 0.05, 'Fuel': 0.05,
            'Sweets': 0.03, 'Shoes': 0.02, 'Utilities': 0.10, 'Construction': 0.02
        }
        
        weighted_growth = 0
        total_w = 0
        
        for _, row in predictors.iterrows():
            item = row['Товар']
            leader_reg = row['Регион-Лидер']
            # Find Leader's value at T-1 (cutoff)
            leader_code = next((k for k, v in sm.regions_map.items() if v == leader_reg), None)
            
            val = sm.data[
                (sm.data['Date'] == cutoff) & 
                (sm.data['Region_code'] == leader_code) & 
                (sm.data['Item'] == item)
            ]['MoM'].values
            
            if len(val) > 0:
                # Forecast for T is Leader(T-1)
                pred_item_growth = val[0] - 100
                w = weights.get(item, 0)
                weighted_growth += pred_item_growth * w
                total_w += w
        
        # Residual (Baseline)
        resid_w = 1.0 - total_w
        # Baseline is index (e.g. 100.5). Growth is 0.5.
        base_growth = pred_base - 100
        
        pred_micro = (weighted_growth + base_growth * resid_w) + 100
        
        actual = df_agg.loc[date, 'Все товары и услуги']
        
        results.append({
            'Date': date,
            'Actual': actual - 100,
            'Baseline': pred_base - 100,
            'Micro': pred_micro - 100
        })
        
    # 4. Metrics
    res_df = pd.DataFrame(results).dropna()
    mae_base = mean_absolute_error(res_df['Actual'], res_df['Baseline'])
    mae_micro = mean_absolute_error(res_df['Actual'], res_df['Micro'])
    
    print("\nРЕЗУЛЬТАТЫ (2023-2025):")
    print("-" * 40)
    print(f"Baseline (Ridge): MAE = {mae_base:.4f}")
    print(f"Micro-Predictors: MAE = {mae_micro:.4f}")
    print("-" * 40)
    
    if mae_micro < mae_base:
        print("✅ Гипотеза подтверждена! Микро-предикторы улучшают прогноз.")
    else:
        print("❌ Гипотеза не подтверждена. Baseline лучше.")

if __name__ == "__main__":
    run_micro_backtest()
