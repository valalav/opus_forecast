import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# Import feature extraction logic classes
from sirena_beta import SirenaBeta
from sirena_momentum import SirenaMomentum

def run_backtest_v4():
    print("="*70)
    print("БЭКТЕСТ V4: ТЕСТИРОВАНИЕ НОВЫХ ФАКТОРОВ (ФЕДЕРАЛЬНЫЙ ТРЕНД + ИМПУЛЬС)")
    print("="*70)
    
    # 1. Load Data
    # Monthly Targets
    try:
        df_monthly = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')
        
        # Robust numeric cleaning
        if 'MoM' in df_monthly.columns:
             if df_monthly['MoM'].dtype == object:
                 df_monthly['MoM'] = df_monthly['MoM'].astype(str).str.replace(',', '.')
             df_monthly['MoM'] = pd.to_numeric(df_monthly['MoM'], errors='coerce')

        if 'Day' in df_monthly.columns:
             df_monthly['Date'] = pd.to_datetime(df_monthly['Day'], format='%d.%m.%Y', errors='coerce')
        elif 'Date' in df_monthly.columns:
             df_monthly['Date'] = pd.to_datetime(df_monthly['Date'], errors='coerce')
             
        if 'Товар' in df_monthly.columns and 'MoM' in df_monthly.columns:
             pivot_m = df_monthly.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
        else:
             pivot_m = df_monthly.set_index('Date')
             
        pivot_m = pivot_m.sort_index()
        target_series = pivot_m['Все товары и услуги'] - 100
        
        # Components for Baseline
        food = pivot_m['Продовольственные товары'] - 100
        nonfood = pivot_m['Непродовольственные товары'] - 100
        services = pivot_m['Услуги'] - 100
        
    except Exception as e:
        print(f"Error loading monthly data: {e}")
        return

    # 2. Generate New Features
    
    # Feature 1: Federal Lag (Russia CPI t-1)
    print("Генерация фактора: Федеральный лаг...")
    sb = SirenaBeta()
    sb.load_data() # Loads all_regions_indices.csv
    # sb.data has 'Russia' and 'KBR' columns (Growth %)
    # We need to align dates
    
    # Feature 2: Momentum (Weekly Acceleration)
    print("Генерация фактора: Недельный импульс...")
    sm = SirenaMomentum()
    sm.load_data() # Loads weekly_prices.csv
    momentum_df = sm.analyze_momentum() 
    # momentum_df has 'Month' (YYYY-MM), 'Acceleration'
    # Need to map 'Month' to Date index
    momentum_df['Date'] = pd.to_datetime(momentum_df['Month'] + '-01')
    momentum_series = momentum_df.set_index('Date')['Acceleration']
    
    # 3. Merge Data
    data = pd.DataFrame({'Target': target_series})
    data['Lag1'] = data['Target'].shift(1)
    data['Lag2'] = data['Target'].shift(2)
    data['Lag12'] = data['Target'].shift(12)
    data['Month'] = data.index.month
    
    # Baseline components
    data['Food_Lag1'] = food.shift(1)
    data['NonFood_Lag1'] = nonfood.shift(1)
    data['Services_Lag1'] = services.shift(1)
    
    # New Features
    # Russia CPI Lag 1
    if sb.data is not None:
        data['Russia_Lag1'] = sb.data['Russia'].shift(1)
    else:
        data['Russia_Lag1'] = np.nan
        
    # Momentum (Acceleration at t-1 predicts t? Or current month accel predicts current month?)
    # In SirenaMomentum: "Acceleration in month M predicts CPI in month M+1"
    # So momentum_series at index '2024-01-01' is the acceleration calculated from Jan weeks.
    # It predicts Feb CPI.
    # So for predicting Feb CPI (target at 2024-02-01), we use Momentum from Jan (2024-01-01).
    # So we need to shift(1)?
    # Wait, SirenaMomentum logic: `target = self.monthly.loc[date_next]`. 
    # It aligned accel(M) with CPI(M+1).
    # So if we join by index, we should align Accel(M) to Target(M+1).
    # Let's shift momentum by 1 to align with Target.
    data['Momentum_Lag1'] = momentum_series.shift(1) 
    
    # Clean
    data = data.dropna()
    
    # 4. Backtest Loop (2023-2025)
    test_dates = pd.date_range('2023-01-01', '2025-10-01', freq='MS')
    
    baseline_features = ['Lag1', 'Lag2', 'Lag12', 'Food_Lag1', 'NonFood_Lag1', 'Services_Lag1']
    new_features = baseline_features + ['Russia_Lag1', 'Momentum_Lag1']
    
    results = []
    
    scaler = RobustScaler()
    model = Ridge(alpha=0.3)
    
    for date in test_dates:
        if date not in data.index: continue
        
        train = data[data.index < date]
        test = data.loc[[date]]
        
        if len(train) < 36: continue
        
        # --- Baseline ---
        X_train_b = scaler.fit_transform(train[baseline_features])
        y_train = train['Target']
        model.fit(X_train_b, y_train)
        
        X_test_b = scaler.transform(test[baseline_features])
        pred_base = model.predict(X_test_b)[0]
        
        # --- V4 (New Features) ---
        X_train_v4 = scaler.fit_transform(train[new_features])
        model.fit(X_train_v4, y_train)
        
        X_test_v4 = scaler.transform(test[new_features])
        pred_v4 = model.predict(X_test_v4)[0]
        
        results.append({
            'Date': date,
            'Actual': test['Target'].values[0],
            'Baseline': pred_base,
            'V4': pred_v4
        })
        
    # 5. Metrics
    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("Нет результатов.")
        return

    mae_base = mean_absolute_error(res_df['Actual'], res_df['Baseline'])
    rmse_base = np.sqrt(mean_squared_error(res_df['Actual'], res_df['Baseline']))
    
    mae_v4 = mean_absolute_error(res_df['Actual'], res_df['V4'])
    rmse_v4 = np.sqrt(mean_squared_error(res_df['Actual'], res_df['V4']))
    
    print("\nРЕЗУЛЬТАТЫ (2023-2025):")
    print("-" * 40)
    print(f"Baseline (Ridge v2.4): MAE = {mae_base:.4f}, RMSE = {rmse_base:.4f}")
    print(f"V4 (Federal + Momentum): MAE = {mae_v4:.4f}, RMSE = {rmse_v4:.4f}")
    print("-" * 40)
    
    if mae_v4 < mae_base:
        imp = (mae_base - mae_v4) / mae_base * 100
        print(f"✅ Улучшение: +{imp:.1f}%")
    else:
        print("❌ Ухудшение. Новые факторы добавляют шум.")

if __name__ == "__main__":
    run_backtest_v4()
