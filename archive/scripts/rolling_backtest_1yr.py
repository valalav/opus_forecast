import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sirena_kbr_v2_4_auto import SirenaKBR_v24
from sirena_bvar import BayesianVAR
from sirena_arima import SirenaARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

def run_rolling_backtest_1yr():
    print("="*70)
    print("ROLLING BACKTEST (1 YEAR, 2-MONTH HORIZON)")
    print("="*70)
    
    # --- 1. Data Loading ---
    # Ridge Data (Aggregates)
    try:
        df_ridge_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')
        # Fix dates
        if 'Day' in df_ridge_raw.columns:
             df_ridge_raw['Date'] = pd.to_datetime(df_ridge_raw['Day'], format='%d.%m.%Y', errors='coerce')
        elif 'Date' in df_ridge_raw.columns:
             df_ridge_raw['Date'] = pd.to_datetime(df_ridge_raw['Date'], errors='coerce')
        if df_ridge_raw['Date'].isna().any():
             df_ridge_raw['Date'] = pd.to_datetime(df_ridge_raw['Date'], errors='coerce')
             
        # Fix MoM numeric format
        if 'MoM' in df_ridge_raw.columns:
             if df_ridge_raw['MoM'].dtype == object:
                 df_ridge_raw['MoM'] = df_ridge_raw['MoM'].astype(str).str.replace(',', '.')
             df_ridge_raw['MoM'] = pd.to_numeric(df_ridge_raw['MoM'], errors='coerce')

        if 'Товар' in df_ridge_raw.columns and 'MoM' in df_ridge_raw.columns:
             df_ridge = df_ridge_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
        else:
             df_ridge = df_ridge_raw.set_index('Date')
        
        df_ridge = df_ridge.sort_index()
    except Exception as e:
        print(f"Error loading Ridge data: {e}")
        return

    # BVAR Data (Macro)
    try:
        df_bvar = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
        cols_to_fix = ['mom', 'Prod', 'Nonprod', 'Serv', 'usd_nom_i', 'Ruonia']
        for col in cols_to_fix:
            if col in df_bvar.columns:
                if df_bvar[col].dtype == object:
                    df_bvar[col] = df_bvar[col].astype(str).str.replace(',', '.')
                df_bvar[col] = pd.to_numeric(df_bvar[col], errors='coerce')
        
        df_bvar['Date'] = pd.to_datetime(df_bvar['Date'], format='%d.%m.%Y', errors='coerce')
        if df_bvar['Date'].isna().any(): df_bvar['Date'] = pd.to_datetime(df_bvar['Date'])
        # Normalize to start of month
        df_bvar['Date'] = df_bvar['Date'].dt.to_period('M').dt.to_timestamp()
        df_bvar = df_bvar.set_index('Date').sort_index()
        
        bvar_data_full = pd.DataFrame()
        bvar_data_full['CPI'] = df_bvar['mom'] - 100
        bvar_data_full['Food'] = df_bvar['Prod'] - 100
        bvar_data_full['NonFood'] = df_bvar['Nonprod'] - 100
        bvar_data_full['Services'] = df_bvar['Serv'] - 100
        bvar_data_full['USD'] = df_bvar['usd_nom_i'] - 100
        bvar_data_full['RUONIA'] = df_bvar['Ruonia']
        bvar_data_full = bvar_data_full.dropna()
        
    except Exception as e:
        print(f"Error loading BVAR data: {e}")
        return

    # Ensure alignment
    common_idx = df_ridge.index.intersection(bvar_data_full.index)
    last_fact_date = common_idx.max()
    print(f"Последняя дата с фактом: {last_fact_date.strftime('%Y-%m-%d')}")
    
    # --- 2. Backtest Loop ---
    # We want to go back 12 months.
    # Cutoffs: last_fact - 1 month, ..., last_fact - 12 months.
    
    results = []
    
    for i in range(1, 13):
        cutoff = last_fact_date - pd.DateOffset(months=i)
        target_1m = cutoff + pd.DateOffset(months=1)
        target_2m = cutoff + pd.DateOffset(months=2)
        
        print(f"Cutoff: {cutoff.strftime('%Y-%m')}. Targets: {target_1m.strftime('%Y-%m')}, {target_2m.strftime('%Y-%m') if target_2m <= last_fact_date else 'N/A'}")
        
        # Train Data
        train_r = df_ridge[df_ridge.index <= cutoff].copy()
        train_b = bvar_data_full[bvar_data_full.index <= cutoff].copy()
        ts_ar = (train_r['Все товары и услуги'] - 100).dropna()
        
        # --- Predictions ---
        horizon = 2
        
        # 1. Ridge
        try:
            model_ridge = SirenaKBR_v24()
            model_ridge.fit(train_r)
            # Extend index for prediction
            fc_dates = [target_1m, target_2m]
            train_r_ext = train_r.copy()
            for d in fc_dates:
                if d not in train_r_ext.index: train_r_ext.loc[d] = np.nan
            
            # Predict Horizon 2
            # Using predict_horizon from v2.4 auto
            fc_r_df = model_ridge.predict_horizon(train_r, start_date=target_1m, horizon=horizon)
            path_r = fc_r_df['MoM'].values # This is Index - 100 based on the return of predict_horizon in auto file
            # Wait, predict_horizon in v2_4_auto returns 'MoM' column which is val - 100. Correct.
        except Exception as e:
            print(f"  Ridge failed: {e}")
            path_r = [np.nan]*horizon

        # 2. BVAR
        try:
            model_bvar = BayesianVAR(train_b, ['CPI', 'Food', 'USD', 'RUONIA'], lags=2)
            model_bvar.fit(lambda1=0.5)
            fc_b = model_bvar.forecast(h=horizon, n_draws=2000)
            path_b = fc_b['median'][:, 0] # CPI MoM
        except Exception as e:
            print(f"  BVAR failed: {e}")
            path_b = [np.nan]*horizon

        # 3. SARIMA
        try:
            model_sarima = SirenaARIMA()
            model_sarima.fit_sarima(ts_ar)
            fc_s = model_sarima.forecast(steps=horizon)
            path_s = fc_s['mean'].values
        except Exception as e:
            print(f"  SARIMA failed: {e}")
            path_s = [np.nan]*horizon
            
        # 4. Ensemble
        # Weights: Ridge 0.6, BVAR 0.3, SARIMA 0.1
        path_e = []
        for k in range(horizon):
            val = 0
            w_sum = 0
            if not np.isnan(path_r[k]): val += 0.6 * path_r[k]; w_sum += 0.6
            if not np.isnan(path_b[k]): val += 0.3 * path_b[k]; w_sum += 0.3
            if not np.isnan(path_s[k]): val += 0.1 * path_s[k]; w_sum += 0.1
            path_e.append(val / w_sum if w_sum > 0 else np.nan)
            
        # --- Evaluation ---
        # Check Actuals
        for h, target_date in enumerate([target_1m, target_2m]):
            if target_date <= last_fact_date:
                actual_idx = df_ridge.loc[target_date, 'Все товары и услуги']
                actual_mom = actual_idx - 100
                
                results.append({
                    'cutoff': cutoff,
                    'horizon': h + 1,
                    'target_date': target_date,
                    'Actual': actual_mom,
                    'Ridge': path_r[h],
                    'BVAR': path_b[h],
                    'SARIMA': path_s[h],
                    'Ensemble': path_e[h]
                })

    # --- 3. Analysis ---
    res_df = pd.DataFrame(results)
    
    print("\nРЕЗУЛЬТАТЫ ПО ГОРИЗОНТАМ:")
    print("-" * 60)
    
    metrics = []
    models = ['Ridge', 'BVAR', 'SARIMA', 'Ensemble']
    
    for h in [1, 2]:
        sub = res_df[res_df['horizon'] == h]
        if sub.empty: continue
        print(f"\nГоризонт {h} мес (N={len(sub)}):")
        
        row_mae = {'Horizon': h, 'Metric': 'MAE'}
        row_rmse = {'Horizon': h, 'Metric': 'RMSE'}
        
        for m in models:
            mae = mean_absolute_error(sub['Actual'], sub[m].fillna(0))
            rmse = np.sqrt(mean_squared_error(sub['Actual'], sub[m].fillna(0)))
            print(f"  {m}: MAE={mae:.4f}, RMSE={rmse:.4f}")
            
            row_mae[m] = mae
            row_rmse[m] = rmse
        
        metrics.append(row_mae)
        metrics.append(row_rmse)
        
    # Plotting
    plt.figure(figsize=(14, 7))
    
    # Plot Actuals
    unique_dates = res_df['target_date'].unique()
    unique_dates = np.sort(unique_dates)
    actuals = [res_df[res_df['target_date'] == d]['Actual'].iloc[0] for d in unique_dates]
    
    plt.plot(unique_dates, actuals, 'k-o', label='Факт', linewidth=2)
    
    # Plot 1-month ahead forecasts (continuous line approx)
    h1 = res_df[res_df['horizon'] == 1].sort_values('target_date')
    plt.plot(h1['target_date'], h1['Ensemble'], 'b--x', label='Ансамбль (1 мес)')
    
    plt.title('Rolling Backtest: Факт vs Прогноз (1 мес вперёд)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('rolling_backtest_1yr.png')
    
    # Save CSV
    res_df.to_csv('rolling_backtest_1yr_results.csv', index=False)
    print("\nФайлы сохранены: rolling_backtest_1yr_results.csv, rolling_backtest_1yr.png")

if __name__ == "__main__":
    run_rolling_backtest_1yr()
