import pandas as pd
import numpy as np
from sirena_bvar import SirenaBVAR
from sirena_kbr_v2_4_auto import SirenaKBR_v24
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

def run_comparison_backtest():
    print("="*60)
    print("CPАВНИТЕЛЬНЫЙ БЭКТЕСТ: Ridge (v2.4) vs BVAR (v3.0)")
    print("="*60)
    
    # 1. Загрузка данных
    # Используем inflation_data.csv для BVAR (нужны экзогенные)
    # и infl_kbr.csv для Ridge (нужны агрегаты)
    
    # --- Data for BVAR ---
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
        
        # Prepare vars
        bvar_data = pd.DataFrame()
        bvar_data['CPI'] = df_bvar['mom'] - 100
        bvar_data['Food'] = df_bvar['Prod'] - 100
        bvar_data['NonFood'] = df_bvar['Nonprod'] - 100
        bvar_data['Services'] = df_bvar['Serv'] - 100
        bvar_data['USD'] = df_bvar['usd_nom_i'] - 100
        bvar_data['RUONIA'] = df_bvar['Ruonia']
        bvar_data = bvar_data.dropna()
        
    except Exception as e:
        print(f"Ошибка загрузки данных BVAR: {e}")
        return

    # --- Data for Ridge (v2.4) ---
    try:
        df_ridge_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal='.')
        if 'Day' in df_ridge_raw.columns:
             df_ridge_raw['Date'] = pd.to_datetime(df_ridge_raw['Day'], format='%d.%m.%Y')
        elif 'Date' in df_ridge_raw.columns:
             df_ridge_raw['Date'] = pd.to_datetime(df_ridge_raw['Date'])
             
        if 'Товар' in df_ridge_raw.columns and 'MoM' in df_ridge_raw.columns:
             df_ridge = df_ridge_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
        else:
             df_ridge = df_ridge_raw.set_index('Date')
        
        df_ridge = df_ridge.sort_index()
    except Exception as e:
        print(f"Ошибка загрузки данных Ridge: {e}")
        return

    # 2. Параметры теста
    # Period: 2022-01 to 2025-10 (Capture the crisis + recovery)
    start_test = pd.Timestamp('2022-01-01')
    end_test = bvar_data.index.max()
    
    test_dates = pd.date_range(start_test, end_test, freq='MS')
    print(f"Период: {start_test.strftime('%Y-%m')} — {end_test.strftime('%Y-%m')} ({len(test_dates)} точек)")
    
    results_ridge = []
    results_bvar = []
    
    model_ridge = SirenaKBR_v24()
    
    # BVAR Config
    bvar_lags = 2
    bvar_draws = 500 # Reduced for speed in this script
    
    for target_date in test_dates:
        print(f"Прогноз на {target_date.strftime('%Y-%m')}...")
        cutoff = target_date - pd.DateOffset(months=1)
        
        # --- Ridge Prediction ---
        try:
            train_r = df_ridge[df_ridge.index <= cutoff].copy()
            # Prepare dummy future row for predict method
            train_r_ext = train_r.copy()
            train_r_ext.loc[target_date] = np.nan
            
            model_ridge.fit(train_r)
            pred_r = model_ridge.predict(train_r_ext, target_date)
            results_ridge.append({
                'date': target_date,
                'actual': df_ridge.loc[target_date, 'Все товары и услуги'] - 100,
                'pred': pred_r['prediction'] - 100
            })
        except Exception as e:
            print(f"  Ridge error: {e}")

        # --- BVAR Prediction ---
        try:
            train_b = bvar_data[bvar_data.index <= cutoff].copy()
            if len(train_b) > 24:
                model_bvar = SirenaBVAR(lags=bvar_lags)
                model_bvar.prepare_data(train_b, variables=['CPI', 'Food', 'USD', 'RUONIA'])
                model_bvar.build_model(prior_tightness=0.2)
                model_bvar.train(draws=bvar_draws, tune=200) # Faster tuning
                
                # Forecast 1 step
                fc = model_bvar.forecast(horizon=1)
                # Median of CPI (index 0)
                pred_val = np.median(fc[:, 0, 0])
                
                results_bvar.append({
                    'date': target_date,
                    'actual': bvar_data.loc[target_date, 'CPI'],
                    'pred': pred_val
                })
        except Exception as e:
            print(f"  BVAR error: {e}")

    # 3. Метрики
    def calc_metrics(res_list, name):
        if not res_list: return
        df = pd.DataFrame(res_list)
        mae = mean_absolute_error(df['actual'], df['pred'])
        rmse = np.sqrt(mean_squared_error(df['actual'], df['pred']))
        print(f"\n{name}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        return df

    df_r = calc_metrics(results_ridge, "Ridge (v2.4)")
    df_b = calc_metrics(results_bvar, "BVAR (PyMC)")
    
    # 4. График
    if df_r is not None and df_b is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(df_r['date'], df_r['actual'], 'k-', label='Факт', lw=2)
        plt.plot(df_r['date'], df_r['pred'], 'b--', label='Ridge v2.4')
        plt.plot(df_b['date'], df_b['pred'], 'r--', label='BVAR')
        plt.title('Сравнение точности: Ridge vs BVAR (2022-2025)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('comparison_ridge_bvar.png')
        print("\nГрафик сохранен: comparison_ridge_bvar.png")

if __name__ == "__main__":
    run_comparison_backtest()
