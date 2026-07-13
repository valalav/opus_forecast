import pandas as pd
import numpy as np
from sirena_bvar import SirenaBVAR
import warnings

warnings.filterwarnings('ignore')

def run_bvar_pipeline():
    print("="*60)
    print("ЗАПУСК СИРЕНА-BVAR (Bayesian VAR)")
    print("="*60)
    
    # 1. Загрузка данных
    try:
        df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
        
        # Ensure numeric conversion for key columns
        cols_to_fix = ['mom', 'Prod', 'Nonprod', 'Serv', 'usd_nom_i', 'Ruonia']
        for col in cols_to_fix:
            if col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
        # Fallback date format
        if df['Date'].isna().any():
             df['Date'] = pd.to_datetime(df['Date'])
             
        df = df.set_index('Date').sort_index()
        
        # Prepare variables for BVAR
        # We need CPI components and Exogenous
        # Available: mom (CPI), Prod, Nonprod, Serv, usd_nom_i, Ruonia
        # We convert indices to MoM growth % (Index - 100)
        
        data = pd.DataFrame()
        data['CPI'] = df['mom'] - 100
        data['Food'] = df['Prod'] - 100
        data['NonFood'] = df['Nonprod'] - 100
        data['Services'] = df['Serv'] - 100
        
        # USD and Ruonia: Use differencing or growth rates?
        # USD Index (usd_nom_i) is likely rate or index. Let's assume it's Index (100=stable).
        # If it's absolute rate (77.0), we should take log diff.
        # Looking at head output earlier: "usd_nom_i; 100,00; 99,20..." -> It's an Index.
        data['USD'] = df['usd_nom_i'] - 100
        
        # RUONIA is rate %. We can use it directly or diff.
        # Levels are better for BVAR if they are stationary, but rates are often I(1).
        # Let's use Levels for now as period is short/stable-ish? No, 2022 shock.
        data['RUONIA'] = df['Ruonia']
        
        data = data.dropna()
        
        # Outlier filtering (2022) - BVAR is sensitive to large shocks
        # Option: Dummy variable? Or just exclude?
        # For now, let's keep it but rely on Heavy Tails (StudentT)? 
        # Current impl uses Normal. Let's just train on post-2015 or exclude 2022.
        # Excluding rows breaks time structure for VAR.
        # Better: Train on full history but maybe separate regime?
        # Let's try full history first.
        
        print(f"Данные готовы: {len(data)} точек. Переменные: {data.columns.tolist()}")
        
    except Exception as e:
        print(f"Ошибка данных: {e}")
        return

    # 2. Инициализация BVAR
    bvar = SirenaBVAR(lags=2)
    
    # 3. Подготовка
    bvar.prepare_data(data, variables=['CPI', 'Food', 'USD', 'RUONIA'])
    # Reduced set for speed and stability
    
    # 4. Построение модели
    print("Построение байесовской модели...")
    bvar.build_model(prior_tightness=0.5)
    
    # 5. Обучение (MCMC)
    print("Запуск MCMC сэмплирования (это может занять время)...")
    # Reduced draws for quick test
    trace = bvar.train(draws=500, tune=500)
    
    # 6. Прогноз
    print("Генерация прогноза...")
    horizon = 12
    last_date = data.index.max()
    fc_samples = bvar.forecast(horizon=horizon)
    
    # 7. Результаты
    # Median forecast for CPI
    cpi_idx = bvar.var_names.index('CPI')
    cpi_paths = fc_samples[:, :, cpi_idx]
    cpi_median = np.median(cpi_paths, axis=0)
    
    dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')
    
    print("\nРЕЗУЛЬТАТЫ BVAR (Медиана):")
    for d, v in zip(dates, cpi_median):
        print(f"{d.strftime('%Y-%m')}: {v:+.2f}%")
        
    # Save plot
    fig = bvar.plot_fan_chart(fc_samples, 'CPI', start_date=dates[0])
    fig.savefig('bvar_forecast_fan.png')
    print("\nГрафик сохранен: bvar_forecast_fan.png")

if __name__ == "__main__":
    run_bvar_pipeline()
