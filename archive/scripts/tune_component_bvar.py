import pandas as pd
import numpy as np
from sirena.models.bvar import BayesianVAR
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def optimize_components():
    # 1. Подготовка данных
    df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
    cols_to_fix = ['mom', 'Prod', 'Nonprod', 'Serv', 'usd_nom_i']
    for col in cols_to_fix:
        if col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
    if df['Date'].isna().any(): df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    
    # Центрируем данные (как в модели)
    data = df.copy()
    for c in cols_to_fix:
        if data[c].mean() > 50:
            data[c] = data[c] - 100

    # 2. Конфигурация
    components = {
        'Prod': ['Prod', 'usd_nom_i'],
        'Nonprod': ['Nonprod', 'usd_nom_i'],
        'Serv': ['Serv']
    }
    
    # Сетка параметров (Grid Search)
    param_grid = []
    for lags in [1, 2, 3, 4, 6]:
        for l1 in [0.1, 0.5, 1.0, 2.0]:
            for l2 in [0.2, 0.5, 1.0]: # Cross-variable tightness
                param_grid.append({'lags': lags, 'lambda1': l1, 'lambda2': l2})
    
    best_params = {}
    
    # Валидация: последние 24 месяца
    test_dates = data.index[-24:]
    
    print(f"Запуск оптимизации на {len(test_dates)} точках валидации...")
    
    for comp_name, cols in components.items():
        print(f"\n--- Оптимизация {comp_name} ---")
        best_mae = float('inf')
        best_cfg = None
        
        # Данные для компонента
        comp_data = data[cols].dropna()
        
        for params in param_grid:
            errors = []
            
            for date in test_dates:
                cutoff = date - pd.DateOffset(months=1)
                train = comp_data[comp_data.index <= cutoff]
                
                if len(train) < 24: continue
                
                try:
                    # Создаем модель
                    model = BayesianVAR(
                        lags=params['lags'], 
                        lambda1=params['lambda1'], 
                        lambda2=params['lambda2'],
                        var_names=cols
                    )
                    model.fit(train, target_col=cols[0])
                    
                    # Прогноз
                    fc = model.forecast(horizon=1) # 1D array
                    pred = fc[0]
                    actual = comp_data.loc[date, cols[0]]
                    
                    errors.append(abs(actual - pred))
                except:
                    continue
            
            if len(errors) > 0:
                mae = np.mean(errors)
                if mae < best_mae:
                    best_mae = mae
                    best_cfg = params
                    # print(f"New best for {comp_name}: {mae:.4f} with {params}")
        
        print(f"🏆 Лучшие параметры для {comp_name}:")
        print(f"MAE: {best_mae:.4f}")
        print(f"Config: {best_cfg}")
        best_params[comp_name] = best_cfg

    print("\n=== Итоговая конфигурация ===")
    print(best_params)

if __name__ == "__main__":
    optimize_components()
