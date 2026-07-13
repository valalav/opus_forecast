import pandas as pd
import numpy as np
from sirena_arima import SirenaARIMA
from sirena_kbr_v2_4_auto import SirenaKBR_v24
from sirena_bvar import BayesianVAR
from sirena_pipeline_v31 import SirenaEnsemble
import warnings

warnings.filterwarnings('ignore')

def compare_all():
    print("="*70)
    print("СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ (v3.1 Benchmark)")
    print("="*70)
    
    # Load Data
    try:
        df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
        # Robust loading
        cols_to_fix = ['mom', 'Prod', 'Nonprod', 'Serv', 'usd_nom_i', 'Ruonia']
        for col in cols_to_fix:
            if col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
        if df['Date'].isna().any(): df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        
        ts = (df['mom'] - 100).dropna()
        
        # 1. AR(1) Baseline
        ar_model = SirenaARIMA()
        ar_model.fit_ar1(ts)
        fc_ar = ar_model.forecast(1)
        
        # 2. SARIMA
        sarima_model = SirenaARIMA()
        sarima_model.fit_sarima(ts)
        fc_sarima = sarima_model.forecast(1)
        
        # 3. Ridge (v2.4)
        # Need infl_kbr.csv for this
        # Assuming it's available and loaded inside SirenaEnsemble
        
        # 4. Ensemble Forecast
        ensemble = SirenaEnsemble()
        ensemble.prepare_data('data/infl_kbr.csv', 'data/inflation_data.csv')
        fc_ensemble = ensemble.forecast(1)
        
        print("\nРЕЗУЛЬТАТЫ НА 1 МЕСЯЦ ВПЕРЕД (ДЕКАБРЬ 2025):")
        print(f"{'Модель':<15} {'Прогноз':>10} {'95% Interval'}")
        print("-" * 45)
        
        print(f"{'AR(1)':<15} {fc_ar['mean'].iloc[0]:>9.2f}% [{fc_ar['lower'].iloc[0]:.2f}, {fc_ar['upper'].iloc[0]:.2f}]")
        print(f"{'SARIMA':<15} {fc_sarima['mean'].iloc[0]:>9.2f}% [{fc_sarima['lower'].iloc[0]:.2f}, {fc_sarima['upper'].iloc[0]:.2f}]")
        
        # Ensemble components
        row = fc_ensemble.iloc[0]
        print(f"{'Ridge v2.5':<15} {row['Ridge']:>9.2f}% [ — ]")
        print(f"{'BVAR':<15} {row['BVAR']:>9.2f}% [ — ]")
        print(f"{'Ансамбль':<15} {row['Ensemble']:>9.2f}% [ — ]")
        
        print("\nСравнение AIC (In-sample fit):")
        print(f"AR(1): {fc_ar['aic']:.1f}")
        print(f"SARIMA: {fc_sarima['aic']:.1f}")
        
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    compare_all()