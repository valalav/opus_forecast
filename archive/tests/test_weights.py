"""
Тест влияния весов ETS и экзогенных на качество прогноза.
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных (длинный формат -> pivot)
df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',', encoding='utf-8')
df_raw['Date'] = pd.to_datetime(df_raw['Date'])
df = df_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
df = df.sort_index()

# Загрузка экзогенных
df_exog = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
df_exog['Date'] = pd.to_datetime(df_exog['Date'], dayfirst=True)
df_exog = df_exog.set_index('Date')

# Объединяем
df = df.join(df_exog[['usd_nom_i', 'Ki_i', 'Ruonia']], how='left')
df['usd_nom_i'] = df['usd_nom_i'].fillna(100)
df['Ruonia'] = df['Ruonia'].fillna(df['Ruonia'].mean())

print(f"Данные: {df.index.min().date()} - {df.index.max().date()}")
print(f"Строк: {len(df)}")

# Подготовка признаков
df['y'] = df['Все товары и услуги']
df['month'] = df.index.month
df['y_lag1'] = df['y'].shift(1)
df['y_lag12'] = df['y'].shift(12)

# Сезонная норма (без 2022, 2010)
clean = df[(df.index.year != 2022) & (df.index.year != 2010)]
seasonal_norm = clean.groupby('month')['y'].mean().to_dict()

df['seasonal_norm'] = df['month'].map(seasonal_norm)
df['deviation_lag1'] = df['y_lag1'] - df['month'].shift(1).map(seasonal_norm)

# Тригонометрическая сезонность
df['sin_m'] = np.sin(2 * np.pi * df['month'] / 12)
df['cos_m'] = np.cos(2 * np.pi * df['month'] / 12)

# Экзогенные лаги
df['usd_lag1'] = df['usd_nom_i'].shift(1)
df['ruonia_lag1'] = df['Ruonia'].shift(1)

df = df.dropna(subset=['y', 'y_lag1', 'y_lag12'])

# Признаки
FEATURES_BASE = ['y_lag1', 'y_lag12', 'sin_m', 'cos_m', 'seasonal_norm', 'deviation_lag1']
FEATURES_EXOG = FEATURES_BASE + ['usd_lag1', 'ruonia_lag1']

# Веса ETS
ETS_CONFIGS = {
    'original': {1: 0.9, 2: 0.0, 3: 0.5, 4: 0.3, 5: 0.9, 6: 0.5, 7: 0.0, 8: 0.5, 9: 0.9, 10: 0.9, 11: 0.0, 12: 0.0},
    'reduced':  {1: 0.5, 2: 0.0, 3: 0.3, 4: 0.2, 5: 0.5, 6: 0.3, 7: 0.0, 8: 0.3, 9: 0.5, 10: 0.5, 11: 0.0, 12: 0.0},
    'low':      {1: 0.3, 2: 0.0, 3: 0.2, 4: 0.1, 5: 0.3, 6: 0.2, 7: 0.0, 8: 0.2, 9: 0.3, 10: 0.3, 11: 0.0, 12: 0.0},
    'minimal':  {m: 0.1 for m in range(1,13)},
    'zero':     {m: 0.0 for m in range(1,13)},
}

def backtest(df, features, ets_weights, test_start='2024-01-01'):
    test_start = pd.Timestamp(test_start)
    results = []
    
    for test_date in df[df.index >= test_start].index:
        train = df[df.index < test_date].dropna(subset=features)
        if len(train) < 36:
            continue
        
        X_train = train[features].values
        y_train = train['y'].values
        
        scaler = RobustScaler()
        X_train_sc = scaler.fit_transform(X_train)
        
        ridge = Ridge(alpha=0.3)
        ridge.fit(X_train_sc, y_train)
        
        test_row = df.loc[[test_date]].dropna(subset=features)
        if len(test_row) == 0:
            continue
            
        X_test = test_row[features].values
        X_test_sc = scaler.transform(X_test)
        
        pred_ridge = ridge.predict(X_test_sc)[0]
        
        month = test_date.month
        pred_ets = seasonal_norm.get(month, 100.5)
        
        w_ets = ets_weights.get(month, 0.3)
        pred_combined = (1 - w_ets) * pred_ridge + w_ets * pred_ets
        
        actual = df.loc[test_date, 'y']
        error = pred_combined - actual
        
        results.append({
            'date': test_date,
            'actual': actual,
            'pred': pred_combined,
            'pred_ridge': pred_ridge,
            'pred_ets': pred_ets,
            'w_ets': w_ets,
            'error': error,
            'year': test_date.year
        })
    
    return pd.DataFrame(results)

print("\n" + "="*70)
print("ЭКСПЕРИМЕНТ: Влияние весов ETS и экзогенных")
print("="*70)

for feat_name, features in [('BASE', FEATURES_BASE), ('EXOG', FEATURES_EXOG)]:
    print(f"\n### {feat_name} ###")
    
    for ets_name, ets_weights in ETS_CONFIGS.items():
        res = backtest(df, features, ets_weights, '2024-01-01')
        if len(res) == 0:
            continue
        
        mae_all = res['error'].abs().mean()
        mae_2024 = res[res['year']==2024]['error'].abs().mean() if len(res[res['year']==2024]) > 0 else 0
        mae_2025 = res[res['year']==2025]['error'].abs().mean() if len(res[res['year']==2025]) > 0 else 0
        
        print(f"  {ets_name:12s}: MAE={mae_all:.4f}  |  2024={mae_2024:.4f}  2025={mae_2025:.4f}")

# Лучшая конфигурация
print("\n" + "="*70)
print("ДЕТАЛИ: EXOG + low weights")
print("="*70)

res = backtest(df, FEATURES_EXOG, ETS_CONFIGS['low'], '2024-01-01')

print("\nПо месяцам:")
for _, row in res.iterrows():
    mark = "**" if abs(row['error']) > 0.5 else ""
    print(f"  {row['date'].strftime('%Y-%m')}: факт={row['actual']:.2f} прогноз={row['pred']:.2f} err={row['error']:+.2f} {mark}")

print(f"\nBias 2024: {res[res['year']==2024]['error'].mean():+.3f}")
print(f"Bias 2025: {res[res['year']==2025]['error'].mean():+.3f}")
