"""
Иерархическое прогнозирование ИПЦ КБР
=====================================
Bottom-up: прогноз субкомпонентов -> агрегация в ИПЦ

Версия: 1.0
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')


class HierarchicalForecaster:
    """Иерархическое прогнозирование ИПЦ по субкомпонентам"""
    
    def __init__(self, ets_weight=0.9):
        self.ets_weight = ets_weight
        self.models = {}
        self.weights = None
        
    def load_data(self):
        """Загрузка данных"""
        # Агрегированные ряды субкомпонентов
        self.subcomp = pd.read_csv('data/subcomp_aggregated.csv', index_col=0)
        self.subcomp.index = pd.PeriodIndex(self.subcomp.index, freq='M')
        
        # Веса субкомпонентов
        sprav = pd.read_csv('data/micro_sprav.csv', sep=';', encoding='utf-8-sig')
        sprav['Weight_num'] = sprav['Weight'].astype(str).str.replace(',', '.').astype(float)
        sprav['Субкомпонент'] = sprav['Субкомпонент'].fillna('Услуги_прочее')
        self.weights = sprav.groupby('Субкомпонент')['Weight_num'].sum()
        
        # Общий ИПЦ для сравнения
        data = pd.read_csv('data/kbr_micro_full.csv')
        data['Date'] = pd.to_datetime(data['Day'], format='%m/%d/%y %H:%M:%S')
        data['YearMonth'] = data['Date'].dt.to_period('M')
        # Item_code=1 — общий ИПЦ
        ipc = data[data['Item_code'] == 1].set_index('YearMonth')['MoM'] - 100
        self.ipc = ipc.sort_index()
        
        print(f"Загружено: {len(self.subcomp.columns)} субкомпонентов, {len(self.subcomp)} месяцев")
        
    def create_features(self, series, train_end_idx):
        """Создание признаков для Ridge"""
        X, y = [], []
        for i in range(12, train_end_idx):
            # Лаги 1, 2, 3, 12
            features = [
                series.iloc[i-1],   # lag1
                series.iloc[i-2],   # lag2
                series.iloc[i-3],   # lag3
                series.iloc[i-12],  # lag12 (сезонность)
            ]
            X.append(features)
            y.append(series.iloc[i])
        return np.array(X), np.array(y)
    
    def forecast_subcomponent(self, name, train_end_idx):
        """Прогноз одного субкомпонента"""
        series = self.subcomp[name].dropna()
        
        if len(series) < train_end_idx or train_end_idx < 24:
            return None
            
        # Ridge модель
        X, y = self.create_features(series, train_end_idx)
        if len(X) < 10:
            return None
            
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        
        # Прогноз на 1 шаг
        last_features = [
            series.iloc[train_end_idx-1],
            series.iloc[train_end_idx-2],
            series.iloc[train_end_idx-3],
            series.iloc[train_end_idx-12],
        ]
        ridge_pred = model.predict([last_features])[0]
        
        # ETS сезонность
        try:
            monthly_means = series.iloc[:train_end_idx].groupby(series.index[:train_end_idx].month).mean()
            next_month = series.index[train_end_idx].month
            ets_seasonal = monthly_means.get(next_month, 0)
        except:
            ets_seasonal = 0
            
        # Ансамбль
        forecast = (1 - self.ets_weight) * ridge_pred + self.ets_weight * ets_seasonal
        
        return forecast
        
    def aggregate_forecast(self, subcomp_forecasts):
        """Агрегация прогнозов субкомпонентов в ИПЦ"""
        total = 0
        total_weight = 0
        for name, fc in subcomp_forecasts.items():
            if fc is not None and name in self.weights.index:
                w = self.weights[name]
                total += fc * w
                total_weight += w
        if total_weight > 0:
            return total / total_weight
        return None
        
    def backtest(self, start_date='2024-01', end_date='2025-10'):
        """Rolling backtest"""
        results = []
        
        start_idx = self.subcomp.index.get_loc(pd.Period(start_date, 'M'))
        end_idx = self.subcomp.index.get_loc(pd.Period(end_date, 'M'))
        
        print(f"\nBacktest: {start_date} — {end_date} ({end_idx - start_idx + 1} месяцев)")
        
        for idx in range(start_idx, end_idx + 1):
            period = self.subcomp.index[idx]
            
            # Прогноз каждого субкомпонента
            subcomp_fc = {}
            for col in self.subcomp.columns:
                subcomp_fc[col] = self.forecast_subcomponent(col, idx)
            
            # Агрегация
            ipc_forecast = self.aggregate_forecast(subcomp_fc)
            
            # Факт
            if period in self.ipc.index:
                actual = self.ipc[period]
            else:
                actual = None
                
            results.append({
                'period': str(period),
                'forecast': ipc_forecast,
                'actual': actual,
                'error': abs(ipc_forecast - actual) if (ipc_forecast and actual) else None
            })
            
        self.results = pd.DataFrame(results)
        return self.results
        
    def evaluate(self):
        """Оценка качества"""
        valid = self.results.dropna()
        mae = valid['error'].mean()
        rmse = np.sqrt((valid['error']**2).mean())
        
        # Hit rate (направление)
        hits = ((valid['forecast'] > 0) == (valid['actual'] > 0)).sum()
        hit_rate = hits / len(valid)
        
        print(f"\n=== РЕЗУЛЬТАТЫ ===")
        print(f"MAE: {mae:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"Hit Rate: {hit_rate:.1%}")
        print(f"Месяцев: {len(valid)}")
        
        return {'mae': mae, 'rmse': rmse, 'hit_rate': hit_rate}


if __name__ == '__main__':
    fc = HierarchicalForecaster(ets_weight=0.9)
    fc.load_data()
    fc.backtest('2024-01', '2025-10')
    fc.evaluate()
    
    print("\n=== ДЕТАЛИ ===")
    print(fc.results.to_string(index=False))
