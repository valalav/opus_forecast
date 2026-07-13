import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings('ignore')

class SirenaKBR_v3:
    """
    СИРЕНА-КБР v3.0: Компонентная модель (Component-wise Forecasting).
    
    Прогноз строится как взвешенная сумма прогнозов компонентов:
    - Базовая инфляция (Прод + Непрод + Услуги)
    - Плодоовощная продукция (сильная сезонность)
    - Топливо
    - ЖКУ (регулируемые тарифы)
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.seasonals = {}
        self.weights = {
            # Приблизительные веса в корзине КБР (2024/2025)
            'Продовольственные товары': 0.38,
            'Непродовольственные товары': 0.35,
            'Услуги': 0.27,
            # Вложенные веса (для компонента внутри группы)
            'Плодоовощная продукция': 0.05,  # ~13% от продовольствия
            'Топливо': 0.05,                 # ~14% от непродовольствия
            'ЖКУ': 0.10                      # ~37% от услуг
        }
        
        # Агрегаты для моделирования
        # Мы моделируем: 
        # 1. Плодоовощи (отдельно)
        # 2. ЖКУ (отдельно)
        # 3. Топливо (отдельно)
        # 4. Остальное Продовольствие (База Прод)
        # 5. Остальное Непродовольствие (База Непрод)
        # 6. Остальные Услуги (База Услуги)
        
        self.components = [
            'Плодоовощная продукция', 'ЖКУ', 'Топливо', 
            'База_Прод', 'База_Непрод', 'База_Услуги'
        ]
        
    def prepare_data(self, df_detailed):
        """
        Подготовка данных: выделение базовых компонентов.
        Вход: df с колонками ['Все товары...', 'Прод...', 'Непрод...', 'Услуги', 'Плодоовощи', 'ЖКУ', 'Топливо']
        """
        df = df_detailed.copy()
        
        # Рассчитываем "Базовые" части (вычитаем волатильные компоненты)
        # Формула: I_base = (I_total*W_total - I_vol*W_vol) / (W_total - W_vol)
        
        # 1. База Продовольствия
        w_prod = self.weights['Продовольственные товары']
        w_fruit = self.weights['Плодоовощная продукция']
        # Упрощенно (аддитивно для малых изменений, но лучше через индексы)
        # I_prod ≈ w_f/w_p * I_f + (1-w_f/w_p) * I_base
        # => I_base = (I_prod - (w_f/w_p)*I_f) / (1 - w_f/w_p)
        
        k_f = w_fruit / w_prod
        df['База_Прод'] = (df['Продовольственные товары'] - k_f * df['Плодоовощная продукция']) / (1 - k_f)
        
        # 2. База Непродовольствия
        w_non = self.weights['Непродовольственные товары']
        w_fuel = self.weights['Топливо']
        k_fuel = w_fuel / w_non
        df['База_Непрод'] = (df['Непродовольственные товары'] - k_fuel * df['Топливо']) / (1 - k_fuel)
        
        # 3. База Услуг
        w_serv = self.weights['Услуги']
        w_jku = self.weights['ЖКУ']
        k_jku = w_jku / w_serv
        df['База_Услуги'] = (df['Услуги'] - k_jku * df['ЖКУ']) / (1 - k_jku)
        
        return df

    def fit(self, df_detailed):
        df = self.prepare_data(df_detailed)
        
        for comp in self.components:
            # Обучаем отдельную Ridge-модель для каждого компонента
            
            # Фичи: Лаг1, Лаг12, Сезонность (sin/cos), Ма1
            data = pd.DataFrame(index=df.index)
            data['y'] = df[comp]
            data['y_lag1'] = df[comp].shift(1)
            data['y_lag12'] = df[comp].shift(12)
            data['month'] = df.index.month
            
            # Сезонная норма (для ETS)
            seasonal = data.groupby('month')['y'].mean()
            self.seasonals[comp] = seasonal
            
            data['seasonal'] = data['month'].map(seasonal)
            data['deviation'] = data['y_lag1'] - data['month'].shift(1).map(seasonal)
            
            data['sin'] = np.sin(2 * np.pi * data['month'] / 12)
            data['cos'] = np.cos(2 * np.pi * data['month'] / 12)
            
            data = data.dropna()
            
            if len(data) > 12:
                X = data[['y_lag1', 'y_lag12', 'seasonal', 'deviation', 'sin', 'cos']]
                y = data['y']
                
                scaler = RobustScaler()
                X_sc = scaler.fit_transform(X)
                
                model = Ridge(alpha=0.5)
                model.fit(X_sc, y)
                
                self.models[comp] = model
                self.scalers[comp] = scaler
        
        return self

    def predict_next(self, df_detailed, horizon=12, params=None):
        if params is None: params = {}
        
        df = self.prepare_data(df_detailed)
        results = []
        
        # История для каждого компонента
        history = {c: list(df[c].values) for c in self.components}
        
        start_date = df.index.max() + pd.DateOffset(months=1)
        
        for i in range(horizon):
            t_date = start_date + pd.DateOffset(months=i)
            t_m = t_date.month
            
            comp_preds = {}
            
            for comp in self.components:
                # Params
                y_lag1 = history[comp][-1]
                y_lag12 = history[comp][-12] if len(history[comp]) > 11 else 100.5
                
                cur_sea = self.seasonals[comp].get(t_m, 100.5)
                
                # Scenario adjustments (e.g., stronger seasonality for Fruits)
                if comp == 'Плодоовощная продукция':
                    strength = params.get('seasonality_strength', 1.0)
                    cur_sea = 100.5 + (cur_sea - 100.5) * strength
                
                prev_sea = self.seasonals[comp].get((t_date - pd.DateOffset(months=1)).month, 100.5)
                deviation = y_lag1 - prev_sea
                
                sin = np.sin(2 * np.pi * t_m / 12)
                cos = np.cos(2 * np.pi * t_m / 12)
                
                X = np.array([[y_lag1, y_lag12, cur_sea, deviation, sin, cos]])
                X_sc = self.scalers[comp].transform(X)
                
                pred_ridge = self.models[comp].predict(X_sc)[0]
                
                # Ансамбль Ridge + ETS
                # Для волатильных компонентов больше веса ETS
                w_ets = 0.3
                if comp in ['Плодоовощная продукция', 'ЖКУ']:
                    w_ets = 0.7
                
                pred = (1 - w_ets) * pred_ridge + w_ets * cur_sea
                
                # Сценарные добавки
                if comp == 'База_Прод':
                    pred += params.get('food_trend_adj', 0.0)
                
                comp_preds[comp] = pred
                history[comp].append(pred)
            
            # Агрегация в общий ИПЦ
            # CPI = w_p * I_p + w_np * I_np + w_s * I_s
            
            # Восстанавливаем группы
            # I_prod = k_f * I_fruit + (1-k_f) * I_base_prod
            w_p = self.weights['Продовольственные товары']
            k_f = self.weights['Плодоовощная продукция'] / w_p
            i_prod = k_f * comp_preds['Плодоовощная продукция'] + (1-k_f) * comp_preds['База_Прод']
            
            w_np = self.weights['Непродовольственные товары']
            k_fuel = self.weights['Топливо'] / w_np
            i_nonprod = k_fuel * comp_preds['Топливо'] + (1-k_fuel) * comp_preds['База_Непрод']
            
            w_s = self.weights['Услуги']
            k_jku = self.weights['ЖКУ'] / w_s
            i_serv = k_jku * comp_preds['ЖКУ'] + (1-k_jku) * comp_preds['База_Услуги']
            
            # Общий ИПЦ
            cpi = w_p * i_prod + w_np * i_nonprod + w_s * i_serv
            
            # Валютный шок (накладывается на Непрод и Продукты)
            fx = params.get('fx_shock_pct', 0.0)
            if fx > 0 and i < 6:
                cpi += fx * 0.02 # simplified pass-through
            
            results.append({
                'Date': t_date,
                'Month': t_date.strftime('%b %Y'),
                'MoM': cpi - 100.0,
                'MoM_Index': cpi,
                'Fruits': comp_preds['Плодоовощная продукция'] - 100,
                'Fuel': comp_preds['Топливо'] - 100,
                'Utilities': comp_preds['ЖКУ'] - 100,
                'Base_Food': comp_preds['База_Прод'] - 100
            })
            
        return pd.DataFrame(results)
