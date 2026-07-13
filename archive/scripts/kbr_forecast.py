"""
СИРЕНА-КБР v2.2 (CHAMPION): Модель прогнозирования региональной инфляции
========================================================================

Финальная архитектура (по результатам 18 тестов):
- Модель: Ridge (Alpha=2.0)
- Фичи: Minimal (Seasonal Norm + Deviation + Component Lags)
- Сезонность: Global Mean (excl. outliers)
- MAE: 0.2798 | KPI: 87%

Инструкция по запуску:
1. Замените файлы в папке data/ на свои (см. FORMATS.md)
2. Запустите: python3 kbr_forecast.py
"""

import pandas as pd
import numpy as np
import os
import sys
from sklearn.linear_model import Ridge, QuantileRegressor
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class SirenaKBR:
    """
    Модель прогнозирования ИПЦ для Кабардино-Балкарской Республики.
    Версия 2.2: Оптимизированная "Minimal" архитектура.
    """
    
    def __init__(self, outlier_years=[2010, 2015, 2022]):
        self.outlier_years = outlier_years
        self.scaler = StandardScaler()
        self.ridge = None
        self.qr_models = {}
        self.seasonal = None
        self.is_fitted = False
        self.weekly_prices = None
        
        # CHAMPION FEATURE SET
        self.feature_cols = [
            'seasonal_norm', 
            'deviation_lag1', 
            'food_lag1', 
            'nonfood_lag1', 
            'services_lag1'
        ]
    
    def set_weekly_data(self, df_weekly):
        """Загрузка недельных данных для nowcasting."""
        self.weekly_prices = df_weekly
        
    def get_fv_yoy(self, year, month):
        """
        Расчет прироста цен на плодоовощи год к году (YoY) по недельным данным.
        """
        if self.weekly_prices is None:
            return None
            
        key_products = ['Картофель', 'Капуста', 'Лук', 'Морковь', 
                        'Огурцы', 'Помидоры', 'Яблоки', 'Бананы']
        yoys = []
        wp = self.weekly_prices
        
        try:
            current_prices = wp[(wp['year'] == year) & (wp['month'] == month)]
            prev_prices = wp[(wp['year'] == year - 1) & (wp['month'] == month)]
            
            if current_prices.empty or prev_prices.empty:
                return None
                
            for prod_keyword in key_products:
                curr = current_prices[current_prices['Товары'].str.contains(prod_keyword, case=False, na=False)]
                prev = prev_prices[prev_prices['Товары'].str.contains(prod_keyword, case=False, na=False)]
                
                if not curr.empty and not prev.empty:
                    p_cur = curr['Значение'].mean()
                    p_prev = prev['Значение'].mean()
                    if p_prev > 0:
                        yoys.append((p_cur / p_prev - 1) * 100)
        except:
            return None
                    
        return np.mean(yoys) if yoys else None
    
    def prepare_features(self, df):
        """Подготовка признаков для модели."""
        df = df.copy()
        
        df['month'] = df.index.month
        df['year'] = df.index.year
        
        # Базовые лаги
        df['y_lag1'] = df['Все товары и услуги'].shift(1)
        
        # Лаги компонент
        df['food_lag1'] = df['Продовольственные товары'].shift(1)
        df['nonfood_lag1'] = df['Непродовольственные товары'].shift(1)
        df['services_lag1'] = df['Услуги'].shift(1)
        
        # Расчет сезонной нормы (исключая годы-выбросы)
        clean = df[~df['year'].isin(self.outlier_years)]
        if len(clean) < 12: clean = df
            
        self.seasonal = clean.groupby('month')['Все товары и услуги'].mean()
        # Заполняем пропуски
        for m in range(1, 13):
            if m not in self.seasonal: self.seasonal[m] = 100.5
                
        df['seasonal_norm'] = df['month'].map(self.seasonal)
        df['deviation_lag1'] = df['y_lag1'] - df['month'].shift(1).map(self.seasonal)
        
        return df
    
    def fit(self, df, quantiles=[0.10, 0.90]):
        """Обучение модели."""
        print("Подготовка данных и обучение...")
        df = self.prepare_features(df)
        
        train_clean = df.dropna(subset=self.feature_cols + ['Все товары и услуги'])
        
        if len(train_clean) < 12:
            print(f"Внимание: мало данных для обучения ({len(train_clean)} точек).")
            
        X = train_clean[self.feature_cols].values
        y = train_clean['Все товары и услуги'].values
        
        if len(X) > 0:
            X_scaled = self.scaler.fit_transform(X)
            
            # Ridge c Alpha=2.0 (Champion setting)
            self.ridge = Ridge(alpha=2.0)
            self.ridge.fit(X_scaled, y)
            
            # Квантильная регрессия для интервалов
            try:
                for q in quantiles:
                    try:
                        qr = QuantileRegressor(quantile=q, alpha=0.0, solver='highs')
                    except:
                        qr = QuantileRegressor(quantile=q, alpha=0.0)
                    qr.fit(X_scaled, y)
                    self.qr_models[q] = qr
            except:
                self.qr_models = {}
            
            self.is_fitted = True
            print(f"Модель обучена. R2: {self.ridge.score(X_scaled, y):.3f}")
        else:
            print("ОШИБКА: Нет данных для обучения.")
            
        return self
    
    def predict_horizon(self, df, start_date, horizon=12, shocks=None):
        """Прогноз на несколько месяцев вперед."""
        if shocks is None: shocks = {}
        
        if not self.is_fitted:
            return pd.DataFrame()

        results = []
        df_work = df.copy()
        
        # История для рекурсии
        history = list(df_work['Все товары и услуги'].values)
        
        # Последние известные значения компонент (Naive extension)
        last_vals = {
            'food': df_work['Продовольственные товары'].iloc[-1],
            'nonfood': df_work['Непродовольственные товары'].iloc[-1],
            'services': df_work['Услуги'].iloc[-1]
        }
        
        print(f"\nРасчет прогноза на {horizon} месяцев (с {start_date.strftime('%Y-%m')}):")
        
        for i in range(horizon):
            target_date = start_date + pd.DateOffset(months=i)
            target_month = target_date.month
            target_year = target_date.year
            
            # Признаки
            current_seasonal = self.seasonal.get(target_month, 100.5)
            # Для deviation нужен лаг сезонности (прошлый месяц)
            prev_date = target_date - pd.DateOffset(months=1)
            prev_seasonal = self.seasonal.get(prev_date.month, 100.5)
            
            # deviation_lag1 = y_lag1 - seasonal_lag1
            current_deviation = history[-1] - prev_seasonal
            
            X_features = [
                current_seasonal,           # seasonal_norm
                current_deviation,          # deviation_lag1
                last_vals['food'],          # food_lag1
                last_vals['nonfood'],       # nonfood_lag1
                last_vals['services']       # services_lag1
            ]
            
            X = np.array([X_features])
            X_scaled = self.scaler.transform(X)
            
            # Прогноз
            pred = self.ridge.predict(X_scaled)[0]
            
            # Nowcasting (для первых 2 месяцев)
            fv_adj = 0
            if i < 2: 
                fv_yoy = self.get_fv_yoy(target_year, target_month)
                if fv_yoy is not None:
                    # Если инфляция плодоовощей отклоняется от 8%
                    fv_adj = 0.06 * ((fv_yoy - 8.0) / 12) * 0.5
                    if abs(fv_adj) > 0.01:
                        pred += fv_adj

            # Шоки
            shock_val = shocks.get(target_date, 0)
            pred += shock_val
            
            # Интервалы
            q10, q90 = pred - 0.3, pred + 0.3
            if self.qr_models:
                try:
                    diff = pred - self.ridge.predict(X_scaled)[0] # Сдвиг интервала
                    q10 = self.qr_models[0.10].predict(X_scaled)[0] + diff
                    q90 = self.qr_models[0.90].predict(X_scaled)[0] + diff
                except: pass

            results.append({
                'Дата': target_date.strftime('%d.%m.%Y'),
                'Месяц': target_month,
                'Прогноз (MoM)': round(pred, 2),
                'Интервал 80%': f"[{q10:.2f}; {q90:.2f}]",
                'Шок': shock_val
            })
            
            history.append(pred)
        
        return pd.DataFrame(results)

def load_data():
    print("Загрузка данных...")
    
    if not os.path.exists('data/infl_kbr.csv'):
        print("ОШИБКА: Файл data/infl_kbr.csv не найден.")
        sys.exit(1)
        
    try:
        # Читаем с поддержкой разных разделителей
        try:
            infl = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')
        except:
            infl = pd.read_csv('data/infl_kbr.csv', sep=';', decimal='.')
            
        infl['Date'] = pd.to_datetime(infl['Day'], format='%d.%m.%Y')
        # Нормализация к 1 числу
        infl['Date'] = infl['Date'].apply(lambda x: pd.Timestamp(year=x.year, month=x.month, day=1))
        
        if infl['MoM'].dtype == object:
             infl['MoM'] = pd.to_numeric(infl['MoM'].str.replace(',', '.'), errors='coerce')
        
        pivot = infl.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
        required = ['Все товары и услуги', 'Продовольственные товары', 
                   'Непродовольственные товары', 'Услуги'] # Без плодоовощей в обязательных, т.к. они только для nowcast
        
        df = pivot[required].copy()
    except Exception as e:
        print(f"ОШИБКА чтения infl_kbr.csv: {e}")
        sys.exit(1)

    # Недельные данные
    df_weekly = None
    if os.path.exists('data/weekly_prices.csv'):
        try:
            wp = pd.read_csv('data/weekly_prices.csv', sep=';', decimal=',') # Пробуем ; и ,
            if 'Товары' not in wp.columns: # Если не вышло, пробуем другой формат
                 wp = pd.read_csv('data/weekly_prices.csv', sep=';', decimal='.')
            
            # Парсинг года и недели
            if 'Сведено' in wp.columns:
                wp[['year', 'week']] = wp['Сведено'].str.split('_', expand=True)
                wp['year'] = wp['year'].astype(int)
                wp['week'] = wp['week'].astype(int)
                wp['month'] = pd.to_datetime(wp['year'].astype(str) + wp['week'].astype(str) + '1', 
                                           format='%Y%W%w').dt.month
                
                if wp['Значение'].dtype == object:
                    wp['Значение'] = pd.to_numeric(wp['Значение'].str.replace(',', '.'), errors='coerce')
                
                df_weekly = wp
                print("✅ Недельные данные загружены")
        except Exception as e:
            print(f"Внимание: Ошибка недельных данных ({e})")
    
    return df, df_weekly

if __name__ == "__main__":
    print("="*60)
    print("ЗАПУСК СИРЕНА-КБР v2.2 (CHAMPION)")
    print("="*60)
    
    df, df_weekly = load_data()
    
    model = SirenaKBR()
    if df_weekly is not None:
        model.set_weekly_data(df_weekly)
        
    model.fit(df)
    
    last_date = df.index.max()
    print(f"Последняя дата данных: {last_date.strftime('%d.%m.%Y')}")
    
    start_forecast = last_date + pd.DateOffset(months=1)
    
    # === НАСТРОЙКИ ===
    shocks = {
        # pd.Timestamp('2026-07-01'): 0.5,
    }
    
    forecast = model.predict_horizon(df, start_forecast, horizon=12, shocks=shocks)
    
    print("\nРЕЗУЛЬТАТЫ ПРОГНОЗА:")
    print("-" * 65)
    print(forecast.to_string(index=False))
    print("-" * 65)
    
    total_infl = forecast['Прогноз (MoM)'].sum()
    print(f"\nСумма индексов: {total_infl:.2f}")
    
    forecast.to_csv('forecast_result.csv', sep=';', index=False, encoding='utf-8-sig')
    print(f"\nРезультат сохранен: forecast_result.csv")
