#!/usr/bin/env python3
"""
СИРЕНА-КБР v3.0 - Пайплайн прогнозирования инфляции
Обновление: каждую неделю с новыми данными

Использование:
1. Обновить файл Сравнение_еженедельных_цен_01.csv
2. Проверить настройки в exog_settings.json
3. Запустить скрипт
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings('ignore')

from sirena_midas import SirenaMIDAS

class SirenaPipeline:
    def __init__(self):
        self.config = self._load_json('config.json')
        self.exog = self._load_json('exog_settings.json')
        
        self.weekly = None
        self.monthly = None
        self.model = None
        self.scaler = None
        self.seasonal = None
        self.weights = None # From weekly file
        self.midas = SirenaMIDAS() # Initialize MIDAS
        
        # Model Features (Champion v2.4)
        self.features = [
            'y_lag1', 'y_lag2', 'y_lag12', 'y_ma3',
            'month_sin', 'month_cos',
            'food_lag1', 'nonfood_lag1', 'services_lag1',
            'seasonal_norm', 'deviation_lag1'
        ]

    def _load_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_data(self):
        print("📥 Загрузка данных...")
        
        # 1. Weekly Data (Real Data Processing)
        try:
            # Load Raw Weekly
            w_path = 'data/weekly_prices.csv' # Hardcoded to real file
            raw_weekly = pd.read_csv(w_path, sep=';', decimal=',')
            
            # Pivot to Wide Format: Index=[Item, Code], Cols=Week
            self.weekly = raw_weekly.pivot_table(
                index=['Товары', 'Rostat_code'], 
                columns='Сведено', 
                values='Значение',
                aggfunc='mean'
            ).reset_index()
            
            # Load Weights & Mapping
            # We need: Rostat_code -> Item_code (Access) -> Weight
            try:
                items_df = pd.read_csv('data/items_names.csv')
                weights_df = pd.read_csv('data/access_weights.csv')
                
                # Clean weights (latest date)
                weights_df['Day'] = pd.to_datetime(weights_df['Day'], format='%m/%d/%y %H:%M:%S', errors='coerce')
                latest_w_date = weights_df['Day'].max()
                current_weights = weights_df[weights_df['Day'] == latest_w_date][['Item_code', 'Weight_vertical']]
                
                # Merge chain: Weekly(Rostat) -> Items(Rosstat->Code) -> Weights(Code->Weight)
                # Items file: Item_code, Item_name, Item_rosstat
                
                # Prepare mapping df
                mapping = items_df[['Item_code', 'Item_rosstat']].dropna()
                mapping['Item_rosstat'] = pd.to_numeric(mapping['Item_rosstat'], errors='coerce')
                
                # Merge 1: Weekly + Mapping
                self.weekly['Rostat_code'] = pd.to_numeric(self.weekly['Rostat_code'], errors='coerce')
                merged = self.weekly.merge(mapping, left_on='Rostat_code', right_on='Item_rosstat', how='left')
                
                # Merge 2: + Weights
                merged = merged.merge(current_weights, on='Item_code', how='left')
                
                # Rename for compatibility
                self.weekly = merged.rename(columns={'Weight_vertical': 'Справка_нед.Вес', 'Товары': 'Наименование '})
                
                # Create weights dict (Weight is usually 0.0 to 1.0 or 0 to 100)
                # In Access export: "0.36513" (likely percent share of total 1.0? or 100?)
                # Example: Beef 0.7% -> 0.007? Or 0.7?
                # Check access_weights.csv output from step 29: "160000002... 0.36513". 
                # This looks like ~36% of something? Or 0.36%?
                # Non-food is 36%. So 0.36 is 36% of CPI.
                # So if a specific item has weight, it might be small (e.g. 0.007 for Beef).
                # We will treat it as "Share of 1". So multiply by 100 to get "Percent Weight".
                
                self.weights = self.weekly.set_index('Наименование ')['Справка_нед.Вес'].to_dict()
                
                # Fix nan weights
                self.weights = {k: (v * 100 if pd.notna(v) else 0) for k, v in self.weights.items()} 
                # Multiplied by 100 to match previous logic (e.g. 0.7 for Beef)
                
            except Exception as e:
                print(f"⚠️ Ошибка загрузки весов (использую равные): {e}")
                self.weights = {}

            print(f"   Недельные данные: {len(self.weekly)} товаров (с весами)")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки недельных данных: {e}")
            return False

        # 2. Monthly Data
        try:
            m_path = self.config['paths']['monthly_data']
            # Try parsing (Access export format vs standard)
            try:
                df = pd.read_csv(m_path, sep=';', decimal='.')
                if 'Day' in df.columns:
                    df['Date'] = pd.to_datetime(df['Day'], format='%d.%m.%Y')
                elif 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                
                # Pivot if long format
                if 'Товар' in df.columns and 'MoM' in df.columns:
                    df = df.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
                elif 'Все товары и услуги' not in df.columns:
                     # Maybe already wide but 'Date' index
                     df = df.set_index('Date')
                
                df = df.sort_index()
                self.monthly = df
                print(f"   Месячные данные: до {df.index.max().strftime('%Y-%m')}")
            except Exception as e:
                print(f"❌ Ошибка парсинга месячных данных: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка загрузки месячных данных: {e}")
            return False
            
        return True

    def calculate_nowcast(self, target_year, target_month, model_trend=0.0):
        """
        Nowcast using MIDAS (Bridge Equation) + Weighted Decomposition.
        """
        print(f"\n🔍 Расчет Nowcast на {target_month:02d}.{target_year}...")
        
        # --- 1. Decomposition (Weighted Average of Drivers) ---
        # Identify previous month
        if target_month == 1:
            prev_month, prev_year = 12, target_year - 1
        else:
            prev_month, prev_year = target_month - 1, target_year
            
        # Filter columns
        week_cols = [c for c in self.weekly.columns if '_' in c and c[0].isdigit()]
        curr_weeks = []
        prev_weeks = []
        
        for col in week_cols:
            try:
                y_str, w_str = col.split('_')
                w_date = datetime.fromisocalendar(int(y_str), int(w_str), 1)
                if w_date.year == target_year and w_date.month == target_month:
                    curr_weeks.append(col)
                elif w_date.year == prev_year and w_date.month == prev_month:
                    prev_weeks.append(col)
            except: pass
        
        if not curr_weeks:
            print("⚠️ Нет недельных данных за целевой месяц. Nowcast невозможен.")
            return None, None

        # Calculate Means & Growth for Decomposition
        df = self.weekly.copy()
        for c in curr_weeks + prev_weeks:
             df[c] = pd.to_numeric(df[c], errors='coerce')
             
        df['mean_curr'] = df[curr_weeks].mean(axis=1)
        df['mean_prev'] = df[prev_weeks].mean(axis=1) if prev_weeks else np.nan
        df['growth'] = (df['mean_curr'] / df['mean_prev'] - 1) * 100
        
        contribs = []
        basket_proxy_current = 0
        
        # Calculate weighted proxy manually for decomposition
        for idx, row in df.iterrows():
            w = row.get('Справка_нед.Вес', 0)
            if pd.isna(w): w = 0
            impact_pp = (row['growth'] * w) / 100
            if w > 0 and abs(row['growth']) > 0.01:
                contribs.append({'Товар': row['Наименование '], 'Рост': row['growth'], 'Вес': w, 'Вклад (п.п.)': impact_pp})
        
        # --- 2. MIDAS Prediction (Bridge Equation) ---
        try:
            print("   Обучение MIDAS (Bridge Model)...")
            # Prepare training data
            # Need Monthly Target DataFrame (cols: 'Все товары и услуги')
            monthly_target = self.monthly[['Все товары и услуги']].copy()
            
            midas_df = self.midas.prepare_data(self.weekly, monthly_target)
            r2 = self.midas.fit(midas_df)
            print(f"   MIDAS R2: {r2:.3f}")
            
            # Prepare Input for Prediction
            # 1. Weekly Proxy for Target Month
            # We computed 'mean_curr' above. We need 'Basket_Cost' proxy growth.
            # Aggregate df['mean_curr'] (sum of prices?) 
            # The MIDAS class does `groupby().mean().sum(axis=1)`.
            # Here we have `df` which is Wide (rows=items).
            # Sum of `mean_curr` is the Basket Cost for current month.
            basket_curr = df['mean_curr'].sum()
            
            # We need previous month basket cost from the *same* set of items
            # `df` has all items.
            basket_prev = df['mean_prev'].sum()
            
            if basket_prev > 0:
                current_proxy_mom = (basket_curr / basket_prev - 1) * 100
            else:
                current_proxy_mom = 0 # Fallback
                
            # 2. Target Lag
            # Last known monthly inflation
            last_target = self.monthly['Все товары и услуги'].iloc[-1] - 100
            
            final_nowcast = self.midas.predict(current_proxy_mom, last_target)
            print(f"   MIDAS Прогноз: {final_nowcast:+.2f}% (Proxy growth: {current_proxy_mom:+.2f}%)")
            
        except Exception as e:
            print(f"⚠️ Ошибка MIDAS: {e}. Использую взвешенное среднее.")
            # Fallback logic (Old weighted sum + residual)
            total_w = df['Справка_нед.Вес'].sum()
            weighted_g = (df['growth'] * df['Справка_нед.Вес']).sum() / 100
            resid = model_trend * (max(0, 100 - total_w) / 100)
            final_nowcast = weighted_g + resid

        # Format contributions
        if contribs:
            contribs_df = pd.DataFrame(contribs).groupby('Товар', as_index=False).agg({
                'Рост': 'mean', 'Вес': 'sum', 'Вклад (п.п.)': 'sum'
            }).sort_values('Вклад (п.п.)', ascending=False)
        else:
            contribs_df = None
            
        return final_nowcast, contribs_df

    def train_model(self):
        """Train Sirena-KBR v2.4 logic on monthly data."""
        print("\n🧠 Обучение модели (Ridge)...")
        df = self.monthly.copy()
        
        # Prepare Features
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['target'] = df['Все товары и услуги']
        
        df['y_lag1'] = df['target'].shift(1)
        df['y_lag2'] = df['target'].shift(2)
        df['y_lag12'] = df['target'].shift(12)
        df['y_ma3'] = df['target'].rolling(3).mean().shift(1)
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        df['food_lag1'] = df['Продовольственные товары'].shift(1)
        df['nonfood_lag1'] = df['Непродовольственные товары'].shift(1)
        df['services_lag1'] = df['Услуги'].shift(1)
        
        # Seasonal
        outliers = self.config['model_params']['outlier_years']
        clean = df[~df['year'].isin(outliers)]
        self.seasonal = clean.groupby('month')['target'].mean()
        
        df['seasonal_norm'] = df['month'].map(self.seasonal)
        df['deviation_lag1'] = df['y_lag1'] - df['month'].shift(1).map(self.seasonal)
        
        # Train
        train = df.dropna(subset=self.features + ['target'])
        train = train[~train['year'].isin(outliers)]
        
        X = train[self.features].values
        y = train['target'].values
        
        self.scaler = RobustScaler()
        X_sc = self.scaler.fit_transform(X)
        
        self.model = Ridge(alpha=self.config['model_params']['ridge_alpha'])
        self.model.fit(X_sc, y)
        
        print("   Модель обучена.")
        self.prepared_df = df # Store for forecasting loop

    def forecast_exogenous(self, horizon=12, fresh_usd=None, fresh_ruonia=None):
        """Forecast exogenous variables (USD, RUONIA)."""
        from statsmodels.tsa.arima.model import ARIMA
        
        # USD
        if fresh_usd is not None:
            # Scenario: Start from fresh_usd, then drift +0.1% per month
            usd_forecast = [fresh_usd]
            for i in range(1, horizon):
                usd_forecast.append(usd_forecast[-1] * 1.001)
        else:
            try:
                model_usd = ARIMA(self.monthly['usd_nom_i'].dropna(), order=(1,1,0))
                res_usd = model_usd.fit()
                usd_forecast = res_usd.forecast(steps=horizon).tolist()
            except:
                last_val = self.monthly['usd_nom_i'].iloc[-1]
                usd_forecast = [last_val] * horizon

        # RUONIA
        if fresh_ruonia is not None:
            # Scenario: Start from fresh_ruonia, then stable
            ruonia_forecast = [fresh_ruonia] * horizon
        else:
            try:
                model_r = ARIMA(self.monthly['Ruonia'].dropna(), order=(1,0,0))
                res_r = model_r.fit()
                ruonia_forecast = res_r.forecast(steps=horizon).tolist()
            except:
                last_val = self.monthly['Ruonia'].iloc[-1]
                ruonia_forecast = [last_val] * horizon
                
        return usd_forecast, ruonia_forecast

    def forecast(self, nowcast_val, target_date):
        """Rolling forecast starting from Nowcast."""
        print(f"\n🔮 Прогноз (Горизонт {self.config['model_params']['horizon_months']} мес)...")
        
        horizon = self.config['model_params']['horizon_months']
        results = []
        
        # Initialize history with actuals + nowcast
        history = list(self.prepared_df['target'].values)
        # Replace or append nowcast?
        # If target_date exists in history, replace. Else append.
        
        # But wait, 'target' is Index (100.5). Nowcast is MoM% (0.2).
        # Convert Nowcast to Index: 100 + 0.2 = 100.2
        nowcast_idx = 100.0 + nowcast_val
        
        # We treat Nowcast as the "Fact" for the first month
        history.append(nowcast_idx) 
        
        # Components naive extension
        last_comps = {
            'food': self.prepared_df['Продовольственные товары'].iloc[-1],
            'nonfood': self.prepared_df['Непродовольственные товары'].iloc[-1],
            'services': self.prepared_df['Услуги'].iloc[-1]
        }
        
        current_date = target_date # Nov 2025
        
        # Store Nowcast result first
        results.append({
            'Date': current_date,
            'MoM': nowcast_val,
            'Type': 'NOWCAST (Weekly)'
        })
        
        # Forecast subsequent months
        for i in range(1, horizon):
            next_date = current_date + pd.DateOffset(months=i)
            t_m = next_date.month
            
            # Features
            y_lag1 = history[-1] # This uses the Nowcast/Previous Pred
            y_lag2 = history[-2]
            # Lag12 approx
            y_lag12 = 100.5 # Simplify for distant future or fetch if available
            y_ma3 = np.mean(history[-3:])
            
            cur_sea = self.seasonal.get(t_m, 100.5)
            prev_sea = self.seasonal.get((next_date - pd.DateOffset(months=1)).month, 100.5)
            dev = y_lag1 - prev_sea
            
            feats = [
                y_lag1, y_lag2, y_lag12, y_ma3,
                np.sin(2*np.pi*t_m/12), np.cos(2*np.pi*t_m/12),
                last_comps['food'], last_comps['nonfood'], last_comps['services'],
                cur_sea, dev
            ]
            
            X = self.scaler.transform([feats])
            pred_ridge = self.model.predict(X)[0]
            
            # ETS Ensemble
            w_ets = self.config['ets_weights'].get(str(t_m), 0.3)
            pred = (1-w_ets)*pred_ridge + w_ets*cur_sea
            
            results.append({
                'Date': next_date,
                'MoM': pred - 100.0,
                'Type': 'MODEL'
            })
            history.append(pred)
            
        return pd.DataFrame(results)

    def run(self):
        if not self.load_data(): return
        
        # Determine target month for Nowcast (Next after last monthly data)
        last_monthly = self.monthly.index.max()
        target_date = last_monthly + pd.DateOffset(months=1)
        
        # 1. Train Model
        self.train_model()
        
        # 2. Get Model Trend (Naive prediction for target month)
        # We need to manually calculate the model prediction for target_date
        # Re-using logic from forecast loop (i=1 logic but for i=0 time)
        # Features from last_monthly
        history = self.prepared_df['target']
        food_last = self.prepared_df['Продовольственные товары'].iloc[-1]
        nonfood_last = self.prepared_df['Непродовольственные товары'].iloc[-1]
        services_last = self.prepared_df['Услуги'].iloc[-1]
        
        m = target_date.month
        y_lag1 = history.iloc[-1]
        y_lag2 = history.iloc[-2]
        y_lag12 = self.prepared_df['target'].iloc[-12] if len(history) > 11 else 100.5
        y_ma3 = history.iloc[-3:].mean()
        
        usd_fc, ruonia_fc = self.forecast_exogenous(1, self.exog['USD_RUB_CURRENT'], self.exog['RUONIA_CURRENT'])
        
        cur_sea = self.seasonal.get(m, 0)
        prev_sea = self.seasonal.get((target_date - pd.DateOffset(months=1)).month, 0)
        
        features = [y_lag1, y_lag2, y_lag12, y_ma3,
                   np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12),
                   food_last, nonfood_last, services_last,
                   cur_sea, y_lag1 - prev_sea]
        
        pred_ridge = self.model.predict(self.scaler.transform([features]))[0]
        ets_w = self.config['ets_weights'].get(str(m), 0.3)
        model_trend_index = (1 - ets_w) * pred_ridge + ets_w * cur_sea
        model_trend_mom = model_trend_index - 100.0
        
        print(f"\n🤖 Тренд модели на {target_date.strftime('%B')}: {model_trend_mom:+.2f}%")

        # 3. Calculate Nowcast (using model_trend for missing items)
        nc_val, contribs = self.calculate_nowcast(target_date.year, target_date.month, model_trend=model_trend_mom)
        
        if nc_val is None:
            print("⚠️ Не удалось рассчитать Nowcast (нет данных). Использую тренд модели.")
            nc_val = model_trend_mom
        
        print(f"\n📊 Результат Nowcast ({target_date.strftime('%B %Y')}): {nc_val:+.2f}%")
        if self.exog['INSIGHT_CPI_MOM']:
            print(f"ℹ️ Инсайд: {self.exog['INSIGHT_CPI_MOM']:+.2f}% (Δ = {nc_val - self.exog['INSIGHT_CPI_MOM']:+.3f})")
            
        print("\n🔝 Топ драйверов роста (из недельных):")
        if contribs is not None:
            print(contribs.head(5).to_string(index=False))
            
        # 4. Full Forecast
        forecast = self.forecast(
            nowcast_val=self.exog['INSIGHT_CPI_MOM'] if self.exog['INSIGHT_CPI_MOM'] else nc_val, 
            target_date=target_date
        )
        
        print("\n📅 Итоговый прогноз:")
        print(forecast.to_string(index=False))
        
        # Save
        forecast.to_csv(self.config['paths']['output_forecast'], index=False)
        if contribs is not None:
            contribs.to_csv(self.config['paths']['output_contributions'], index=False)

if __name__ == "__main__":
    from datetime import datetime
    pipeline = SirenaPipeline()
    pipeline.run()
