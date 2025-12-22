import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler, StandardScaler
from sirena.legacy_usd import SirenaUSD
import warnings
import os

from exog_functions import load_last_exog_values, generate_auto_exog_forecast, load_manual_exog, save_manual_exog

# --- SETUP ---
st.set_page_config(
    page_title="СИРЕНА-КБР v4.2: Прогноз Инфляции",
    layout="wide",
    initial_sidebar_state="expanded"
)
warnings.filterwarnings('ignore')

# --- MODEL CLASS (FINAL v2.4) ---
class SirenaKBR_v24:
    """
    СИРЕНА-КБР v2.4: Финальная оптимизированная модель.
    MAE: 0.3317 (-12.4% vs v2.0)
    """
    def __init__(self):
        # v2.4 Parameters
        self.OUTLIER_YEARS = [2010, 2022]
        self.ETS_WEIGHTS = {
            1: 0.9, 2: 0.0, 3: 0.5, 4: 0.3, 5: 0.9, 6: 0.5,
            7: 0.0, 8: 0.5, 9: 0.9, 10: 0.9, 11: 0.0, 12: 0.0
        }
        self.ALPHA = 0.3
        self.scaler = RobustScaler()
        self.ridge = None
        self.seasonal_norm = None
        self.weekly_prices = None
        
        # Full Feature Set v2.4
        self.feature_cols = [
            'y_lag1', 'y_lag2', 'y_lag12', 'y_ma3',
            'month_sin', 'month_cos',
            'food_lag1', 'nonfood_lag1', 'services_lag1',
            'seasonal_norm', 'deviation_lag1'
        ]
    
    def set_weekly_data(self, df_weekly):
        self.weekly_prices = df_weekly
        
    def get_fv_yoy(self, year, month):
        """Nowcasting signal from weekly data (Fruit/Veg deviation)."""
        if self.weekly_prices is None: return None
        key_products = ['Картофель', 'Капуста', 'Лук', 'Морковь', 'Огурцы', 'Помидоры', 'Яблоки', 'Бананы']
        yoys = []
        wp = self.weekly_prices
        try:
            current = wp[(wp['year'] == year) & (wp['month'] == month)]
            prev = wp[(wp['year'] == year - 1) & (wp['month'] == month)]
            if current.empty or prev.empty: return None
            for k in key_products:
                c = current[current['Товары'].str.contains(k, case=False, na=False)]
                p = prev[prev['Товары'].str.contains(k, case=False, na=False)]
                if not c.empty and not p.empty:
                    p_cur, p_prev_val = c['Значение'].mean(), p['Значение'].mean()
                    if p_prev_val > 0: yoys.append((p_cur / p_prev_val - 1) * 100)
        except: return None
        return np.mean(yoys) if yoys else None

    def get_weekly_nowcast(self, year, month, df_monthly):
        """
        Calculate MoM estimate using MIDAS (Bridge Model).
        """
        if self.weekly_prices is None: return None
        
        try:
            from sirena.legacy.sirena_midas import SirenaMIDAS
            midas = SirenaMIDAS()
            
            # Pivot weekly to wide for MIDAS
            w_wide = self.weekly_prices.pivot_table(
                index=['Товары', 'Rostat_code'], 
                columns='Сведено', 
                values='Значение', 
                aggfunc='mean'
            ).reset_index()
            
            # Prepare Training Data
            monthly_target = df_monthly[['Все товары и услуги']].copy()
            midas_df = midas.prepare_data(w_wide, monthly_target)
            midas.fit(midas_df)
            
            # Calculate Proxy for Target Month
            # Identify weeks
            week_cols = [c for c in w_wide.columns if '_' in str(c) and str(c)[0].isdigit()]
            curr_weeks = []
            prev_weeks = []
            
            # Need previous month
            if month == 1: p_m, p_y = 12, year - 1
            else: p_m, p_y = month - 1, year
            
            for col in week_cols:
                try:
                    y_str, w_str = str(col).split('_')
                    d = datetime.fromisocalendar(int(y_str), int(w_str), 1)
                    if d.year == year and d.month == month:
                        curr_weeks.append(col)
                    elif d.year == p_y and d.month == p_m:
                        prev_weeks.append(col)
                except: pass
            
            if not curr_weeks or not prev_weeks:
                return None
                
            # Calculate Proxy Growth (Robust to missing items)
            mean_curr = w_wide[curr_weeks].mean(axis=1)
            mean_prev = w_wide[prev_weeks].mean(axis=1)
            
            # Intersection of valid items
            common_items = mean_curr.notna() & mean_prev.notna()
            
            if not common_items.any(): return None
            
            curr_basket = mean_curr[common_items].sum()
            prev_basket = mean_prev[common_items].sum()
            
            if prev_basket == 0: return None
            
            proxy_mom = (curr_basket / prev_basket - 1) * 100
            last_target = df_monthly['Все товары и услуги'].iloc[-1] - 100
            
            pred = midas.predict(proxy_mom, last_target)
            return pred
            
        except Exception as e:
            # print(f"MIDAS error: {e}")
            return None
    
    def prepare_features(self, df):
        df = df.copy()
        df['month'] = df.index.month
        df['year'] = df.index.year
        
        # Lags
        df['y_lag1'] = df['Все товары и услуги'].shift(1)
        df['y_lag2'] = df['Все товары и услуги'].shift(2)
        df['y_lag12'] = df['Все товары и услуги'].shift(12)
        df['y_ma3'] = df['Все товары и услуги'].rolling(3).mean().shift(1)
        
        # Trig seasonality
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Components
        df['food_lag1'] = df['Продовольственные товары'].shift(1)
        df['nonfood_lag1'] = df['Непродовольственные товары'].shift(1)
        df['services_lag1'] = df['Услуги'].shift(1)
        
        # Seasonal Norm
        clean = df[~df['year'].isin(self.OUTLIER_YEARS)]
        if len(clean) < 12: clean = df
        self.seasonal_norm = clean.groupby('month')['Все товары и услуги'].mean()
        
        # Fill missing months if any
        for m in range(1, 13):
            if m not in self.seasonal_norm: self.seasonal_norm[m] = 100.5
            
        df['seasonal_norm'] = df['month'].map(self.seasonal_norm)
        df['deviation_lag1'] = df['y_lag1'] - df['month'].shift(1).map(self.seasonal_norm)
        return df
    
    def fit(self, df):
        df = self.prepare_features(df)
        train_clean = df.dropna(subset=self.feature_cols + ['Все товары и услуги'])
        train_clean = train_clean[~train_clean['year'].isin(self.OUTLIER_YEARS)]
        
        if len(train_clean) > 0:
            X = train_clean[self.feature_cols].values
            y = train_clean['Все товары и услуги'].values
            X_scaled = self.scaler.fit_transform(X)
            
            self.ridge = Ridge(alpha=self.ALPHA)
            self.ridge.fit(X_scaled, y)
        return self

    def predict(self, df: pd.DataFrame, target_date: pd.Timestamp):
        """Прогноз на указанную дату."""
        if self.ridge is None:
             raise ValueError("Модель не обучена")
        
        # Подготовка признаков
        df = self.prepare_features(df)
        
        test_row = df.loc[[target_date]]
        
        # Прогноз Ridge
        X_test = self.scaler.transform(test_row[self.feature_cols].values)
        pred_ridge = self.ridge.predict(X_test)[0]
        
        # Прогноз ETS (сезонная норма)
        target_month = target_date.month
        pred_ets = self.seasonal_norm.get(target_month, 100.5)
        
        # Адаптивная комбинация
        ets_weight = self.ETS_WEIGHTS.get(target_month, 0.3)
        ridge_weight = 1 - ets_weight
        
        pred_combined = ridge_weight * pred_ridge + ets_weight * pred_ets
        
        return {
            'date': target_date,
            'prediction': pred_combined,
            'pred_ridge': pred_ridge,
            'pred_ets': pred_ets,
            'ets_weight': ets_weight,
            'month': target_month,
        }
    
    def backtest(self, df: pd.DataFrame, start_date: str = '2019-01-01', 
                 end_date: str = None):
        """Бэктестинг модели."""
        # Valid data range
        valid_df = df.dropna(subset=['Все товары и услуги'])
        last_fact = valid_df.index.max()
        
        if end_date is None:
            end_date = last_fact
        else:
            end_date = min(pd.Timestamp(end_date), last_fact)
            
        test_dates = pd.date_range(start_date, end_date, freq='MS')
        results = []
        
        for target_date in test_dates:
            if target_date not in df.index:
                continue
                
            # Check if actual exists
            actual = df.loc[target_date, 'Все товары и услуги']
            if pd.isna(actual):
                continue

            cutoff = target_date - pd.DateOffset(months=1)
            train_df = df[df.index <= cutoff].copy()
            train_df = train_df.dropna(subset=['Все товары и услуги']) # Train only on valid data
            
            if len(train_df) < 36:
                continue
            
            try:
                self.fit(train_df)
                test_df = df[df.index <= target_date].copy()
                pred_result = self.predict(test_df, target_date)
                
                results.append({
                    'date': target_date,
                    'actual': actual,
                    'prediction': pred_result['prediction'],
                    'error': actual - pred_result['prediction'],
                })
            except Exception as e:
                continue
        
        return pd.DataFrame(results)
    
    def get_metrics(self, results: pd.DataFrame):
        """Расчёт метрик качества."""
        mae = results['error'].abs().mean()
        rmse = np.sqrt((results['error'] ** 2).mean())
        kpi_count = (results['error'].abs() <= 0.5).sum()
        kpi_pct = kpi_count / len(results) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'KPI': f"{kpi_count}/{len(results)} ({kpi_pct:.1f}%)",
        }

    def predict_horizon(self, df, start_date, horizon=12, shocks=None, 
                       food_trend_adj=0.0, fx_shock_pct=0.0, seasonality_strength=1.0):
        if shocks is None: shocks = {}
        results = []
        df_work = df.copy()
        history = list(df_work['Все товары и услуги'].values)
        
        # Naive extension for components
        last_vals = {
            'food': df_work['Продовольственные товары'].iloc[-1] + food_trend_adj,
            'nonfood': df_work['Непродовольственные товары'].iloc[-1],
            'services': df_work['Услуги'].iloc[-1]
        }

        for i in range(horizon):
            t_date = start_date + pd.DateOffset(months=i)
            t_m, t_y = t_date.month, t_date.year
            
            # Feature prep for one step
            cur_sea_raw = self.seasonal_norm.get(t_m, 100.5)
            
            # Apply Seasonality Strength
            seasonal_dev = cur_sea_raw - 100.5
            cur_sea = 100.5 + (seasonal_dev * seasonality_strength)
            
            prev_sea = self.seasonal_norm.get((t_date - pd.DateOffset(months=1)).month, 100.5)
            cur_dev = history[-1] - prev_sea
            
            # Lags approximation
            y_lag1 = history[-1]
            y_lag2 = history[-2] if len(history) > 1 else history[-1]
            y_lag12 = history[-12] if len(history) > 11 else 100.5
            y_ma3 = np.mean(history[-3:]) if len(history) > 2 else history[-1]
            
            X_feat = [
                y_lag1, y_lag2, y_lag12, y_ma3,
                np.sin(2 * np.pi * t_m / 12),
                np.cos(2 * np.pi * t_m / 12),
                last_vals['food'], last_vals['nonfood'], last_vals['services'],
                cur_sea,
                cur_dev
            ]
            
            X = np.array([X_feat])
            X_sc = self.scaler.transform(X)
            
            # 1. Ridge Prediction
            pred_ridge = self.ridge.predict(X_sc)[0]
            
            # 2. ETS Prediction
            pred_ets = cur_sea
            
            # 3. Weighted Combination
            w_ets = self.ETS_WEIGHTS.get(t_m, 0.3)
            pred_combined = (1 - w_ets) * pred_ridge + w_ets * pred_ets
            
            # 4. Nowcasting (Detailed Weekly)
            nowcast_val = 0
            used_nowcast = False
            if i == 0: # Only for the first month (immediate future)
                weekly_mom = self.get_weekly_nowcast(t_y, t_m, df)
                if weekly_mom is not None:
                    # Blend Model (50%) and Real Weekly Data (50%)
                    # This is a strong correction based on actuals
                    pred_combined = (pred_combined * 0.5) + (weekly_mom * 0.5)
                    nowcast_val = weekly_mom - pred_combined # Diff for display
                    used_nowcast = True
            
            # 4b. Fallback Nowcasting (Fruit/Veg specific)
            fv_adj = 0
            if not used_nowcast and i < 2:
                fv_yoy = self.get_fv_yoy(t_y, t_m)
                if fv_yoy is not None:
                    fv_adj = 0.06 * ((fv_yoy - 8.0) / 12) * 0.5
                    if abs(fv_adj) > 0.01: pred_combined += fv_adj
            
            # 5. Shocks
            shock_val = shocks.get(t_date, 0)
            pred_combined += shock_val
            
            # 6. FX Pass-through
            fx_effect = 0
            if fx_shock_pct != 0 and i < 6:
                decay_weights = [0.3, 0.25, 0.2, 0.15, 0.05, 0.05]
                fx_effect = fx_shock_pct * 0.1 * decay_weights[i]
                pred_combined += fx_effect

            results.append({
                'Date': t_date,
                'Month': t_date.strftime('%b %Y'),
                'MoM': pred_combined - 100.0, 
                'MoM_Index': pred_combined,
                'Ridge': pred_ridge,
                'ETS': pred_ets,
                'Weight_ETS': w_ets,
                'Shock': shock_val,
                'Nowcast': nowcast_val + fv_adj, # Combine both types for display
                'FX_Effect': fx_effect
            })
            history.append(pred_combined)
            
        return pd.DataFrame(results)

# --- BVAR INTEGRATION ---
@st.cache_resource
def run_bvar_forecast(horizon=12, cutoff_date=None):
    try:
        from sirena.models.bvar import BayesianVAR
        
        # Load Data
        df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
        # Fix loading if needed
        cols_to_fix = ['mom', 'Prod', 'Nonprod', 'Serv', 'usd_nom_i', 'Ruonia']
        for col in cols_to_fix:
            if col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(',', '.')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y', errors='coerce')
        if df['Date'].isna().any(): df['Date'] = pd.to_datetime(df['Date'])
        
        # Normalize to start of month for alignment
        df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp()
        
        df = df.set_index('Date').sort_index()
        
        # Prepare
        data = pd.DataFrame()
        data['CPI'] = df['mom'] - 100
        data['Food'] = df['Prod'] - 100
        data['NonFood'] = df['Nonprod'] - 100
        data['Services'] = df['Serv'] - 100
        data['USD'] = df['usd_nom_i'] - 100
        data['RUONIA'] = df['Ruonia']
        data = data.dropna()
        
        # Align with Ridge cutoff
        if cutoff_date is not None:
            data = data[data.index <= cutoff_date]
        
        # Train (Analytical BVAR v2.0)
        # Using slightly looser priors to allow data to speak
        model = BayesianVAR(lags=4, lambda1=1.0, lambda2=0.5, lambda3=1.0, var_names=['CPI', 'Food', 'USD', 'RUONIA'])
        model.fit(data, target_col='CPI')
        
        # Forecast
        fc = model.forecast_full(horizon=horizon)
        median_path = fc['median'][:, 0] # CPI is index 0
        
        last_date = data.index.max()
        dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')
        
        return pd.DataFrame({'Date': dates, 'BVAR': median_path})
        
    except Exception as e:
        st.warning(f"BVAR недоступен: {e}")
        return None

# --- SARIMA INTEGRATION ---
@st.cache_resource
def run_sarima_forecast(horizon=12, cutoff_date=None):
    try:
        from sirena.legacy.sirena_arima import SirenaARIMA

        # Load Data (Aggregated)
        df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')

        # Fix numeric
        if 'MoM' in df_raw.columns:
             if df_raw['MoM'].dtype == object:
                 df_raw['MoM'] = df_raw['MoM'].astype(str).str.replace(',', '.')
             df_raw['MoM'] = pd.to_numeric(df_raw['MoM'], errors='coerce')

        if 'Day' in df_raw.columns:
             df_raw['Date'] = pd.to_datetime(df_raw['Day'], format='%d.%m.%Y', errors='coerce')
        elif 'Date' in df_raw.columns:
             df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')

        if 'Товар' in df_raw.columns and 'MoM' in df_raw.columns:
             df = df_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
        else:
             df = df_raw.set_index('Date')

        df = df.sort_index()

        # Align with Ridge cutoff
        if cutoff_date is not None:
            df = df[df.index <= cutoff_date]

        ts = df['Все товары и услуги'].dropna() - 100

        model = SirenaARIMA()
        model.fit_sarima(ts)
        fc = model.forecast(steps=horizon)

        last_date = ts.index.max()
        dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')

        return pd.DataFrame({'Date': dates, 'SARIMA': fc['mean'].values})

    except Exception as e:
        st.warning(f"SARIMA недоступна: {e}")
        return None


# --- LIGHTGBM INTEGRATION ---
@st.cache_resource
def run_lightgbm_forecast(horizon=12, cutoff_date=None):
    try:
        from sirena.legacy.sirena_lightgbm import SirenaLightGBM

        df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')

        if 'MoM' in df_raw.columns:
            if df_raw['MoM'].dtype == object:
                df_raw['MoM'] = df_raw['MoM'].astype(str).str.replace(',', '.')
            df_raw['MoM'] = pd.to_numeric(df_raw['MoM'], errors='coerce')

        if 'Day' in df_raw.columns:
            df_raw['Date'] = pd.to_datetime(df_raw['Day'], format='%d.%m.%Y', errors='coerce')
        elif 'Date' in df_raw.columns:
            df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')

        if 'Товар' in df_raw.columns and 'MoM' in df_raw.columns:
            df = df_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
        else:
            df = df_raw.set_index('Date')

        df = df.sort_index()

        if cutoff_date is not None:
            df = df[df.index <= cutoff_date]

        model = SirenaLightGBM()
        model.fit(df)

        last_date = df.index.max()
        fc = model.forecast(horizon=horizon, start_date=last_date + pd.DateOffset(months=1))

        dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')

        return pd.DataFrame({'Date': dates, 'LightGBM': fc})

    except Exception as e:
        st.warning(f"LightGBM недоступен: {e}")
        return None


# --- PROPHET INTEGRATION ---
@st.cache_resource
def run_prophet_forecast(horizon=12, cutoff_date=None):
    try:
        from sirena.legacy.sirena_prophet import SirenaProphet

        df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')

        if 'MoM' in df_raw.columns:
            if df_raw['MoM'].dtype == object:
                df_raw['MoM'] = df_raw['MoM'].astype(str).str.replace(',', '.')
            df_raw['MoM'] = pd.to_numeric(df_raw['MoM'], errors='coerce')

        if 'Day' in df_raw.columns:
            df_raw['Date'] = pd.to_datetime(df_raw['Day'], format='%d.%m.%Y', errors='coerce')
        elif 'Date' in df_raw.columns:
            df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')

        if 'Товар' in df_raw.columns and 'MoM' in df_raw.columns:
            df = df_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
        else:
            df = df_raw.set_index('Date')

        df = df.sort_index()

        if cutoff_date is not None:
            df = df[df.index <= cutoff_date]

        model = SirenaProphet()
        model.fit(df, 'Все товары и услуги')
        fc = model.forecast(horizon=horizon)

        return pd.DataFrame({'Date': fc['dates'], 'Prophet': fc['mean']})

    except Exception as e:
        st.warning(f"Prophet недоступен: {e}")
        return None


# --- ETS INTEGRATION ---
@st.cache_resource
def run_ets_forecast(horizon=12, cutoff_date=None):
    try:
        from sirena.legacy.sirena_ets import SirenaETS

        df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')

        if 'MoM' in df_raw.columns:
            if df_raw['MoM'].dtype == object:
                df_raw['MoM'] = df_raw['MoM'].astype(str).str.replace(',', '.')
            df_raw['MoM'] = pd.to_numeric(df_raw['MoM'], errors='coerce')

        if 'Day' in df_raw.columns:
            df_raw['Date'] = pd.to_datetime(df_raw['Day'], format='%d.%m.%Y', errors='coerce')
        elif 'Date' in df_raw.columns:
            df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')

        if 'Товар' in df_raw.columns and 'MoM' in df_raw.columns:
            df = df_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
        else:
            df = df_raw.set_index('Date')

        df = df.sort_index()

        if cutoff_date is not None:
            df = df[df.index <= cutoff_date]

        model = SirenaETS()
        model.fit(df, 'Все товары и услуги')
        fc = model.forecast(horizon=horizon)

        last_date = df.index.max()
        dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')

        return pd.DataFrame({'Date': dates, 'ETS': fc['mean']})

    except Exception as e:
        st.warning(f"ETS недоступен: {e}")
        return None


# --- EBM INTEGRATION (v4.0.2) ---
@st.cache_resource
def run_ebm_forecast(horizon=12, cutoff_date=None):
    """
    EBM (Explainable Boosting Machine) прогноз.
    Заменяет LSTM в ансамбле — лучше метрики и интерпретируемость.
    """
    try:
        from sirena.models.ebm import EBMForecaster

        df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')

        if 'MoM' in df_raw.columns:
            if df_raw['MoM'].dtype == object:
                df_raw['MoM'] = df_raw['MoM'].astype(str).str.replace(',', '.')
            df_raw['MoM'] = pd.to_numeric(df_raw['MoM'], errors='coerce')

        if 'Day' in df_raw.columns:
            df_raw['Date'] = pd.to_datetime(df_raw['Day'], format='%d.%m.%Y', errors='coerce')
        elif 'Date' in df_raw.columns:
            df_raw['Date'] = pd.to_datetime(df_raw['Date'], errors='coerce')

        if 'Товар' in df_raw.columns and 'MoM' in df_raw.columns:
            df = df_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
        else:
            df = df_raw.set_index('Date')

        df = df.sort_index()

        if cutoff_date is not None:
            df = df[df.index <= cutoff_date]

        model = EBMForecaster()
        model.fit(df, 'Все товары и услуги')
        fc = model.forecast(horizon=horizon)

        # EBM returns index ~100, convert to MoM (deviation from 100)
        fc_mom = fc - 100.0

        last_date = df.index.max()
        dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')

        return pd.DataFrame({'Date': dates, 'EBM': fc_mom}), model

    except Exception as e:
        st.warning(f"EBM недоступен: {e}")
        return None, None


def get_ebm_feature_importance():
    """Получить важность признаков EBM для визуализации."""
    try:
        from sirena.models.ebm import EBMForecaster

        df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')
        if df_raw['MoM'].dtype == object:
            df_raw['MoM'] = df_raw['MoM'].astype(str).str.replace(',', '.')
        df_raw['MoM'] = pd.to_numeric(df_raw['MoM'], errors='coerce')
        df_raw['Date'] = pd.to_datetime(df_raw['Day'], format='%d.%m.%Y', errors='coerce')
        df = df_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first').sort_index()

        model = EBMForecaster()
        model.fit(df, 'Все товары и услуги')
        return model.get_feature_importance()
    except:
        return None

# --- DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.DataFrame()
    df_weekly = None
    
    try:
        # Загружаем агрегированные данные (для истории общего ИПЦ)
        df_agg_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal='.')
        # В новом файле формат даты - YYYY-MM-DD, а не DD.MM.YYYY (как в старом)
        # Попробуем оба формата
        try:
            df_agg_raw['Date'] = pd.to_datetime(df_agg_raw['Day'], format='%d.%m.%Y')
        except:
            df_agg_raw['Date'] = pd.to_datetime(df_agg_raw['Day'], format='%Y-%m-%d', errors='coerce')
            if df_agg_raw['Date'].isna().all():
                # Fallback если вдруг формат другой
                df_agg_raw['Date'] = pd.to_datetime(df_agg_raw['Day'])
            
        if 'Товар' in df_agg_raw.columns and 'MoM' in df_agg_raw.columns:
             df = df_agg_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
        else:
             df = df_agg_raw.set_index('Date')
             
        df = df[['Все товары и услуги', 'Продовольственные товары', 'Непродовольственные товары', 'Услуги']].copy()
        df = df.sort_index()

        # Добавляем экзогенные переменные из inflation_data.csv (Ki_i, usd_nom_i)
        # Ki_i_lag1 — лучший экзогенный признак (MAE -9% vs baseline)
        try:
            exog_df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
            exog_df['Date'] = pd.to_datetime(exog_df['Date'], format='%d.%m.%Y', errors='coerce')
            if exog_df['Date'].isna().any():
                exog_df['Date'] = pd.to_datetime(exog_df['Date'])
            exog_df['Date'] = exog_df['Date'].dt.to_period('M').dt.to_timestamp()
            exog_df = exog_df.set_index('Date')

            # Добавляем Ki_i (ключевая ставка ЦБ)
            if 'Ki_i' in exog_df.columns:
                ki_col = exog_df['Ki_i']
                if ki_col.dtype == object:
                    ki_col = ki_col.astype(str).str.replace(',', '.')
                df['Ki_i'] = pd.to_numeric(ki_col, errors='coerce')

            # Добавляем usd_nom_i (курс доллара)
            if 'usd_nom_i' in exog_df.columns:
                usd_col = exog_df['usd_nom_i']
                if usd_col.dtype == object:
                    usd_col = usd_col.astype(str).str.replace(',', '.')
                df['usd_nom_i'] = pd.to_numeric(usd_col, errors='coerce')
        except Exception as e:
            pass  # Экзогенные переменные опциональны

    except Exception as e:
        st.error(f"Ошибка загрузки infl_kbr.csv: {e}")
        return None, None

    if os.path.exists('data/weekly_prices.csv'):
        try:
            w = pd.read_csv('data/weekly_prices.csv', sep=';', decimal=',')
            if 'Товары' not in w.columns:
                 w = pd.read_csv('data/weekly_prices.csv', sep=';', decimal='.')
            if 'Сведено' in w.columns:
                w[['year', 'week']] = w['Сведено'].str.split('_', expand=True).astype(int)
                w['month'] = pd.to_datetime(w['year'].astype(str) + w['week'].astype(str) + '1', format='%Y%W%w').dt.month
                df_weekly = w
        except: pass
        
    return df.sort_index(), df_weekly

# --- MAIN APP ---

df, df_weekly = load_data()

if df is not None:
    model = SirenaKBR_v24()
    if df_weekly is not None:
        model.set_weekly_data(df_weekly)
    model.fit(df)
    
    # --- SIDEBAR ---
    st.sidebar.title("⚙️ Параметры")
    horizon = st.sidebar.slider("Горизонт прогноза (мес)", 6, 24, 12)
    font_size = st.sidebar.slider("Размер шрифта (px)", 12, 24, 16)
    
    st.markdown(f"""
    <style>
    html, body, [class*="css"] {{
        font_size: {font_size}px;
    }}
    </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.subheader("Экзогенные шоки")
    jkh_val = st.sidebar.number_input("Рост ЖКХ (%)", 0.0, 20.0, 0.0, step=0.1)
    jkh_date = st.sidebar.date_input("Дата ЖКХ", datetime(2026, 7, 1))
    
    with st.sidebar.expander("🎛 Сценарный анализ"):
        st.caption("Моделирование 'Что-если'")

        # Загрузка актуальных значений из настроек
        actual_exog = load_last_exog_values()

        st.caption("Рыночные параметры:")
        current_usd = st.number_input("Текущий курс USD", 50.0, 150.0, actual_exog['usd'], 0.5)
        current_key_rate = st.number_input("Ключевая ставка (%)", 5.0, 30.0, actual_exog['key_rate'], 0.5)
        
        st.caption("Шоки и тренды:")
        food_trend_adj = st.slider("Тренд Продовольствия (п.п.)", -2.0, 2.0, 0.0, 0.1)
        fx_shock_pct = st.number_input("Валютный шок (девальвация %)", 0.0, 50.0, 0.0, 1.0)
        seasonality_strength = st.slider("Сила сезонности", 0.5, 1.5, 1.0, 0.1)
    
    shocks = {}
    if jkh_val > 0:
        effect = (jkh_val * 0.0909) 
        d = pd.Timestamp(jkh_date).replace(day=1)
        shocks[d] = effect
    
    last_date = df.index.max()
    start_date = last_date + pd.DateOffset(months=1)
    
    forecast = model.predict_horizon(df, start_date, horizon, shocks, 
                                   food_trend_adj=food_trend_adj, 
                                   fx_shock_pct=fx_shock_pct,
                                   seasonality_strength=seasonality_strength)
    
    # --- EXOGENOUS PROJECTIONS ---
    exog_dates = forecast['Date']
    
    # Check manual vs auto
    manual_exog = load_manual_exog()
    if manual_exog and manual_exog.get('source') == 'manual':
        # Ensure length matches horizon
        usd_proj = manual_exog['USD'][:horizon]
        key_rate_proj = manual_exog['KeyRate'][:horizon]
        # Fill if short?
        if len(usd_proj) < horizon:
            usd_proj += [usd_proj[-1]] * (horizon - len(usd_proj))
            key_rate_proj += [key_rate_proj[-1]] * (horizon - len(key_rate_proj))
    else:
        auto_exog = generate_auto_exog_forecast(last_date, horizon, current_usd, current_key_rate)
        usd_proj = auto_exog['USD']
        key_rate_proj = auto_exog['KeyRate']
        
    # Apply FX shock slider on top
    if fx_shock_pct != 0:
        # Apply gradually over 3 months
        usd_proj = [u * (1.0 + (fx_shock_pct / 100.0) * (min(i+1, 3)/3.0)) for i, u in enumerate(usd_proj)]
    
    # --- DASHBOARD LAYOUT ---
    
    st.title("📊 СИРЕНА-КБР v4.0")
    st.markdown("""
    **Ансамблевая модель** из 7 компонентов: Ridge, BVAR, SARIMA, LightGBM, Prophet, ETS, EBM.
    **Веса:** Ridge 40% | BVAR 20% | LightGBM 15% | Prophet 10% | SARIMA/ETS/EBM 5%
    """)
    
    # Metrics
    cum_infl = (np.prod(forecast['MoM_Index']/100) - 1) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Накопленная инфляция", f"+{cum_infl:.2f}%", help=f"За {horizon} месяцев")
    col2.metric("Средняя мес. инфляция", f"+{forecast['MoM'].mean():.2f}%")
    col3.metric("Пиковый месяц", f"{forecast.loc[forecast['MoM'].idxmax(), 'Month']}", f"+{forecast['MoM'].max():.2f}%")
    
    nowcast_val = forecast['Nowcast'].iloc[0] if len(forecast) > 0 else 0
    col4.metric("Nowcast сигнал", f"{nowcast_val:+.2f} п.п.")

    # --- TABS ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["📈 Прогноз", "🔧 Экзогенные", "✅ Бэктест", "🛠 Методология", "🧠 История (Opus)", "🗺 Регионы", "🔒 Инсайдер", "🔍 EBM"])
    
    with tab1:
        # --- 1. FORECAST CHART ---
        st.subheader("🔮 Прогноз на 12 месяцев")
        
        # Calculate Historical Medians (Seasonality Baseline)
        # Exclude outliers for robust median
        clean_hist = df[~df.index.year.isin([2010, 2022])]
        monthly_medians = clean_hist.groupby(clean_hist.index.month)['Все товары и услуги'].median()
        
        # Map medians to forecast dates
        forecast_medians = forecast['Date'].dt.month.map(monthly_medians) - 100.0
        
        fig_fc = go.Figure()
        
        # Forecast
        fig_fc.add_trace(go.Scatter(
            x=forecast['Date'], y=forecast['MoM'], 
            name='Прогноз (Ridge)', 
            line=dict(color='#2563eb', width=4),
            mode='lines+markers'
        ))
        
        # Seasonal Norm (Median)
        fig_fc.add_trace(go.Scatter(
            x=forecast['Date'], y=forecast_medians,
            name='Норма (Медиана)',
            line=dict(color='gray', width=1, dash='dot'),
            mode='lines'
        ))
        
        # Calculate Alternative Models (v4.4 ensemble)
        with st.spinner("Расчет ансамбля моделей (v4.6 + Shock Dummies)..."):
            bvar_df = run_bvar_forecast(horizon, cutoff_date=last_date)
            sarima_df = run_sarima_forecast(horizon, cutoff_date=last_date)
            lightgbm_df = run_lightgbm_forecast(horizon, cutoff_date=last_date)
            prophet_df = run_prophet_forecast(horizon, cutoff_date=last_date)
            ets_df = run_ets_forecast(horizon, cutoff_date=last_date)
            ebm_result = run_ebm_forecast(horizon, cutoff_date=last_date)
            ebm_df = ebm_result[0] if ebm_result else None

            # v4.2: Ridge Extended and Bayesian Ridge forecasts
            ridge_ext_df = None
            bayes_ridge_df = None
            try:
                from sirena.models.ridge_extended import RidgeExtendedForecaster
                from sirena.models.bayesian_ridge import BayesianRidgeForecaster

                # Ridge Extended
                rex = RidgeExtendedForecaster()
                rex.fit(df)
                rex_vals = []
                for h in range(horizon):
                    target_date = last_date + pd.DateOffset(months=h+1)
                    df_ext = df.copy()
                    df_ext.loc[target_date] = np.nan
                    try:
                        pred = rex.predict(df_ext, target_date)['prediction'] - 100
                        rex_vals.append(pred)
                    except:
                        rex_vals.append(np.nan)
                if rex_vals and not all(np.isnan(rex_vals)):
                    ridge_ext_df = pd.DataFrame({
                        'Date': pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS'),
                        'Ridge Ext': rex_vals
                    })

                # Bayesian Ridge
                br = BayesianRidgeForecaster()
                br.fit(df)
                br_vals = []
                br_ci_lower = []
                br_ci_upper = []
                for h in range(horizon):
                    target_date = last_date + pd.DateOffset(months=h+1)
                    df_ext = df.copy()
                    df_ext.loc[target_date] = np.nan
                    try:
                        pred = br.predict_with_ci(df_ext, target_date)
                        br_vals.append(pred['prediction'] - 100)
                        br_ci_lower.append(pred['ci_lower'] - 100)
                        br_ci_upper.append(pred['ci_upper'] - 100)
                    except:
                        br_vals.append(np.nan)
                        br_ci_lower.append(np.nan)
                        br_ci_upper.append(np.nan)
                if br_vals and not all(np.isnan(br_vals)):
                    bayes_ridge_df = pd.DataFrame({
                        'Date': pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS'),
                        'Bayes Ridge': br_vals,
                        'CI Lower': br_ci_lower,
                        'CI Upper': br_ci_upper
                    })
            except Exception as e:
                pass  # v4.2 models not available

            # v4.4: NGBoost (лучшая модель!)
            ngboost_df = None
            try:
                from sirena.models.ngboost_model import NGBoostForecaster, NGBOOST_AVAILABLE
                if NGBOOST_AVAILABLE:
                    ngb = NGBoostForecaster()
                    ngb.fit(df)
                    ngb_vals = []
                    ngb_ci_lower = []
                    ngb_ci_upper = []
                    for h in range(horizon):
                        target_date = last_date + pd.DateOffset(months=h+1)
                        df_ext = df.copy()
                        df_ext.loc[target_date] = np.nan
                        try:
                            pred = ngb.predict(df_ext, target_date)
                            ngb_vals.append(pred['prediction'] - 100)
                            ngb_ci_lower.append(pred['ci_lower'] - 100)
                            ngb_ci_upper.append(pred['ci_upper'] - 100)
                        except:
                            ngb_vals.append(np.nan)
                            ngb_ci_lower.append(np.nan)
                            ngb_ci_upper.append(np.nan)
                    if ngb_vals and not all(np.isnan(ngb_vals)):
                        ngboost_df = pd.DataFrame({
                            'Date': pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS'),
                            'NGBoost': ngb_vals,
                            'CI Lower': ngb_ci_lower,
                            'CI Upper': ngb_ci_upper
                        })
            except Exception as e:
                pass  # NGBoost not available

            # v4.6: Ridge Shock Dummies
            shock_dummies_df = None
            try:
                from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster

                shock_model = RidgeShockDummiesForecaster(use_2022_dummy=False)
                shock_model.fit(df)
                shock_vals = []
                for h in range(horizon):
                    target_date = last_date + pd.DateOffset(months=h+1)
                    df_ext = df.copy()
                    df_ext.loc[target_date] = np.nan
                    try:
                        pred = shock_model.predict(df_ext, target_date)['prediction'] - 100
                        shock_vals.append(pred)
                    except:
                        shock_vals.append(np.nan)
                if shock_vals and not all(np.isnan(shock_vals)):
                    shock_dummies_df = pd.DataFrame({
                        'Date': pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS'),
                        'Shock Dummies': shock_vals
                    })
            except Exception as e:
                pass  # Ridge Shock Dummies not available

            # LMMR Claude Implementation (Linear Mixed Model Regression)
            lmmr_df = None
            try:
                from sirena.models.lmmr_claude import LMMRForecasterClaude

                lmmr_model = LMMRForecasterClaude()
                lmmr_model.fit(df)
                lmmr_vals = []
                for h in range(horizon):
                    target_date = last_date + pd.DateOffset(months=h+1)
                    df_ext = df.copy()
                    df_ext.loc[target_date] = np.nan
                    try:
                        pred = lmmr_model.predict(df_ext, target_date)['prediction'] - 100
                        lmmr_vals.append(pred)
                    except:
                        lmmr_vals.append(np.nan)
                if lmmr_vals and not all(np.isnan(lmmr_vals)):
                    lmmr_df = pd.DataFrame({
                        'Date': pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS'),
                        'LMMR': lmmr_vals
                    })
            except Exception as e:
                pass  # LMMR model not available

            # LMMR Hybrid (ЛУЧШАЯ LMMR! MAE 0.310, -37.4% vs SARIMA)
            lmmr_hybrid_df = None
            try:
                from sirena.models.lmmr_hybrid import LMMRHybridForecaster

                lmmr_hybrid = LMMRHybridForecaster(alpha=0.5)
                lmmr_hybrid.fit(df)
                lmmr_hybrid_vals = []
                for h in range(horizon):
                    target_date = last_date + pd.DateOffset(months=h+1)
                    df_ext = df.copy()
                    df_ext.loc[target_date] = np.nan
                    try:
                        pred = lmmr_hybrid.predict(df_ext, target_date)['prediction'] - 100
                        lmmr_hybrid_vals.append(pred)
                    except:
                        lmmr_hybrid_vals.append(np.nan)
                if lmmr_hybrid_vals and not all(np.isnan(lmmr_hybrid_vals)):
                    lmmr_hybrid_df = pd.DataFrame({
                        'Date': pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS'),
                        'LMMR Hybrid': lmmr_hybrid_vals
                    })
            except Exception as e:
                pass  # LMMR Hybrid not available

            # v4.6: NGBoost Shock (ЛУЧШАЯ МОДЕЛЬ! MAE 0.298, -5.56% vs Ridge)
            ngboost_shock_df = None
            try:
                from sirena.models.ngboost_shock import NGBoostShockForecaster

                ngb_shock = NGBoostShockForecaster()
                ngb_shock.fit(df)
                ngb_shock_vals = []
                ngb_shock_ci_lower = []
                ngb_shock_ci_upper = []
                for h in range(horizon):
                    target_date = last_date + pd.DateOffset(months=h+1)
                    df_ext = df.copy()
                    df_ext.loc[target_date] = np.nan
                    try:
                        pred = ngb_shock.predict(df_ext, target_date)
                        ngb_shock_vals.append(pred['prediction'] - 100)
                        ngb_shock_ci_lower.append(pred.get('ci_lower', pred['prediction'] - 0.3) - 100)
                        ngb_shock_ci_upper.append(pred.get('ci_upper', pred['prediction'] + 0.3) - 100)
                    except:
                        ngb_shock_vals.append(np.nan)
                        ngb_shock_ci_lower.append(np.nan)
                        ngb_shock_ci_upper.append(np.nan)
                if ngb_shock_vals and not all(np.isnan(ngb_shock_vals)):
                    ngboost_shock_df = pd.DataFrame({
                        'Date': pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS'),
                        'NGBoost Shock': ngb_shock_vals,
                        'CI Lower': ngb_shock_ci_lower,
                        'CI Upper': ngb_shock_ci_upper
                    })
            except Exception as e:
                pass  # NGBoost Shock not available

        # Collect all available models with new weights (v4.7)
        # LMMR Hybrid: 15% (MAE 0.310), NGBoost Shock: 28%, Shock Dummies: 17%, NGBoost: 12%, Ridge Ext: 10%, остальные: 18%
        model_weights = {
            'LMMR Hybrid': (lmmr_hybrid_df['LMMR Hybrid'].values if lmmr_hybrid_df is not None else None, 0.15),  # ЛУЧШАЯ LMMR!
            'LMMR': (lmmr_df['LMMR'].values if lmmr_df is not None else None, 0.05),  # Claude LMMR (fallback)
            'NGBoost Shock': (ngboost_shock_df['NGBoost Shock'].values if ngboost_shock_df is not None else None, 0.28),
            'Shock Dummies': (shock_dummies_df['Shock Dummies'].values if shock_dummies_df is not None else None, 0.17),
            'NGBoost': (ngboost_df['NGBoost'].values if ngboost_df is not None else None, 0.12),
            'Ridge Ext': (ridge_ext_df['Ridge Ext'].values if ridge_ext_df is not None else None, 0.10),
            'Ridge': (forecast['MoM'].values, 0.08),
            'BVAR': (bvar_df['BVAR'].values if bvar_df is not None else None, 0.08),
            'LightGBM': (lightgbm_df['LightGBM'].values if lightgbm_df is not None else None, 0.05),
            'Bayes Ridge': (bayes_ridge_df['Bayes Ridge'].values if bayes_ridge_df is not None else None, 0.03),
            'Prophet': (prophet_df['Prophet'].values if prophet_df is not None else None, 0.02),
            'SARIMA': (sarima_df['SARIMA'].values if sarima_df is not None else None, 0.02),
            'ETS': (ets_df['ETS'].values if ets_df is not None else None, 0.01),
            'EBM': (ebm_df['EBM'].values if ebm_df is not None else None, 0.01),
        }

        # Calculate ensemble with available models
        ensemble_vals = None
        available_models = {k: v for k, v in model_weights.items() if v[0] is not None and len(v[0]) == horizon}

        if available_models:
            total_weight = sum(w for _, w in available_models.values())
            ensemble_vals = np.zeros(horizon)

            for name, (vals, weight) in available_models.items():
                normalized_weight = weight / total_weight
                ensemble_vals += vals * normalized_weight

            model_list = ', '.join(available_models.keys())
            st.caption(f"Ансамбль: {model_list} (всего {len(available_models)} моделей)")

        # Add Traces
        if ensemble_vals is not None:
            fig_fc.add_trace(go.Scatter(
                x=forecast['Date'], y=ensemble_vals,
                name='Ансамбль (v4.0)', line=dict(color='#8b5cf6', width=4)
            ))

        if bvar_df is not None:
            fig_fc.add_trace(go.Scatter(
                x=bvar_df['Date'], y=bvar_df['BVAR'],
                name='BVAR', line=dict(color='#f97316', width=2, dash='dot')
            ))

        if sarima_df is not None:
            fig_fc.add_trace(go.Scatter(
                x=sarima_df['Date'], y=sarima_df['SARIMA'],
                name='SARIMA', line=dict(color='#10b981', width=2, dash='dot')
            ))

        if lightgbm_df is not None:
            fig_fc.add_trace(go.Scatter(
                x=lightgbm_df['Date'], y=lightgbm_df['LightGBM'],
                name='LightGBM', line=dict(color='#ef4444', width=2, dash='dot')
            ))

        if prophet_df is not None:
            fig_fc.add_trace(go.Scatter(
                x=prophet_df['Date'], y=prophet_df['Prophet'],
                name='Prophet', line=dict(color='#3b82f6', width=2, dash='dot')
            ))

        if ets_df is not None:
            fig_fc.add_trace(go.Scatter(
                x=ets_df['Date'], y=ets_df['ETS'],
                name='ETS', line=dict(color='#a855f7', width=1, dash='dash')
            ))

        if ebm_df is not None:
            fig_fc.add_trace(go.Scatter(
                x=ebm_df['Date'], y=ebm_df['EBM'],
                name='EBM', line=dict(color='#ec4899', width=1, dash='dash')
            ))

        # v4.2: Ridge Extended (лучшая модель по MAE)
        if ridge_ext_df is not None:
            fig_fc.add_trace(go.Scatter(
                x=ridge_ext_df['Date'], y=ridge_ext_df['Ridge Ext'],
                name='Ridge Ext (v4.2)', line=dict(color='#1d4ed8', width=3)
            ))

        # v4.2: Bayesian Ridge with CI
        if bayes_ridge_df is not None:
            # CI band
            fig_fc.add_trace(go.Scatter(
                x=bayes_ridge_df['Date'], y=bayes_ridge_df['CI Upper'],
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            fig_fc.add_trace(go.Scatter(
                x=bayes_ridge_df['Date'], y=bayes_ridge_df['CI Lower'],
                mode='lines', line=dict(width=0), fill='tonexty',
                fillcolor='rgba(124, 58, 237, 0.15)',
                name='95% CI (Bayes)', hoverinfo='skip'
            ))
            fig_fc.add_trace(go.Scatter(
                x=bayes_ridge_df['Date'], y=bayes_ridge_df['Bayes Ridge'],
                name='Bayes Ridge (v4.2)', line=dict(color='#7c3aed', width=2, dash='dash')
            ))

        # v4.4: NGBoost (лучшая модель!) with CI
        if ngboost_df is not None:
            # CI band (красная полупрозрачная)
            fig_fc.add_trace(go.Scatter(
                x=ngboost_df['Date'], y=ngboost_df['CI Upper'],
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            fig_fc.add_trace(go.Scatter(
                x=ngboost_df['Date'], y=ngboost_df['CI Lower'],
                mode='lines', line=dict(width=0), fill='tonexty',
                fillcolor='rgba(220, 38, 38, 0.15)',
                name='90% CI (NGBoost)', hoverinfo='skip'
            ))
            fig_fc.add_trace(go.Scatter(
                x=ngboost_df['Date'], y=ngboost_df['NGBoost'],
                name='NGBoost (v4.4)', line=dict(color='#dc2626', width=3)
            ))

        # v4.6: Ridge Shock Dummies
        if shock_dummies_df is not None:
            fig_fc.add_trace(go.Scatter(
                x=shock_dummies_df['Date'], y=shock_dummies_df['Shock Dummies'],
                name='Shock Dummies (v4.6)', line=dict(color='#059669', width=2)
            ))

        # v4.6: NGBoost Shock (ЛУЧШАЯ МОДЕЛЬ!) with CI
        if ngboost_shock_df is not None:
            # CI band (зелёная полупрозрачная)
            fig_fc.add_trace(go.Scatter(
                x=ngboost_shock_df['Date'], y=ngboost_shock_df['CI Upper'],
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            fig_fc.add_trace(go.Scatter(
                x=ngboost_shock_df['Date'], y=ngboost_shock_df['CI Lower'],
                mode='lines', line=dict(width=0), fill='tonexty',
                fillcolor='rgba(16, 185, 129, 0.2)',
                name='90% CI (NGBoost Shock)', hoverinfo='skip'
            ))
            fig_fc.add_trace(go.Scatter(
                x=ngboost_shock_df['Date'], y=ngboost_shock_df['NGBoost Shock'],
                name='🏆 NGBoost Shock (v4.6)', line=dict(color='#10b981', width=4)
            ))

        # v4.7: LMMR Hybrid (ЛУЧШАЯ LMMR! MAE 0.310)
        if lmmr_hybrid_df is not None:
            fig_fc.add_trace(go.Scatter(
                x=lmmr_hybrid_df['Date'], y=lmmr_hybrid_df['LMMR Hybrid'],
                name='LMMR Hybrid (v4.7)', line=dict(color='#f59e0b', width=3)
            ))

        fig_fc.add_hline(y=4.0/12, line_dash="dash", line_color="green", annotation_text="Цель 4%")
        fig_fc.update_layout(height=450, hovermode="x unified", title="Траектория прогноза v4.7 (NGBoost Shock — лучшая, MAE 0.298; LMMR Hybrid — лучшая LMMR, MAE 0.310)")
        st.plotly_chart(fig_fc, use_container_width=True)
        
        # --- 2. LAST 12 MONTHS PERFORMANCE ---
        st.subheader("✅ Прогноз vs Факт (последние 12 мес)")
        
        # Get backtest data for the last 12 months
        @st.cache_data
        def get_recent_performance():
            # Run backtest on full history
            bt = model.backtest(df, start_date='2023-01-01')
            # Filter last 12 available points
            if not bt.empty:
                return bt.iloc[-12:]
            return pd.DataFrame()
            
        recent_perf = get_recent_performance()
        
        if not recent_perf.empty:
            # Calculate metrics for this specific period
            mae_12m = (recent_perf['actual'] - recent_perf['prediction']).abs().mean()
            
            col_p1, col_p2 = st.columns([3, 1])
            
            with col_p1:
                fig_perf = go.Figure()
                fig_perf.add_trace(go.Bar(
                    x=recent_perf['date'], y=recent_perf['actual']-100,
                    name='Факт', marker_color='#94a3b8'
                ))
                fig_perf.add_trace(go.Scatter(
                    x=recent_perf['date'], y=recent_perf['prediction']-100,
                    name='Прогноз модели', line=dict(color='#2563eb', width=3)
                ))
                # Add medians for context
                recent_medians = recent_perf['date'].dt.month.map(monthly_medians) - 100.0
                fig_perf.add_trace(go.Scatter(
                    x=recent_perf['date'], y=recent_medians,
                    name='Медиана', line=dict(color='gray', width=1, dash='dot')
                ))
                
                fig_perf.update_layout(height=350, title="Точность модели на последних данных")
                st.plotly_chart(fig_perf, use_container_width=True)
                
            with col_p2:
                st.metric("MAE (12 мес)", f"{mae_12m:.2f}")
                st.caption("Средняя ошибка на этом участке")
                
                # Table
                disp_perf = recent_perf[['date', 'actual', 'prediction']].copy()
                disp_perf['actual'] = disp_perf['actual'] - 100
                disp_perf['prediction'] = disp_perf['prediction'] - 100
                disp_perf['diff'] = disp_perf['actual'] - disp_perf['prediction']
                disp_perf['date'] = disp_perf['date'].dt.strftime('%Y-%m')
                st.dataframe(disp_perf.style.format({'actual': '{:.2f}', 'prediction': '{:.2f}', 'diff': '{:+.2f}'}), height=300)

        # --- 3. HISTORY ---
        st.subheader("📜 История инфляции (с 2019)")
        
        hist_start = pd.Timestamp('2019-01-01')
        hist_plot = df[df.index >= hist_start].copy()
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=hist_plot.index, y=hist_plot['Все товары и услуги']-100,
            name='ИПЦ (MoM)', line=dict(color='#334155')
        ))
        
        # Add outliers highlights
        outliers = hist_plot[hist_plot.index.year.isin([2022])]
        if not outliers.empty:
             fig_hist.add_trace(go.Scatter(
                x=outliers.index, y=outliers['Все товары и услуги']-100,
                mode='markers', name='Шок 2022', marker=dict(color='red', size=6)
            ))
            
        fig_hist.update_layout(height=300, title="Динамика инфляции")
        st.plotly_chart(fig_hist, use_container_width=True)
        
        st.subheader("Детальный прогноз (Таблица)")
        forecast_display = forecast.copy()
        forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')

        # Merge additional models (v4.4)
        if ensemble_vals is not None:
            forecast_display['Ensemble'] = ensemble_vals
        if ngboost_df is not None:
            forecast_display['NGBoost'] = ngboost_df['NGBoost'].values
        if ridge_ext_df is not None:
            forecast_display['Ridge Ext'] = ridge_ext_df['Ridge Ext'].values
        if bayes_ridge_df is not None:
            forecast_display['Bayes Ridge'] = bayes_ridge_df['Bayes Ridge'].values
        if bvar_df is not None:
            forecast_display['BVAR'] = bvar_df['BVAR'].values
        if sarima_df is not None:
            forecast_display['SARIMA'] = sarima_df['SARIMA'].values
        if lightgbm_df is not None:
            forecast_display['LightGBM'] = lightgbm_df['LightGBM'].values
        if prophet_df is not None:
            forecast_display['Prophet'] = prophet_df['Prophet'].values
        if ets_df is not None:
            forecast_display['ETS_Model'] = ets_df['ETS'].values
        if ebm_df is not None:
            forecast_display['EBM'] = ebm_df['EBM'].values

        format_dict = {
            'MoM': '{:+.2f}%',
            'Ensemble': '{:+.2f}%',
            'NGBoost': '{:+.2f}%',
            'Ridge Ext': '{:+.2f}%',
            'Bayes Ridge': '{:+.2f}%',
            'BVAR': '{:+.2f}%',
            'SARIMA': '{:+.2f}%',
            'LightGBM': '{:+.2f}%',
            'Prophet': '{:+.2f}%',
            'ETS_Model': '{:+.2f}%',
            'EBM': '{:+.2f}%',
            'MoM_Index': '{:.2f}',
            'Ridge': '{:.2f}',
            'ETS': '{:.2f}',
            'Shock': '{:.2f}',
            'Nowcast': '{:.2f}',
            'FX_Effect': '{:+.2f}'
        }
        # Filter only existing columns
        format_dict = {k: v for k, v in format_dict.items() if k in forecast_display.columns}

        st.dataframe(forecast_display.style.format(format_dict))

    with tab2:
        st.subheader("Прогноз внешних условий")
        
        # Load current state
        manual_data = load_manual_exog()
        
        # Prepare Data for Editor
        if manual_data and manual_data.get('source') == 'manual':
            current_exog_df = pd.DataFrame({
                'Date': manual_data['dates'],
                'USD': manual_data['USD'],
                'KeyRate': manual_data['KeyRate'],
                'RUONIA': manual_data['RUONIA']
            })
            st.info(f"Используется ручной прогноз (обновлен: {manual_data.get('updated_at')})")
        else:
            # Generate Auto
            auto_exog = generate_auto_exog_forecast(last_date, horizon, current_usd, current_key_rate)
            current_exog_df = pd.DataFrame({
                'Date': auto_exog['dates'],
                'USD': auto_exog['USD'],
                'KeyRate': auto_exog['KeyRate'],
                'RUONIA': auto_exog['RUONIA']
            })
            st.success("Используется автоматический прогноз (AR1)")

        # Editor
        edited_df = st.data_editor(current_exog_df, num_rows="dynamic", key='exog_editor')
        
        col_save, col_auto = st.columns(2)
        
        if col_save.button("💾 Сохранить изменения"):
            data_to_save = {
                'dates': edited_df['Date'].tolist(),
                'USD': edited_df['USD'].tolist(),
                'KeyRate': edited_df['KeyRate'].tolist(),
                'RUONIA': edited_df['RUONIA'].tolist()
            }
            save_manual_exog(data_to_save)
            st.rerun()
            
        if col_auto.button("🔄 Сбросить на Авто"):
            if os.path.exists('manual_exog_forecast.json'):
                os.remove('manual_exog_forecast.json')
            st.rerun()
            
        # Viz
        fig_ex = go.Figure()
        fig_ex.add_trace(go.Scatter(x=edited_df['Date'], y=edited_df['USD'], name='USD', line=dict(color='green')))
        fig_ex.add_trace(go.Scatter(x=edited_df['Date'], y=edited_df['KeyRate'], name='Key Rate', line=dict(color='red'), yaxis='y2'))
        fig_ex.update_layout(
            yaxis=dict(title='USD'),
            yaxis2=dict(title='Rate %', overlaying='y', side='right'),
            height=350,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_ex, use_container_width=True)

    with tab3:
        st.subheader("Сравнительный бэктест v4.7 (последние 12 месяцев + LMMR Hybrid)")

        # Версия кэша — изменить при добавлении новых моделей!
        BACKTEST_CACHE_VERSION = "v4.7.0_lmmr_hybrid"

        @st.cache_data
        def run_comparative_backtest_cached(_cache_version=BACKTEST_CACHE_VERSION):
            try:
                # Load BVAR data locally (Source of Truth for latest Actuals)
                bvar_df_full = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
                cols_to_fix = ['mom', 'Prod', 'Nonprod', 'Serv', 'usd_nom_i', 'Ruonia']
                for col in cols_to_fix:
                    if col in bvar_df_full.columns:
                        if bvar_df_full[col].dtype == object:
                            bvar_df_full[col] = bvar_df_full[col].astype(str).str.replace(',', '.')
                        bvar_df_full[col] = pd.to_numeric(bvar_df_full[col], errors='coerce')
                
                bvar_df_full['Date'] = pd.to_datetime(bvar_df_full['Date'], format='%d.%m.%Y', errors='coerce')
                if bvar_df_full['Date'].isna().any(): bvar_df_full['Date'] = pd.to_datetime(bvar_df_full['Date'])
                bvar_df_full['Date'] = bvar_df_full['Date'].dt.to_period('M').dt.to_timestamp()
                bvar_df_full = bvar_df_full.set_index('Date').sort_index()
                
                # Prepare BVAR vars (CPI is Actual Target)
                bvar_data = pd.DataFrame()
                bvar_data['CPI'] = bvar_df_full['mom'] - 100
                bvar_data['Food'] = bvar_df_full['Prod'] - 100
                bvar_data['NonFood'] = bvar_df_full['Nonprod'] - 100
                bvar_data['Services'] = bvar_df_full['Serv'] - 100
                bvar_data['USD'] = bvar_df_full['usd_nom_i'] - 100
                bvar_data['RUONIA'] = bvar_df_full['Ruonia']
                bvar_data = bvar_data.dropna()

                # Determine dates (last 12 months from data end)
                last_fact = bvar_data.index.max()
                test_dates = pd.date_range(end=last_fact, periods=12, freq='MS')
                
                results = []

                from sirena_arima import SirenaARIMA
                from sirena.models.bvar import BayesianVAR
                from sirena.models.ebm import EBMForecaster
                from sirena.models.ets import ETSForecaster
                from sirena.models.lightgbm import LightGBMForecaster
                from sirena.models.prophet import ProphetForecaster
                # v4.2 models
                from sirena.models.ridge_extended import RidgeExtendedForecaster
                from sirena.models.bayesian_ridge import BayesianRidgeForecaster
                try:
                    from sirena.models.catboost_model import CatBoostForecaster, CATBOOST_AVAILABLE
                except ImportError:
                    CATBOOST_AVAILABLE = False
                # v4.3 models
                from sirena.models.elasticnet import ElasticNetForecaster
                from sirena.models.huber import HuberForecaster
                # v4.4 models
                try:
                    from sirena.models.ngboost_model import NGBoostForecaster, NGBOOST_AVAILABLE
                except ImportError:
                    NGBOOST_AVAILABLE = False
                # v4.6 models
                from sirena.models.ridge_shock_dummies import RidgeShockDummiesForecaster
                try:
                    from sirena.models.ngboost_shock import NGBoostShockForecaster
                    NGBOOST_SHOCK_AVAILABLE = True
                except ImportError:
                    NGBOOST_SHOCK_AVAILABLE = False
                # v4.7 models
                from sirena.models.lmmr_hybrid import LMMRHybridForecaster

                # Local Ridge instance to avoid state issues
                ridge_local = SirenaKBR_v24()

                for date in test_dates:
                    cutoff = date - pd.DateOffset(months=1)

                    # Actual (from fresh data)
                    if date in bvar_data.index:
                        actual = bvar_data.loc[date, 'CPI']
                    else:
                        continue # Skip if no actual

                    # Ridge (train on df, handle missing recent data gracefully)
                    # If df doesn't have data up to cutoff, Ridge uses available history
                    train_r = df[df.index <= cutoff].copy()
                    train_r = train_r.dropna(subset=['Все товары и услуги'])

                    train_r_ext = train_r.copy()
                    train_r_ext.loc[date] = np.nan
                    try:
                        ridge_local.fit(train_r)
                        pred_r = ridge_local.predict(train_r_ext, date)['prediction'] - 100
                    except: pred_r = np.nan

                    # BVAR
                    try:
                        train_b = bvar_data[bvar_data.index <= cutoff].copy()
                        model_b = BayesianVAR(lags=4, lambda1=1.0, var_names=['CPI', 'Food', 'USD', 'RUONIA'])
                        model_b.fit(train_b, target_col='CPI')
                        fc = model_b.forecast_full(horizon=1)
                        pred_b = fc['median'][0, 0]
                    except: pred_b = np.nan

                    # SARIMA
                    try:
                        ts = train_r['Все товары и услуги'].dropna() - 100
                        model_s = SirenaARIMA()
                        model_s.fit_sarima(ts)
                        pred_s = model_s.forecast(1)['mean'].iloc[0]
                    except: pred_s = np.nan

                    # LightGBM (already returns MoM deviation)
                    try:
                        model_lgb = LightGBMForecaster()
                        model_lgb.fit(train_r, 'Все товары и услуги')
                        pred_lgb = model_lgb.forecast(horizon=1)[0]
                    except: pred_lgb = np.nan

                    # Prophet
                    try:
                        model_p = ProphetForecaster()
                        model_p.fit(train_r, 'Все товары и услуги')
                        pred_p = model_p.forecast(horizon=1)[0]
                    except: pred_p = np.nan

                    # ETS
                    try:
                        model_ets = ETSForecaster()
                        model_ets.fit(train_r, 'Все товары и услуги')
                        pred_ets = model_ets.forecast(horizon=1)[0]
                    except: pred_ets = np.nan

                    # EBM
                    try:
                        model_ebm = EBMForecaster()
                        model_ebm.fit(train_r, 'Все товары и услуги')
                        pred_ebm = model_ebm.forecast(horizon=1)[0] - 100
                    except: pred_ebm = np.nan

                    # v4.2: Ridge Extended
                    try:
                        model_rex = RidgeExtendedForecaster()
                        model_rex.fit(train_r, 'Все товары и услуги')
                        pred_rex = model_rex.predict(train_r_ext, date)['prediction'] - 100
                    except: pred_rex = np.nan

                    # v4.2: Bayesian Ridge
                    try:
                        model_br = BayesianRidgeForecaster()
                        model_br.fit(train_r, 'Все товары и услуги')
                        pred_br = model_br.predict_with_ci(train_r_ext, date)['prediction'] - 100
                    except: pred_br = np.nan

                    # v4.2: CatBoost
                    pred_cb = np.nan
                    if CATBOOST_AVAILABLE:
                        try:
                            model_cb = CatBoostForecaster()
                            model_cb.fit(train_r, 'Все товары и услуги')
                            pred_cb = model_cb.forecast(horizon=1)[0] - 100
                        except: pred_cb = np.nan

                    # v4.3: ElasticNet
                    try:
                        model_en = ElasticNetForecaster()
                        model_en.fit(train_r, 'Все товары и услуги')
                        pred_en = model_en.predict(train_r_ext, date)['prediction'] - 100
                    except: pred_en = np.nan

                    # v4.3: Huber
                    try:
                        model_hub = HuberForecaster()
                        model_hub.fit(train_r, 'Все товары и услуги')
                        pred_hub = model_hub.predict(train_r_ext, date)['prediction'] - 100
                    except: pred_hub = np.nan

                    # LMMR Claude Implementation (Linear Mixed Model Regression)
                    try:
                        from sirena.models.lmmr_claude import LMMRForecasterClaude
                        model_lmmr = LMMRForecasterClaude()
                        model_lmmr.fit(train_r, 'Все товары и услуги')
                        pred_lmmr = model_lmmr.predict(train_r_ext, date)['prediction'] - 100
                    except: pred_lmmr = np.nan

                    # v4.7: LMMR Hybrid (ЛУЧШАЯ LMMR! MAE 0.310)
                    try:
                        model_lmmr_hybrid = LMMRHybridForecaster(alpha=0.5)
                        model_lmmr_hybrid.fit(train_r, 'Все товары и услуги')
                        pred_lmmr_hybrid = model_lmmr_hybrid.predict(train_r_ext, date)['prediction'] - 100
                    except: pred_lmmr_hybrid = np.nan

                    # v4.4: NGBoost
                    pred_ngb = np.nan
                    if NGBOOST_AVAILABLE:
                        try:
                            model_ngb = NGBoostForecaster()
                            model_ngb.fit(train_r, 'Все товары и услуги')
                            pred_ngb = model_ngb.predict(train_r_ext, date)['prediction'] - 100
                        except: pred_ngb = np.nan

                    # v4.6: Ridge Shock Dummies
                    try:
                        model_shock = RidgeShockDummiesForecaster(use_2022_dummy=False)
                        model_shock.fit(train_r, 'Все товары и услуги')
                        pred_shock = model_shock.predict(train_r_ext, date)['prediction'] - 100
                    except: pred_shock = np.nan

                    # v4.6: NGBoost Shock (ЛУЧШАЯ МОДЕЛЬ! MAE 0.298)
                    pred_ngb_shock = np.nan
                    if NGBOOST_SHOCK_AVAILABLE:
                        try:
                            model_ngb_shock = NGBoostShockForecaster()
                            model_ngb_shock.fit(train_r, 'Все товары и услуги')
                            pred_ngb_shock = model_ngb_shock.predict(train_r_ext, date)['prediction'] - 100
                        except: pred_ngb_shock = np.nan

                    # Ensemble (7 models with weights, LMMR is separate)
                    preds = {
                        'Ridge': (pred_r, 0.40),
                        'BVAR': (pred_b, 0.20),
                        'LightGBM': (pred_lgb, 0.15),
                        'Prophet': (pred_p, 0.10),
                        'SARIMA': (pred_s, 0.05),
                        'ETS': (pred_ets, 0.05),
                        'EBM': (pred_ebm, 0.05)
                    }
                    valid_preds = {k: v for k, v in preds.items() if not np.isnan(v[0])}
                    if valid_preds:
                        total_w = sum(w for _, w in valid_preds.values())
                        pred_e = sum(p * w / total_w for p, w in valid_preds.values())
                    else:
                        pred_e = pred_r if not np.isnan(pred_r) else np.nan

                    results.append({
                        'Date': date,
                        'Actual': actual,
                        'Ridge': pred_r,
                        'Ridge Ext': pred_rex,
                        'Bayes Ridge': pred_br,
                        'CatBoost': pred_cb,
                        'ElasticNet': pred_en,
                        'Huber': pred_hub,
                        'NGBoost': pred_ngb,
                        'NGBoost Shock': pred_ngb_shock,
                        'LMMR': pred_lmmr,
                        'LMMR Hybrid': pred_lmmr_hybrid,
                        'Shock Dummies': pred_shock,
                        'BVAR': pred_b,
                        'SARIMA': pred_s,
                        'LightGBM': pred_lgb,
                        'Prophet': pred_p,
                        'ETS': pred_ets,
                        'EBM': pred_ebm,
                        'Ensemble': pred_e
                    })
                    
                return pd.DataFrame(results)
            except Exception as e:
                st.error(f"Backtest error: {e}")
                return pd.DataFrame()

        with st.spinner("Выполнение сравнительного бэктеста (16 моделей v4.7 + LMMR Hybrid)..."):
            bt_results = run_comparative_backtest_cached(BACKTEST_CACHE_VERSION)

        if not bt_results.empty:
            # Metrics for all models + ensemble (v4.7 added LMMR Hybrid)
            all_models = ['Ridge', 'Ridge Ext', 'Bayes Ridge', 'CatBoost', 'ElasticNet', 'Huber', 'NGBoost', 'NGBoost Shock', 'LMMR', 'LMMR Hybrid', 'Shock Dummies', 'BVAR', 'LightGBM', 'Prophet', 'SARIMA', 'ETS', 'EBM', 'Ensemble']
            metrics = []
            for m in all_models:
                if m in bt_results.columns:
                    valid_vals = bt_results[m].dropna()
                    if len(valid_vals) > 0:
                        mae = (bt_results.loc[valid_vals.index, 'Actual'] - valid_vals).abs().mean()
                        metrics.append((m, mae))

            # Sort by MAE
            metrics.sort(key=lambda x: x[1])

            # Display metrics in 4 rows (14 models)
            st.markdown("#### MAE по моделям (отсортировано)")
            cols1 = st.columns(4)
            cols2 = st.columns(4)
            cols3 = st.columns(4)
            cols4 = st.columns(4)
            for i, (m, mae) in enumerate(metrics):
                if i < 4:
                    cols1[i].metric(f"{m}", f"{mae:.3f}")
                elif i < 8:
                    cols2[i-4].metric(f"{m}", f"{mae:.3f}")
                elif i < 12:
                    cols3[i-8].metric(f"{m}", f"{mae:.3f}")
                else:
                    cols4[i-12].metric(f"{m}", f"{mae:.3f}")

            # Plot
            fig_bt = go.Figure()
            # KPI Zone
            fig_bt.add_trace(go.Scatter(
                x=bt_results['Date'], y=bt_results['Actual'] + 0.5,
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            fig_bt.add_trace(go.Scatter(
                x=bt_results['Date'], y=bt_results['Actual'] - 0.5,
                mode='lines', line=dict(width=0), fill='tonexty',
                fillcolor='rgba(150, 150, 150, 0.2)',
                name='KPI (±0.5 п.п.)', hoverinfo='skip'
            ))

            # Fact (Dots)
            fig_bt.add_trace(go.Scatter(
                x=bt_results['Date'], y=bt_results['Actual'],
                name='Факт', mode='markers',
                marker=dict(color='black', size=10, symbol='circle')
            ))

            # Model traces with colors (v4.7 added LMMR Hybrid)
            model_colors = {
                'Ridge': '#2563eb',
                'Ridge Ext': '#1d4ed8',  # v4.2: darker blue
                'Bayes Ridge': '#7c3aed',  # v4.2: purple
                'CatBoost': '#14b8a6',  # v4.2: teal
                'ElasticNet': '#0891b2',  # v4.3: cyan
                'Huber': '#ca8a04',  # v4.3: yellow/amber
                'NGBoost': '#dc2626',  # v4.4: red
                'NGBoost Shock': '#10b981',  # v4.6: emerald (ЛУЧШАЯ МОДЕЛЬ! MAE 0.298)
                'LMMR Hybrid': '#f59e0b',  # v4.7: amber (ЛУЧШАЯ LMMR! MAE 0.310)
                'Shock Dummies': '#059669',  # v4.6: green
                'BVAR': '#f97316',
                'LightGBM': '#ef4444',
                'Prophet': '#3b82f6',
                'SARIMA': '#10b981',
                'ETS': '#a855f7',
                'EBM': '#ec4899'
            }
            for m, color in model_colors.items():
                if m in bt_results.columns:
                    # v4.7: LMMR Hybrid + NGBoost Shock выделяем жирной линией как лучшие модели
                    dash_style = 'solid' if m in ['Ridge Ext', 'Bayes Ridge', 'ElasticNet', 'Huber', 'NGBoost', 'NGBoost Shock', 'LMMR Hybrid', 'Shock Dummies'] else 'dot'
                    width = 4 if m == 'NGBoost Shock' else (3 if m in ['LMMR Hybrid', 'Shock Dummies', 'NGBoost'] else (2 if m in ['Ridge Ext', 'Bayes Ridge', 'ElasticNet', 'Huber'] else 1))
                    fig_bt.add_trace(go.Scatter(
                        x=bt_results['Date'], y=bt_results[m],
                        name=m, line=dict(dash=dash_style, color=color, width=width)
                    ))

            # Ensemble (bold)
            fig_bt.add_trace(go.Scatter(
                x=bt_results['Date'], y=bt_results['Ensemble'],
                name='Ансамбль (7)', line=dict(color='#8b5cf6', width=3)
            ))
            fig_bt.update_layout(
                title="Сравнение точности моделей (1 мес горизонт)",
                height=500,
                legend=dict(orientation='h', yanchor='bottom', y=-0.3)
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            st.subheader("Детальные результаты")
            st.dataframe(bt_results.set_index('Date').style.format("{:.2f}"))

    with tab4:
        st.markdown("""
        ### 🛠 Методология модели СИРЕНА-КБР v2.4
        
        Модель представляет собой **гибридный ансамбль**, объединяющий машинное обучение (Ridge Regression) с экспертными правилами сезонности (ETS).
        
        #### 1. Архитектура
        *   **Базовая модель (ML):** Ridge Regression (`alpha=0.3`)
        *   **Сезонная модель (ETS):** Историческая норма по месяцам (исключая аномальные 2010 и 2022 годы).
        *   **Ансамблирование:** Адаптивное взвешивание. Для каждого месяца определен вес доверия к сезонности vs ML.
            *   *Пример:* В мае и сентябре (высокая волатильность) вес ETS = 0.9 (мы верим истории).
            *   *Пример:* В феврале и ноябре вес ETS = 0.0 (мы верим регрессии).
            
        #### 2. Признаки (Features)
        *   **Лаги:** `y_lag1`, `y_lag2`, `y_lag12`
        *   **Тренды:** Скользящее среднее `y_ma3`
        *   **Компоненты:** Лаги продовольствия, непродовольствия и услуг.
        *   **Сезонность:** `sin`/`cos` кодирование месяца + `seasonal_norm`.
        *   **Отклонения:** `deviation_lag1` (отклонение прошлого месяца от нормы).
        
        #### 3. Препроцессинг
        *   **Масштабирование:** `RobustScaler` (устойчив к выбросам).
        *   **Очистка:** Исключение лет-выбросов (2010, 2022) из обучающей выборки для расчета сезонных норм.
        
        #### 4. Nowcasting
        *   Используются недельные данные по плодоовощной продукции для корректировки прогноза на ближайший месяц (если наблюдается аномальный рост цен на "борщевой набор").
        """)

    with tab5:
        st.subheader("🔬 Тестирование гипотез Opus")
        st.markdown("""
        В процессе оптимизации были проверены следующие гипотезы, предложенные AI Opus:
        
        | Гипотеза | Статус | Результат | Комментарий |
        |----------|--------|-----------|-------------|
        | **Bias Correction** | ❌ Отклонено | MAE 0.3749 | Не дало улучшения на тестовой выборке |
        | **Adaptive Window** | ❌ Отклонено | MAE 0.3981 | Фиксированное окно (84 мес) хуже, чем Expanding |
        | **Спец. обработка (Май, Сен)** | ✅ Принято | **MAE 0.3333** | Высокие веса сезонности (ETS=0.9) для волатильных месяцев |
        | **RobustScaler** | ✅ Принято | **MAE 0.3317** | Лучшая устойчивость к выбросам 2022 года |
        
        **Итоговая архитектура v2.4:**
        - **Модель:** Hybrid (Ridge + Seasonal ETS)
        - **Веса:** Адаптивные по месяцам (янв/май/сен/окт - доминирует сезонность)
        - **Признаки:** Лаги + Тригонометрия + Компоненты
        """)
        
        st.subheader("Декомпозиция текущего прогноза")
        sel_month = st.selectbox("Месяц", forecast['Month'], key='decomp_month')
        row = forecast[forecast['Month'] == sel_month].iloc[0]
        
        # Waterfall
        st.info(f"В месяце {sel_month} вес сезонной компоненты (ETS): **{row['Weight_ETS']*100:.0f}%**")
        
        col_d1, col_d2 = st.columns(2)
        col_d1.metric("Ridge Прогноз", f"{row['Ridge']-100:.2f}%")
        col_d2.metric("ETS Прогноз", f"{row['ETS']-100:.2f}%")

    with st.expander("🖼 Галерея аналитических графиков"):
        st.caption("Статические отчеты из папки data/plots/")
        plots_dir = 'data/plots'
        if os.path.exists(plots_dir):
            plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
            plot_files.sort()
            
            cols = st.columns(2)
            for i, f in enumerate(plot_files):
                with cols[i % 2]:
                    st.image(os.path.join(plots_dir, f), caption=f, use_column_width=True)
        else:
            st.warning("Папка data/plots/ не найдена.")

    with tab6:
        st.subheader("Межрегиональный анализ")
        
        start_year = st.slider("Начало анализа (год)", 2010, 2024, 2010, key='reg_start_year')
        start_date = f"{start_year}-01-01"
        
        @st.cache_data
        def get_regional_analysis():
            try:
                from sirena_regions import SirenaRegions
                sr = SirenaRegions()
                sr.load_data()
                return sr
            except Exception as e:
                return None

        sr = get_regional_analysis()
        
        if sr:
            st.info(f"Анализ связей инфляции КБР с другими регионами РФ (данные с {start_year} года)")
            
            target_code = 7 # KBR
            
            # 1. Best Predictors Table
            st.markdown("#### 🏆 Лучшие регионы-предикторы (опережающие индикаторы)")
            predictors = sr.find_best_predictors(target_code)
            st.dataframe(predictors)
            
            # 2. Correlation Analysis
            st.markdown("#### 📊 Сравнение динамики")
            
            # Get list of regions for dropdown
            reg_names = sorted(list(sr.regions_map.values()))
            selected_reg_name = st.selectbox("Выберите регион для сравнения", reg_names, index=0)
            
            # Find code
            sel_code = next((k for k, v in sr.regions_map.items() if v == selected_reg_name), None)
            
            if sel_code:
                # Plot KBR vs Selected Region (CPI)
                df_reg = sr.data[sr.data['Item'] == 'CPI']
                
                # Filter by date
                df_reg = df_reg[df_reg['Date'] >= pd.Timestamp(start_date)]
                
                pivot = df_reg.pivot_table(index='Date', columns='Region_code', values='MoM')
                
                if sel_code in pivot.columns and target_code in pivot.columns:
                    common = pivot[[target_code, sel_code]].dropna()
                    
                    fig_reg = go.Figure()
                    fig_reg.add_trace(go.Scatter(x=common.index, y=common[target_code]-100, name='КБР', line=dict(color='#2563eb', width=2)))
                    fig_reg.add_trace(go.Scatter(x=common.index, y=common[sel_code]-100, name=selected_reg_name, line=dict(color='gray', width=1)))
                    
                    # Calculate correlation
                    if len(common) > 1:
                        corr = common.corr().iloc[0,1]
                        st.metric(f"Корреляция с КБР (ИПЦ)", f"{corr:.2f}")
                    
                    fig_reg.update_layout(height=400, title=f"Динамика инфляции: КБР vs {selected_reg_name}")
                    st.plotly_chart(fig_reg, use_container_width=True)
            
            # 3. Micro Analysis
            @st.cache_data
            def get_micro_analysis(start_d):
                try:
                    from sirena_micro_regions import SirenaMicroRegions
                    sm = SirenaMicroRegions()
                    sm.load_data()
                    return sm.find_micro_predictors(7, start_date=start_d)
                except:
                    return None

            micro_preds = get_micro_analysis(start_date)
            
            if micro_preds is not None:
                st.markdown("---")
                st.markdown("#### 🛒 Товарные лидеры (Micro-Predictors)")
                st.markdown(f"Регионы, изменение цен в которых с опережением на 1 месяц предсказывает цены в КБР (анализ с {start_year} года):")
                st.dataframe(micro_preds, use_container_width=True)
                
        else:
            st.error("Не удалось загрузить региональные данные. Запустите extract_regions.py")

    # --- TAB 7: INSIDER ---
    with tab7:
        st.subheader("🔒 Инсайдерский прогноз")
        st.info("Прогноз с учетом предварительных данных (inflation_data.csv)")

        # Load insider data
        @st.cache_data
        def load_insider_data():
            try:
                insider_path = 'data/inflation_data.csv'
                if not os.path.exists(insider_path):
                    insider_path = '/home/valalav/opus_forecast/data/inflation_data.csv'

                df_ins = pd.read_csv(insider_path, sep=';', decimal=',')
                df_ins['Date'] = pd.to_datetime(df_ins['Date'], format='%d.%m.%Y')
                df_ins = df_ins.set_index('Date').sort_index()
                # Convert mom from index (100.x) to MoM change
                df_ins['MoM'] = df_ins['mom'] - 100
                return df_ins
            except Exception as e:
                st.error(f"Ошибка загрузки: {e}")
                return None

        df_insider = load_insider_data()

        if df_insider is not None:
            # Show latest data
            st.markdown("#### 📊 Последние данные (инсайдер)")

            last_insider_date = df_insider.index.max()
            last_insider_mom = df_insider.loc[last_insider_date, 'MoM']

            col_i1, col_i2, col_i3 = st.columns(3)
            col_i1.metric("Последний месяц", last_insider_date.strftime('%B %Y'))
            col_i2.metric("ИПЦ MoM", f"{last_insider_mom:+.2f}%")
            col_i3.metric("Источник", "Инсайдер")

            # Show recent history
            st.markdown("#### 📈 Динамика (последние 12 месяцев)")
            recent = df_insider.tail(12)[['MoM', 'Prod', 'Nonprod', 'Serv']].copy()
            recent.columns = ['Всего', 'Продовольствие', 'Непродовольствие', 'Услуги']
            recent['Продовольствие'] = recent['Продовольствие'] - 100
            recent['Непродовольствие'] = recent['Непродовольствие'] - 100
            recent['Услуги'] = recent['Услуги'] - 100

            fig_ins = go.Figure()
            fig_ins.add_trace(go.Scatter(x=recent.index, y=recent['Всего'], name='Всего', line=dict(color='#2563eb', width=3)))
            fig_ins.add_trace(go.Scatter(x=recent.index, y=recent['Продовольствие'], name='Продовольствие', line=dict(color='#10b981', width=2)))
            fig_ins.add_trace(go.Scatter(x=recent.index, y=recent['Непродовольствие'], name='Непродовольствие', line=dict(color='#f97316', width=2)))
            fig_ins.add_trace(go.Scatter(x=recent.index, y=recent['Услуги'], name='Услуги', line=dict(color='#8b5cf6', width=2)))
            fig_ins.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_ins.update_layout(
                height=400,
                title="Инфляция MoM по компонентам",
                yaxis_title="MoM, %",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig_ins, use_container_width=True)

            # Build forecast from insider data
            st.markdown("---")
            st.markdown("#### 🔮 Прогноз с учетом инсайдерских данных")

            # Prepare data for Ridge model
            df_for_model = pd.DataFrame(index=df_insider.index)
            df_for_model['Все товары и услуги'] = df_insider['mom']
            df_for_model['Продовольственные товары'] = df_insider['Prod']
            df_for_model['Непродовольственные товары'] = df_insider['Nonprod']
            df_for_model['Услуги'] = df_insider['Serv']

            # Train model on insider data
            model_ins = SirenaKBR_v24()
            model_ins.fit(df_for_model)

            # Generate forecast
            horizon_ins = st.slider("Горизонт прогноза (месяцев)", 3, 24, 12, key="insider_horizon")

            forecast_dates_ins = pd.date_range(
                start=last_insider_date + pd.DateOffset(months=1),
                periods=horizon_ins,
                freq='MS'
            )

            # Use recursive forecast
            start_date_ins = last_insider_date + pd.DateOffset(months=1)
            fc_results_df = model_ins.predict_horizon(
                df_for_model, 
                start_date=start_date_ins, 
                horizon=horizon_ins
            )
            
            predictions_ins = fc_results_df['MoM'].values
            forecast_ins = pd.DataFrame({
                'Date': fc_results_df['Date'],
                'MoM': predictions_ins
            })

            # Also run BVAR with insider data
            try:
                from sirena.models.bvar import BayesianVAR
                bvar_cols = ['MoM', 'Prod', 'Nonprod', 'Serv']
                bvar_data = df_insider[['mom', 'Prod', 'Nonprod', 'Serv']].copy()
                bvar_data.columns = bvar_cols
                bvar_data['MoM'] = bvar_data['MoM'] - 100
                bvar_data['Prod'] = bvar_data['Prod'] - 100
                bvar_data['Nonprod'] = bvar_data['Nonprod'] - 100
                bvar_data['Serv'] = bvar_data['Serv'] - 100

                bvar_model = BayesianVAR(lags=4, lambda1=1.0, var_names=bvar_cols)
                bvar_model.fit(bvar_data, target_col='MoM')
                bvar_fc = bvar_model.forecast_full(horizon=horizon_ins)
                bvar_vals = bvar_fc['median'][:, 0]  # CPI column
            except:
                bvar_vals = None

            # Component BVAR
            try:
                from sirena_component_bvar import SirenaComponentBVAR
                comp_model = SirenaComponentBVAR()
                comp_model.fit(df_insider)
                comp_fc = comp_model.predict(horizon=horizon_ins)
                comp_vals = comp_fc['CPI'].values - 100
                weights = comp_model.get_weights()
                st.caption(f"⚖️ Веса компонентов: 🍞 {weights['Prod']:.2f}, 👕 {weights['Nonprod']:.2f}, 💇 {weights['Serv']:.2f}")
            except Exception as e:
                comp_vals = None

            # Calculate ensemble
            if bvar_vals is not None:
                ensemble_ins = 0.6 * np.array(predictions_ins) + 0.4 * bvar_vals
            else:
                ensemble_ins = np.array(predictions_ins)

            # Plot forecast
            fig_fc_ins = go.Figure()

            # History
            hist_12 = df_insider.tail(12)
            fig_fc_ins.add_trace(go.Scatter(
                x=hist_12.index, y=hist_12['MoM'],
                name='Факт (инсайдер)', line=dict(color='#64748b', width=2),
                mode='lines+markers'
            ))

            # Ridge forecast
            fig_fc_ins.add_trace(go.Scatter(
                x=forecast_ins['Date'], y=forecast_ins['MoM'],
                name='Ridge прогноз', line=dict(color='#2563eb', width=3),
                mode='lines+markers'
            ))

            # BVAR forecast
            if bvar_vals is not None:
                fig_fc_ins.add_trace(go.Scatter(
                    x=forecast_ins['Date'], y=bvar_vals,
                    name='BVAR прогноз', line=dict(color='#f97316', width=2, dash='dot')
                ))

            # Component BVAR forecast
            if comp_vals is not None:
                fig_fc_ins.add_trace(go.Scatter(
                    x=forecast_ins['Date'], y=comp_vals,
                    name='BVAR (Компонентный)', line=dict(color='#10b981', width=2, dash='dash')
                ))

            # Ensemble
            fig_fc_ins.add_trace(go.Scatter(
                x=forecast_ins['Date'], y=ensemble_ins,
                name='Ансамбль (60/40)', line=dict(color='#8b5cf6', width=4)
            ))

            fig_fc_ins.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_fc_ins.update_layout(
                height=500,
                title=f"Прогноз инфляции КБР (база: {last_insider_date.strftime('%B %Y')})",
                yaxis_title="MoM, %",
                legend=dict(orientation="h", yanchor="bottom", y=1.02)
            )
            st.plotly_chart(fig_fc_ins, use_container_width=True)

            # Summary table
            st.markdown("#### 📋 Таблица прогноза")
            summary_ins = pd.DataFrame({
                'Дата': forecast_ins['Date'].dt.strftime('%Y-%m'),
                'Ridge': [f"{v:.2f}%" for v in predictions_ins],
                'BVAR': [f"{v:.2f}%" for v in bvar_vals] if bvar_vals is not None else ['—'] * horizon_ins,
                'BVAR (Comp)': [f"{v:.2f}%" for v in comp_vals] if comp_vals is not None else ['—'] * horizon_ins,
                'Ансамбль': [f"{v:.2f}%" for v in ensemble_ins]
            })
            st.dataframe(summary_ins, use_container_width=True, hide_index=True)

            # YoY estimate
            st.markdown("---")
            st.markdown("#### 📊 Оценка годовой инфляции")

            # Calculate cumulative inflation for next 12 months
            cum_12m = (1 + np.array(ensemble_ins[:12]) / 100).prod() - 1
            st.metric(
                "Накопленная инфляция (12 мес)",
                f"{cum_12m * 100:.1f}%",
                delta=f"vs текущий YoY"
            )

            # USD Forecast Section
            st.markdown("---")
            st.markdown("#### 💵 Прогноз курса доллара")
            
            try:
                # Prepare data for USD model
                df_usd = pd.DataFrame()
                if 'usd_nom_i' in df_insider.columns:
                    df_usd['usd_nom_i'] = df_insider['usd_nom_i']
                    
                    usd_model = SirenaUSD()
                    usd_model.fit(df_usd)
                    
                    col_usd1, col_usd2 = st.columns(2)
                    with col_usd1:
                        usd_scen = st.selectbox(
                            "Сценарий курса:", 
                            ['base', 'flat', 'growth', 'strong_growth', 'appreciation'],
                            format_func=lambda x: {
                                'base': 'Базовый (ARIMA)', 
                                'flat': 'Фиксация (0%)', 
                                'growth': 'Ослабление (+1%/мес)',
                                'strong_growth': 'Сильное ослабление (+2%)',
                                'appreciation': 'Укрепление (-1%)'
                            }[x]
                        )
                    with col_usd2:
                        current_rate = st.number_input("Текущий курс (RUB/$):", value=100.0, step=0.1)
                    
                    # Predict Indices
                    usd_fc = usd_model.predict(horizon=horizon_ins, scenario=usd_scen)
                    
                    # Calculate Absolute Path
                    multipliers = usd_fc['USD_Index'].values / 100.0
                    abs_path = [current_rate]
                    for m in multipliers:
                        abs_path.append(abs_path[-1] * m)
                    
                    # Plot
                    fig_usd = go.Figure()
                    
                    # Forecast (Absolute)
                    fig_usd.add_trace(go.Scatter(
                        x=usd_fc['Date'], y=abs_path[1:], 
                        name='Прогноз (RUB)',
                        line=dict(color='green', width=3)
                    ))
                    
                    # Reconstruct history (last 12 months)
                    hist_idx = df_insider['usd_nom_i'].tail(12).values
                    hist_dates = df_insider.index[-12:]
                    hist_abs = []
                    curr = current_rate
                    for idx in reversed(hist_idx):
                        prev = curr / (idx / 100.0)
                        hist_abs.append(prev)
                        curr = prev
                    hist_abs = list(reversed(hist_abs))
                    
                    fig_usd.add_trace(go.Scatter(
                        x=hist_dates, y=hist_abs,
                        name='История (расчетная)',
                        line=dict(color='gray', dash='dot')
                    ))
                    
                    fig_usd.add_trace(go.Scatter(
                        x=[hist_dates[-1]], y=[current_rate],
                        mode='markers', name='Текущий', marker=dict(color='blue', size=8)
                    ))
                    
                    fig_usd.update_layout(height=300, title="Прогноз курса доллара (RUB/USD)", yaxis_title="Рублей за Доллар")
                    st.plotly_chart(fig_usd, use_container_width=True)
                else:
                    st.warning("Нет данных по курсу доллара (usd_nom_i)")
                
            except Exception as e:
                st.warning(f"Ошибка модели курса: {e}")

        else:
            st.warning("Файл inflation_data.csv не найден")

    # --- TAB 8: EBM INTERPRETATION ---
    with tab8:
        st.subheader("🔍 EBM: Интерпретация прогноза")
        st.info("""
        **EBM (Explainable Boosting Machine)** — интерпретируемая модель машинного обучения.

        Показывает вклад каждого признака в прогноз инфляции.
        """)

        # Load EBM model and get importance
        @st.cache_data
        def get_ebm_analysis():
            try:
                from sirena.models.ebm import EBMForecaster

                df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')
                if df_raw['MoM'].dtype == object:
                    df_raw['MoM'] = df_raw['MoM'].astype(str).str.replace(',', '.')
                df_raw['MoM'] = pd.to_numeric(df_raw['MoM'], errors='coerce')
                df_raw['Date'] = pd.to_datetime(df_raw['Day'], format='%d.%m.%Y', errors='coerce')
                df = df_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first').sort_index()

                model = EBMForecaster()
                model.fit(df, 'Все товары и услуги')
                importance = model.get_feature_importance()

                return importance, model
            except Exception as e:
                st.error(f"Ошибка EBM: {e}")
                return None, None

        importance_df, ebm_model = get_ebm_analysis()

        if importance_df is not None:
            # Feature Importance Chart
            st.markdown("#### 📊 Важность признаков")

            # Translate feature names
            feature_names_ru = {
                'y_lag1': 'ИПЦ (лаг 1 мес)',
                'y_lag2': 'ИПЦ (лаг 2 мес)',
                'y_lag12': 'ИПЦ (лаг 12 мес)',
                'y_ma3': 'Скользящее среднее (3 мес)',
                'month_sin': 'Сезонность (sin)',
                'month_cos': 'Сезонность (cos)',
                'food_lag1': 'Продовольствие (лаг 1)',
                'nonfood_lag1': 'Непродовольствие (лаг 1)',
                'services_lag1': 'Услуги (лаг 1)'
            }

            importance_df['feature_ru'] = importance_df['feature'].map(
                lambda x: feature_names_ru.get(x, x)
            )

            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                y=importance_df['feature_ru'],
                x=importance_df['importance'],
                orientation='h',
                marker_color='#2563eb'
            ))
            fig_imp.update_layout(
                height=400,
                title="Вклад признаков в прогноз EBM",
                xaxis_title="Важность",
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            # Interpretation
            st.markdown("#### 💡 Интерпретация")

            top_features = importance_df.head(3)

            col_e1, col_e2 = st.columns(2)

            with col_e1:
                st.markdown("**Топ-3 драйвера прогноза:**")
                for i, row in top_features.iterrows():
                    st.markdown(f"- **{row['feature_ru']}**: важность {row['importance']:.4f}")

            with col_e2:
                st.markdown("**Что это значит:**")
                if 'month_cos' in top_features['feature'].values or 'month_sin' in top_features['feature'].values:
                    st.markdown("- 🗓 Сезонность — ключевой фактор")
                if 'nonfood_lag1' in top_features['feature'].values:
                    st.markdown("- 🛒 Непродовольственные товары влияют сильно")
                if 'food_lag1' in top_features['feature'].values:
                    st.markdown("- 🍎 Продовольствие — важный компонент")
                if 'y_lag1' in top_features['feature'].values or 'y_lag2' in top_features['feature'].values:
                    st.markdown("- 📈 Инерция (прошлые значения) важна")

            # Model comparison
            st.markdown("---")
            st.markdown("#### 📈 EBM vs LSTM (бэктест)")

            st.markdown("""
            | Метрика | EBM | LSTM | Улучшение |
            |---------|-----|------|-----------|
            | MAE | **0.40** | 0.43 | -7% |
            | KPI (<0.5%) | **72%** | 68% | +4 п.п. |
            | Интерпретируемость | ✅ Полная | ❌ Нет | — |
            """)

            # Table
            st.markdown("---")
            st.markdown("#### 📋 Полная таблица важности")
            st.dataframe(
                importance_df[['feature', 'feature_ru', 'importance']].rename(columns={
                    'feature': 'Признак (EN)',
                    'feature_ru': 'Признак (RU)',
                    'importance': 'Важность'
                }),
                use_container_width=True,
                hide_index=True
            )

        else:
            st.error("Не удалось загрузить EBM модель")

else:
    st.error("Данные не загружены.")