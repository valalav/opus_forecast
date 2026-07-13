import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler, StandardScaler
import warnings
import os

from exog_functions import load_last_exog_values, generate_auto_exog_forecast, load_manual_exog, save_manual_exog
# --- SETUP ---
st.set_page_config(
    page_title="СИРЕНА-КБР v2.4: Прогноз Инфляции",
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
            from sirena_midas import SirenaMIDAS
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
                
            # Calculate Proxy Growth
            curr_basket = w_wide[curr_weeks].mean(axis=1).sum()
            prev_basket = w_wide[prev_weeks].mean(axis=1).sum()
            
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
        from sirena_bvar import BayesianVAR
        
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
        
        # Train (Analytical BVAR)
        # Using slightly looser priors to allow data to speak
        model = BayesianVAR(data, ['CPI', 'Food', 'USD', 'RUONIA'], lags=4)
        model.fit(lambda1=1.0, lambda2=0.5, lambda3=1.0)
        
        # Forecast
        fc = model.forecast(h=horizon, n_draws=2000)
        median_path = fc['median'][:, 0] # CPI is index 0
        
        last_date = data.index.max()
        dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')
        
        return pd.DataFrame({'Date': dates, 'BVAR': median_path})
        
    except Exception as e:
        st.warning(f"BVAR недоступен: {e}")
        return None

# --- SARIMA INTEGRATION ---
@st.cache_resource
def run_sarima_forecast(horizon=12):
    try:
        from sirena_arima import SirenaARIMA
        
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
        
        st.caption("Рыночные параметры:")
        current_usd = st.number_input("Текущий курс USD", 50.0, 150.0, 77.0, 0.5)
        current_key_rate = st.number_input("Ключевая ставка (%)", 5.0, 30.0, 21.0, 0.5)
        
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
    
    # Проверяем ручные значения, иначе генерируем авто
    manual_exog = load_manual_exog()
    if manual_exog and manual_exog.get('source') == 'manual':
        # Используем ручные значения
        usd_proj = manual_exog['USD'][:horizon]
        key_rate_proj = manual_exog['KeyRate'][:horizon]
        ruonia_proj = manual_exog['RUONIA'][:horizon]
        exog_source = "Ручной"
    else:
        # Генерируем AR(1) прогноз от последних фактов
        auto_exog = generate_auto_exog_forecast(last_date, horizon)
        usd_proj = auto_exog['USD']
        key_rate_proj = auto_exog['KeyRate']
        ruonia_proj = auto_exog['RUONIA']
        exog_source = "Авто (AR1)"
    
    # Добавляем FX шок поверх прогноза если задан
    if fx_shock_pct != 0:
        usd_proj = [u * (1 + fx_shock_pct/100) for u in usd_proj]
        exog_source += f" + FX шок {fx_shock_pct:+.1f}%"
    
    # --- DASHBOARD LAYOUT ---
    
    st.title("📊 СИРЕНА-КБР v2.4 (Final)")
    st.markdown("""
    **Финальная версия модели** с интеграцией новых данных из Access.  
    **MAE:** 0.3317 (Проверена и подтверждена как лучшая)
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Прогноз", "✅ Бэктест", "🛠 Методология", "🧠 История (Opus)", "🗺 Регионы"])
    
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
        
        # Calculate Alternative Models
        with st.spinner("Расчет ансамбля моделей..."):
            bvar_df = run_bvar_forecast(horizon, cutoff_date=last_date)
            sarima_df = run_sarima_forecast(horizon)
            
        # Calculate Ensemble
        ensemble_vals = None
        if bvar_df is not None and sarima_df is not None:
            # Weighted average: Ridge (0.6) + BVAR (0.3) + SARIMA (0.1)
            ridge_vals = forecast['MoM'].values
            bvar_vals = bvar_df['BVAR'].values
            sarima_vals = sarima_df['SARIMA'].values
            
            if len(ridge_vals) == len(bvar_vals) == len(sarima_vals):
                ensemble_vals = (ridge_vals * 0.6) + (bvar_vals * 0.3) + (sarima_vals * 0.1)

        # Add Traces
        if ensemble_vals is not None:
            fig_fc.add_trace(go.Scatter(
                x=forecast['Date'], y=ensemble_vals, 
                name='Ансамбль (v3.1)', line=dict(color='#8b5cf6', width=4)
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
        
        fig_fc.add_hline(y=4.0/12, line_dash="dash", line_color="green", annotation_text="Цель 4%")
        fig_fc.update_layout(height=400, hovermode="x unified", title="Траектория прогноза vs Норма")
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
        
        # Merge additional models
        if ensemble_vals is not None:
            forecast_display['Ensemble'] = ensemble_vals
        if bvar_df is not None:
            forecast_display['BVAR'] = bvar_df['BVAR'].values
        if sarima_df is not None:
            forecast_display['SARIMA'] = sarima_df['SARIMA'].values
            
        st.dataframe(forecast_display.style.format({
            'MoM': '{:+.2f}%',
            'Ensemble': '{:+.2f}%',
            'BVAR': '{:+.2f}%',
            'SARIMA': '{:+.2f}%',
            'MoM_Index': '{:.2f}',
            'Ridge': '{:.2f}',
            'ETS': '{:.2f}',
            'Shock': '{:.2f}',
            'Nowcast': '{:.2f}',
            'FX_Effect': '{:+.2f}'
        }))

    with tab2:
        st.subheader("Динамика отдельных компонентов")
        
        comp_select = st.multiselect("Выберите компоненты", ['Ridge', 'ETS', 'Nowcast', 'FX_Effect'], default=['Ridge', 'ETS'])
        
        fig_comp = go.Figure()
        for c in comp_select:
            fig_comp.add_trace(go.Scatter(x=forecast['Date'], y=forecast[c], name=c))
            
        fig_comp.update_layout(title="Компоненты прогноза (п.п.)", height=400)
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.subheader("Сценарий внешних условий (Экзогенные)")
        fig_exog = go.Figure()
        
        # USD
        fig_exog.add_trace(go.Scatter(x=exog_dates, y=usd_proj, name='Курс USD (прогноз)', 
                                     line=dict(color='#16a34a', width=2)))
        
        # Key Rate (on secondary axis or same if scaled? Rates are ~20, USD ~80. Better use dual axis)
        fig_exog.add_trace(go.Scatter(x=exog_dates, y=key_rate_proj, name='Ключевая ставка',
                                     line=dict(color='#dc2626', width=2, dash='dash'), yaxis='y2'))
        
        fig_exog.update_layout(
            title="Предпосылки сценария",
            yaxis=dict(title="Курс USD (руб)"),
            yaxis2=dict(title="Ставка (%)", overlaying='y', side='right'),
            height=400
        )
        st.plotly_chart(fig_exog, use_container_width=True)

    with tab3:
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

    with tab4:
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

    with tab5:
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

else:
    st.error("Данные не загружены.")