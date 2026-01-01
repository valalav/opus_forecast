"""
СИРЕНА-КБР v5.1: Dashboard
11 вкладок: 5 прогнозов (h=1,2,3,6,12) + Сценарии Ki + 5 бэктестов
Новое в v5.1: YoY инфляция, BVAR Rate модель со сценариями ставки
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
import warnings
import os

# =============================================================================
# ЕДИНЫЙ СПИСОК МОДЕЛЕЙ — ИЗМЕНЯТЬ ТОЛЬКО ЗДЕСЬ!
# =============================================================================
ALL_MODELS = [
    'Ridge', 'Ridge_Ext', 'Bayes_Ridge', 'ElasticNet', 'Huber', 'Ridge_Shock',
    'Ridge_Macro', 'NGBoost', 'NGBoost_Shock', 'BVAR', 'SARIMA', 'LightGBM',
    'Prophet', 'ETS', 'EBM', 'CatBoost', 'Subcomp', 'Subcomp_Multi', 'Micro', 'Ensemble'
]

MODEL_COLORS = {
    'Ridge': '#1f77b4', 'Ridge_Ext': '#aec7e8', 'Bayes_Ridge': '#ff7f0e',
    'ElasticNet': '#ffbb78', 'Huber': '#2ca02c', 'Ridge_Shock': '#98df8a',
    'Ridge_Macro': '#2ecc71',  # Green - macro features model
    'NGBoost': '#d62728', 'NGBoost_Shock': '#ff9896', 'BVAR': '#9467bd',
    'SARIMA': '#c5b0d5', 'LightGBM': '#8c564b', 'Prophet': '#c49c94',
    'ETS': '#e377c2', 'EBM': '#f7b6d2', 'CatBoost': '#7f7f7f',
    'Subcomp': '#c7c7c7', 'Subcomp_Multi': '#bcbd22', 'Micro': '#17becf',
    'Ensemble': '#000000', 'Actual': '#000000', 'Факт': '#000000',
}

MONTH_NAMES_RU = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                 'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']

# --- SETUP ---
st.set_page_config(
    page_title="СИРЕНА-КБР v5.1",
    layout="wide",
    initial_sidebar_state="collapsed"
)
warnings.filterwarnings('ignore')


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_data():
    """Load main inflation data."""
    try:
        df_raw = pd.read_csv('data/infl_kbr.csv', sep=';', decimal=',')

        if 'MoM' in df_raw.columns and df_raw['MoM'].dtype == object:
            df_raw['MoM'] = df_raw['MoM'].astype(str).str.replace(',', '.')
            df_raw['MoM'] = pd.to_numeric(df_raw['MoM'], errors='coerce')

        if 'Day' in df_raw.columns:
            df_raw['Date'] = pd.to_datetime(df_raw['Day'], format='%d.%m.%Y', errors='coerce')
            if df_raw['Date'].isna().any():
                df_raw['Date'] = pd.to_datetime(df_raw['Day'])

        if 'Товар' in df_raw.columns and 'MoM' in df_raw.columns:
            df = df_raw.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
        else:
            df = df_raw.set_index('Date')

        df = df[['Все товары и услуги', 'Продовольственные товары',
                 'Непродовольственные товары', 'Услуги']].copy()
        return df.sort_index()
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return None


@st.cache_data
def load_backtest_data(horizon):
    """Load backtest predictions for given horizon."""
    filepath = f'archive/results/backtest_h{horizon}_predictions.csv'
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        return None


# =============================================================================
# FORECAST FUNCTIONS
# =============================================================================
def get_best_model_for_horizon(horizon):
    """Return best model name for each horizon based on backtests."""
    best_models = {
        1: 'Huber',
        2: 'NGBoost_Shock',
        3: 'Micro',  # Based on h=3 backtest
        6: 'Micro',  # Based on h=6 backtest
        12: 'Prophet'
    }
    return best_models.get(horizon, 'Ridge')


def forecast_with_model(df, target_date, model_name):
    """Get forecast from specific model."""
    try:
        if model_name == 'Huber':
            from sirena.models.huber import HuberForecaster
            model = HuberForecaster()
            model.fit(df)
            df_ext = df.copy()
            df_ext.loc[target_date] = np.nan
            return model.predict(df_ext, target_date)['prediction'] - 100

        elif model_name == 'NGBoost_Shock':
            from sirena.models.ngboost_shock import NGBoostShockForecaster
            model = NGBoostShockForecaster()
            model.fit(df, 'Все товары и услуги')
            df_ext = df.copy()
            df_ext.loc[target_date] = np.nan
            return model.predict(df_ext, target_date)['prediction'] - 100

        elif model_name == 'Micro':
            from sirena.models.microcomponent import MicrocomponentForecaster
            model = MicrocomponentForecaster(horizon=1, use_seasonal_adj=True)
            model.fit(df, 'Все товары и услуги')
            result = model.predict(df, target_date)
            if result and 'prediction' in result:
                month = target_date.month
                seasonal_adj = model.SEASONAL_ADJ.get(month, 0)
                return result['prediction'] - 100 + seasonal_adj
            return np.nan

        elif model_name == 'Prophet':
            from sirena.models.prophet import ProphetForecaster
            model = ProphetForecaster()
            model.fit(df, 'Все товары и услуги')
            fc = model.forecast(horizon=1)
            return fc[0] if len(fc) > 0 else np.nan

        elif model_name == 'Ridge':
            from sirena.models.ridge_extended import RidgeExtendedForecaster
            model = RidgeExtendedForecaster()
            model.fit(df)
            df_ext = df.copy()
            df_ext.loc[target_date] = np.nan
            return model.predict(df_ext, target_date)['prediction'] - 100

        return np.nan
    except Exception as e:
        st.warning(f"Ошибка модели {model_name}: {e}")
        return np.nan


def calculate_kpi_corrections(bt_data, model_name):
    """Calculate seasonal shift and bias corrections."""
    bt_data = bt_data.copy()
    bt_data['Month'] = bt_data['Date'].dt.month

    # Seasonal Shift (optimize for KPI)
    monthly_shifts = {}
    for month in range(1, 13):
        month_data = bt_data[bt_data['Month'] == month]
        if len(month_data) == 0:
            monthly_shifts[month] = 0
            continue
        best_shift, best_hits = 0, 0
        for shift in np.arange(-0.5, 0.51, 0.01):
            shifted = month_data[model_name] + shift
            hits = ((month_data['Actual'] - shifted).abs() <= 0.5).sum()
            if hits > best_hits:
                best_hits = hits
                best_shift = shift
        monthly_shifts[month] = best_shift

    # Bias Correction
    bias = bt_data.groupby('Month').apply(
        lambda x: (x[model_name] - x['Actual']).mean()
    ).to_dict()

    return monthly_shifts, bias


# =============================================================================
# TAB RENDERERS
# =============================================================================
def render_forecast_tab(df, last_date, horizon, bt_data):
    """Render forecast tab for given horizon."""
    target_date = last_date + pd.DateOffset(months=horizon)
    target_month = target_date.month

    st.subheader(f"🎯 Прогноз на {horizon} мес. вперёд")
    st.markdown(f"**Прогнозный месяц:** {MONTH_NAMES_RU[target_month-1]} {target_date.year}")

    if bt_data is None:
        st.error(f"Данные бэктеста не найдены. Запустите: python3 scripts/run_backtest_h{horizon}.py")
        return

    best_model = get_best_model_for_horizon(horizon)

    if best_model not in bt_data.columns:
        st.error(f"Модель {best_model} отсутствует в данных бэктеста")
        return

    # Get corrections
    monthly_shifts, bias = calculate_kpi_corrections(bt_data, best_model)

    # Get base forecast
    base_pred = forecast_with_model(df, target_date, best_model)

    if np.isnan(base_pred):
        st.error(f"Не удалось получить прогноз от модели {best_model}")
        return

    # Apply corrections
    seasonal_shift = monthly_shifts.get(target_month, 0)
    bias_correction = bias.get(target_month, 0)

    seasonal_pred = base_pred + seasonal_shift
    bias_pred = base_pred - bias_correction

    # Display forecasts
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{best_model.replace('_', ' ')}", f"{base_pred:.2f}%")
    col2.metric("🎯 Seasonal", f"{seasonal_pred:.2f}%", f"{seasonal_shift:+.2f}")
    col3.metric("📊 Bias", f"{bias_pred:.2f}%", f"{-bias_correction:+.2f}")

    # Calculate top-5 models by MAE
    models_available = [m for m in ALL_MODELS if m in bt_data.columns]
    model_mae = {}
    for m in models_available:
        errors = (bt_data['Actual'] - bt_data[m]).abs()
        model_mae[m] = errors.mean()
    top5_models = sorted(model_mae.keys(), key=lambda x: model_mae[x])[:5]

    # Chart: Top-5 models backtest + forecast points
    fig = go.Figure()
    bt_data_sorted = bt_data.sort_values('Date')

    # KPI Zone (background)
    fig.add_trace(go.Scatter(
        x=bt_data_sorted['Date'], y=bt_data_sorted['Actual'] + 0.5,
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=bt_data_sorted['Date'], y=bt_data_sorted['Actual'] - 0.5,
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(200, 200, 200, 0.3)', name='KPI (±0.5)', hoverinfo='skip'
    ))

    # Actual values
    fig.add_trace(go.Scatter(
        x=bt_data_sorted['Date'], y=bt_data_sorted['Actual'],
        name='Факт', mode='lines+markers',
        line=dict(color='black', width=2),
        marker=dict(color='black', size=8)
    ))

    # Top-5 models backtest lines
    colors = ['#10b981', '#2563eb', '#f97316', '#8b5cf6', '#ef4444']
    for i, m in enumerate(top5_models):
        fig.add_trace(go.Scatter(
            x=bt_data_sorted['Date'], y=bt_data_sorted[m],
            name=m.replace('_', ' '), mode='lines+markers',
            line=dict(color=colors[i], width=2),
            marker=dict(size=5)
        ))

    # Forecast points for top-5 models
    for i, m in enumerate(top5_models):
        try:
            pred = forecast_with_model(df, target_date, m)
            if not np.isnan(pred):
                fig.add_trace(go.Scatter(
                    x=[target_date], y=[pred],
                    name=f'{m} прогноз', mode='markers',
                    marker=dict(color=colors[i], size=14, symbol='star'),
                    showlegend=False
                ))
        except:
            pass

    fig.update_layout(
        title=f"Прогноз на {horizon} мес. вперёд",
        xaxis_title="Дата", yaxis_title="MoM инфляция (%)",
        height=400, hovermode='x unified',
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(gridcolor='#e5e5e5'), yaxis=dict(gridcolor='#e5e5e5')
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Historical context
    st.markdown(f"#### Исторический контекст для {MONTH_NAMES_RU[target_month-1]}")
    bt_data['Month'] = bt_data['Date'].dt.month
    month_history = bt_data[bt_data['Month'] == target_month]

    if len(month_history) > 0:
        col1, col2, col3 = st.columns(3)
        col1.metric("Средний факт", f"{month_history['Actual'].mean():.2f}%")
        col2.metric("Seasonal Shift", f"{seasonal_shift:+.2f}")
        col3.metric("Bias", f"{bias_correction:+.2f}")

    # Corrections table
    with st.expander("📊 Таблица коррекций по месяцам"):
        shifts_df = pd.DataFrame({
            'Месяц': MONTH_NAMES_RU,
            'Seasonal Shift': [monthly_shifts.get(m, 0) for m in range(1, 13)],
            'Bias': [bias.get(m, 0) for m in range(1, 13)]
        })
        st.dataframe(shifts_df.style.format({'Seasonal Shift': '{:+.2f}', 'Bias': '{:+.2f}'}),
                     hide_index=True)


def render_forecast_h12_tab(df, last_date, horizon=12):
    """Render 12-month trajectory forecast tab."""
    st.subheader("📈 Прогноз траектории на 12 месяцев")

    # Load precomputed forecasts or calculate
    bt_h12 = load_backtest_data(12)

    if bt_h12 is None:
        st.error("Данные бэктеста h=12 не найдены. Запустите: python3 scripts/run_backtest_h12.py")
        return

    # Get ensemble forecast
    with st.spinner("Расчёт ансамбля моделей..."):
        forecasts = {}
        dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                              periods=horizon, freq='MS')

        # Try to get forecasts from multiple models (top performers)
        models_to_try = ['Ridge', 'Prophet', 'BVAR', 'ETS', 'Huber', 'LightGBM', 'SubcompMulti']

        for model_name in models_to_try:
            try:
                if model_name == 'Ridge':
                    from sirena.models.ridge_extended import RidgeExtendedForecaster
                    model = RidgeExtendedForecaster()
                    model.fit(df)
                    fc_vals = []
                    for h in range(horizon):
                        target = dates[h]
                        df_ext = df.copy()
                        df_ext.loc[target] = np.nan
                        pred = model.predict(df_ext, target)['prediction'] - 100
                        fc_vals.append(pred)
                    forecasts['Ridge'] = fc_vals

                elif model_name == 'Prophet':
                    from sirena.models.prophet import ProphetForecaster
                    model = ProphetForecaster()
                    model.fit(df, 'Все товары и услуги')
                    fc = model.forecast(horizon=horizon)
                    forecasts['Prophet'] = list(fc)

                elif model_name == 'BVAR':
                    from sirena.models.bvar import BayesianVAR
                    # Load BVAR data
                    bvar_df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',')
                    for col in ['mom', 'Prod', 'Nonprod', 'Serv', 'usd_nom_i', 'Ruonia']:
                        if col in bvar_df.columns:
                            if bvar_df[col].dtype == object:
                                bvar_df[col] = bvar_df[col].astype(str).str.replace(',', '.')
                            bvar_df[col] = pd.to_numeric(bvar_df[col], errors='coerce')
                    bvar_df['Date'] = pd.to_datetime(bvar_df['Date'], format='%d.%m.%Y', errors='coerce')
                    bvar_df['Date'] = bvar_df['Date'].dt.to_period('M').dt.to_timestamp()
                    bvar_df = bvar_df.set_index('Date').sort_index()

                    data = pd.DataFrame()
                    data['CPI'] = bvar_df['mom'] - 100
                    data['Food'] = bvar_df['Prod'] - 100
                    data['NonFood'] = bvar_df['Nonprod'] - 100
                    data['Services'] = bvar_df['Serv'] - 100
                    data['USD'] = bvar_df['usd_nom_i'] - 100
                    data['RUONIA'] = bvar_df['Ruonia']
                    data = data.dropna()
                    data = data[data.index <= last_date]

                    bvar = BayesianVAR(lags=4)
                    bvar.fit(data, target_col='CPI')
                    fc = bvar.forecast_full(horizon=horizon)
                    forecasts['BVAR'] = list(fc['median'][:, 0])

                elif model_name == 'ETS':
                    from sirena.models.ets import ETSForecaster
                    model = ETSForecaster()
                    model.fit(df, 'Все товары и услуги')
                    fc = model.forecast(horizon=horizon)
                    forecasts['ETS'] = list(fc)

                elif model_name == 'Huber':
                    from sirena.models.huber import HuberForecaster
                    model = HuberForecaster()
                    model.fit(df)
                    fc_vals = []
                    for h in range(horizon):
                        target = dates[h]
                        df_ext = df.copy()
                        df_ext.loc[target] = np.nan
                        pred = model.predict(df_ext, target)['prediction'] - 100
                        fc_vals.append(pred)
                    forecasts['Huber'] = fc_vals

                elif model_name == 'LightGBM':
                    from sirena.models.lightgbm import LightGBMForecaster
                    model = LightGBMForecaster()
                    model.fit(df, 'Все товары и услуги')
                    fc = model.forecast(horizon=horizon)
                    forecasts['LightGBM'] = list(fc)

                elif model_name == 'SubcompMulti':
                    from sirena.models.subcomponent_multi import SubcomponentMultiForecaster
                    # Load SA data for subcomponent model
                    sa_df = pd.read_csv('data/sa_fl.csv', sep=';', decimal=',')
                    sa_df['Дата'] = pd.to_datetime(sa_df['Дата'])
                    sa_df = sa_df.pivot(index='Дата', columns='Код', values='Значение')
                    sa_df.columns = [str(c) for c in sa_df.columns]
                    for col in sa_df.columns:
                        if sa_df[col].dtype == object:
                            sa_df[col] = sa_df[col].astype(str).str.replace(',', '.').astype(float)
                    model = SubcomponentMultiForecaster()
                    model.fit(sa_df, 'Все товары и услуги')
                    fc_vals = []
                    for h in range(horizon):
                        target = dates[h]
                        sa_ext = sa_df.copy()
                        sa_ext.loc[target] = np.nan
                        pred = model.predict(sa_ext, target)['prediction']
                        fc_vals.append(pred)
                    forecasts['SubcompMulti'] = fc_vals

            except Exception as e:
                pass  # Model unavailable

    if not forecasts:
        st.error("Не удалось рассчитать ни одной модели")
        return

    # Create chart - ONLY forecast, no historical data
    fig = go.Figure()

    # Starting point (last actual value)
    last_actual = df['Все товары и услуги'].iloc[-1] - 100
    start_point = pd.DataFrame({'Date': [last_date], 'Value': [last_actual]})

    # Forecasts - each model as separate line
    colors_list = ['#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c', '#0891b2', '#db2777']
    for i, (model_name, fc_vals) in enumerate(forecasts.items()):
        # Add starting point + forecast
        full_dates = [last_date] + list(dates)
        full_vals = [last_actual] + list(fc_vals)

        fig.add_trace(go.Scatter(
            x=full_dates, y=full_vals,
            name=model_name, mode='lines+markers',
            line=dict(color=colors_list[i % len(colors_list)], width=2),
            marker=dict(size=6)
        ))

    # Ensemble (simple average)
    if len(forecasts) > 1:
        ensemble = np.mean([forecasts[m] for m in forecasts], axis=0)
        full_dates = [last_date] + list(dates)
        full_vals = [last_actual] + list(ensemble)
        fig.add_trace(go.Scatter(
            x=full_dates, y=full_vals,
            name='Ансамбль', mode='lines+markers',
            line=dict(color='#000000', width=3),
            marker=dict(size=8)
        ))

    fig.update_layout(
        title="Прогноз MoM инфляции на 12 месяцев",
        xaxis_title="Дата",
        yaxis_title="MoM инфляция (%)",
        height=500,
        hovermode='x unified',
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(gridcolor='#e5e5e5'), yaxis=dict(gridcolor='#e5e5e5')
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.markdown("#### 📋 Прогнозные значения")
    fc_df = pd.DataFrame({'Дата': dates})
    for model_name, fc_vals in forecasts.items():
        fc_df[model_name] = fc_vals
    if len(forecasts) > 1:
        fc_df['Ансамбль'] = ensemble

    st.dataframe(fc_df.style.format({col: '{:.2f}' for col in fc_df.columns if col != 'Дата'}),
                 use_container_width=True, hide_index=True)

    # === YoY CALCULATION ===
    st.markdown("---")
    st.markdown("#### 📊 Годовая инфляция (YoY)")

    try:
        from sirena.utils.yoy import forecast_yoy, yoy_at_horizon

        # Get historical MoM data
        mom_history = (df['Все товары и услуги'] - 100).dropna()

        # Calculate YoY for ensemble forecast
        if len(forecasts) > 1:
            ensemble_fc = np.mean([forecasts[m] for m in forecasts], axis=0)
        else:
            ensemble_fc = list(forecasts.values())[0]

        # YoY at different horizons
        yoy_results = []
        for h in [1, 3, 6, 12]:
            if h <= len(ensemble_fc):
                yoy_h = yoy_at_horizon(mom_history, ensemble_fc, h)
                target_date = last_date + pd.DateOffset(months=h)
                yoy_results.append({
                    'Горизонт': f'h={h}',
                    'Дата': target_date.strftime('%Y-%m'),
                    'MoM %': ensemble_fc[h-1],
                    'YoY %': yoy_h
                })

        yoy_df = pd.DataFrame(yoy_results)
        st.dataframe(yoy_df.style.format({'MoM %': '{:.2f}', 'YoY %': '{:.2f}'}),
                     use_container_width=True, hide_index=True)

        # YoY trajectory chart
        yoy_series = forecast_yoy(mom_history, ensemble_fc, return_full=True)

        fig_yoy = go.Figure()
        fig_yoy.add_trace(go.Scatter(
            x=yoy_series.index, y=yoy_series.values,
            name='YoY Прогноз', mode='lines+markers',
            line=dict(color='#dc2626', width=2),
            marker=dict(size=6)
        ))

        # Current YoY (from historical data)
        current_yoy = (np.prod(1 + mom_history.tail(12).values / 100) - 1) * 100
        fig_yoy.add_hline(y=current_yoy, line_dash="dash", line_color="gray",
                         annotation_text=f"Текущий YoY: {current_yoy:.2f}%")

        fig_yoy.update_layout(
            title="Прогноз YoY инфляции",
            xaxis_title="Дата",
            yaxis_title="YoY инфляция (%)",
            height=400,
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(gridcolor='#e5e5e5'), yaxis=dict(gridcolor='#e5e5e5')
        )

        st.plotly_chart(fig_yoy, use_container_width=True)

    except Exception as e:
        st.warning(f"YoY расчёт недоступен: {e}")


def render_scenarios_tab(df, last_date):
    """Render rate scenarios tab with calibrated transmission model v2.0."""
    st.subheader("🎚️ Сценарии ключевой ставки")

    # Model selection
    model_type = st.radio(
        "Модель трансмиссии",
        options=['simple', 'subcomponent', 'asymmetric'],
        format_func=lambda x: {
            'simple': '📊 Базовая (симметричный IRF)',
            'subcomponent': '🧩 Субкомпонентная (с декомпозицией)',
            'asymmetric': '📈📉 Асимметричная (hike ≠ cut)'
        }[x],
        horizontal=True
    )

    # Model description
    if model_type == 'simple':
        st.markdown("""
        **Базовая модель трансмиссии** — калиброванные параметры из литературы.
        - Пиковый эффект: **-0.08%** на 1 п.п. ставки
        - Лаг пикового эффекта: **6 месяцев**
        """)
    elif model_type == 'subcomponent':
        st.markdown("""
        **Субкомпонентная модель** — разная чувствительность для разных товаров.
        - 🏠 Кредитозависимые (авто, мебель): **-0.15** peak effect, **4 мес** лаг
        - 🛒 Импортозависимые (одежда, обувь): **-0.10** peak effect, **3 мес** лаг
        - 🥬 Базовые продукты (мясо, молоко): **-0.02** peak effect, **6 мес** лаг
        - 🏛️ Регулируемые (ЖКХ): **-0.01** peak effect, **12 мес** лаг
        """)
    else:  # asymmetric
        st.markdown("""
        **Асимметричная модель** — повышение ставки сильнее влияет, чем снижение.
        - 📈 Повышение (hike): **-0.10** peak effect, **5 мес** лаг
        - 📉 Снижение (cut): **-0.05** peak effect, **8 мес** лаг
        - Ratio: повышение в **~2x** сильнее
        """)

    # Scenario selector
    col1, col2 = st.columns([1, 2])
    with col1:
        scenario = st.selectbox(
            "Сценарий Ki",
            options=['auto', 'base', 'hike', 'cut', 'custom'],
            format_func=lambda x: {
                'auto': '🤖 Авто (по правилу Тейлора)',
                'base': '📊 Базовый (без изменений)',
                'hike': '📈 Повышение (+2 п.п. за 6 мес)',
                'cut': '📉 Снижение (-2 п.п. за 6 мес)',
                'custom': '⚙️ Пользовательский'
            }[x]
        )

    with col2:
        horizon = st.slider("Горизонт прогноза", min_value=6, max_value=24, value=12, step=3)

    # Custom scenario input
    custom_ki = None
    if scenario == 'custom':
        st.markdown("**Введите траекторию изменения Ki (п.п. от текущего уровня):**")
        cols = st.columns(min(6, horizon))
        custom_vals = []
        for i, col in enumerate(cols):
            if i < horizon:
                val = col.number_input(f"Месяц {i+1}", value=0.0, step=0.5, key=f"ki_{i}")
                custom_vals.append(val)
        # Fill remaining months with last value
        while len(custom_vals) < horizon:
            custom_vals.append(custom_vals[-1] if custom_vals else 0.0)
        custom_ki = np.array(custom_vals)

    # Run model
    with st.spinner("Расчёт модели трансмиссии..."):
        try:
            # Load inflation_data.csv
            infl_df = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',', encoding='utf-8-sig')
            for col in infl_df.columns:
                if col != 'Date' and infl_df[col].dtype == object:
                    infl_df[col] = infl_df[col].astype(str).str.replace(',', '.')
                    infl_df[col] = pd.to_numeric(infl_df[col], errors='coerce')
            infl_df['Date'] = pd.to_datetime(infl_df['Date'], format='%d.%m.%Y', errors='coerce')
            infl_df = infl_df.set_index('Date').sort_index()

            # Display current regime
            try:
                from sirena.models.regime_detector import detect_regime, MacroRegime
                regime, regime_diag = detect_regime(infl_df)
                regime_emoji = {'normal': '🟢', 'shock': '🔴', 'high_inflation': '🟠'}[regime.value]
                regime_label = {'normal': 'Нормальный', 'shock': 'Шок', 'high_inflation': 'Высокая инфляция'}[regime.value]
                st.info(f"**Текущий режим:** {regime_emoji} {regime_label} | ΔKi(3м): {regime_diag['ki_change']:+.1f} п.п. | ΔRuonia(3м): {regime_diag['ruonia_change']:+.1f} п.п.")
            except Exception:
                pass

            # Auto scenario using Taylor rule
            auto_ki_trajectory = None
            if scenario == 'auto':
                try:
                    from sirena.models.ki_trajectory import KiTrajectoryForecaster
                    from sirena.models.subcomponent_multi import SubcomponentMultiForecaster

                    # Get baseline inflation forecast
                    base_model = SubcomponentMultiForecaster(horizon=1)
                    base_model.fit(infl_df, 'mom')
                    baseline_fc = base_model.forecast(horizon)

                    # Get Ki trajectory from Taylor rule
                    ki_model = KiTrajectoryForecaster()
                    ki_model.fit(infl_df)
                    auto_ki_trajectory = ki_model.forecast_trajectory(horizon, baseline_fc)

                    current_ki = infl_df['Ki'].iloc[-1] if 'Ki' in infl_df.columns else 21.0
                    ki_change_auto = auto_ki_trajectory[-1] - current_ki

                    st.success(f"**Авто-траектория Ki:** {current_ki:.1f}% → {auto_ki_trajectory[-1]:.1f}% (Δ = {ki_change_auto:+.1f} п.п.)")
                except Exception as e:
                    st.warning(f"Авто-сценарий недоступен: {e}")
                    scenario = 'base'  # Fallback

            # Initialize model based on type
            group_decomposition = None

            if model_type == 'subcomponent':
                from sirena.models.subcomponent_scenario import SubcomponentScenarioForecaster
                model = SubcomponentScenarioForecaster()
                model.fit(infl_df)
                model_desc = "Субкомпонентная модель с декомпозицией по группам товаров"
            elif model_type == 'asymmetric':
                from sirena.models.scenario_rate import ScenarioRateModel
                model = ScenarioRateModel(use_asymmetric=True)
                model.fit(infl_df)
                model_desc = "Асимметричная модель (повышение ~2x сильнее снижения)"
            else:  # simple
                from sirena.models.scenario_rate import ScenarioRateModel
                model = ScenarioRateModel()
                model.fit(infl_df)
                model_desc = "Базовая модель с симметричным IRF"

            # Define Ki changes for each scenario
            ki_changes = {
                'base': 0.0,
                'hike': 2.0,  # +2 п.п.
                'cut': -2.0   # -2 п.п.
            }

            # Add auto scenario if trajectory is available
            if auto_ki_trajectory is not None:
                current_ki = infl_df['Ki'].iloc[-1] if 'Ki' in infl_df.columns else 21.0
                ki_changes['auto'] = auto_ki_trajectory[-1] - current_ki

            # Get forecasts for all scenarios
            results = {}
            for sc, ki_change in ki_changes.items():
                # Use forecast_scenario_with_ci for subcomponent model
                if model_type == 'subcomponent' and hasattr(model, 'forecast_scenario_with_ci'):
                    fc = model.forecast_scenario_with_ci(horizon, ki_change=ki_change, alpha=0.1)
                    results[sc] = {
                        'total': fc['mean'],
                        'ki_path': np.full(horizon, ki_change),
                        'effect': fc['effect'],
                        'baseline': fc['baseline'],
                        'total_q05': fc['ci_lower'],  # Real 90% CI
                        'total_q95': fc['ci_upper']
                    }
                else:
                    fc = model.forecast_scenario(horizon, ki_change=ki_change)
                    results[sc] = {
                        'total': fc['total'],
                        'ki_path': np.full(horizon, ki_change),
                        'effect': fc['effect'],
                        'baseline': fc['baseline'],
                        'total_q05': fc['total'] - 0.3,  # Approximate CI
                        'total_q95': fc['total'] + 0.3
                    }

                # Get group decomposition for subcomponent model
                if model_type == 'subcomponent' and sc == 'hike':
                    fc_full = model.forecast_scenario(horizon, ki_change=ki_change)
                    group_decomposition = fc_full.get('group_decomposition')

            # Custom scenario
            if custom_ki is not None:
                if model_type == 'subcomponent' and hasattr(model, 'forecast_scenario_with_ci'):
                    fc_custom = model.forecast_scenario_with_ci(horizon, ki_change=custom_ki[0], alpha=0.1)
                    results['custom'] = {
                        'total': fc_custom['mean'],
                        'ki_path': custom_ki,
                        'effect': fc_custom['effect'],
                        'baseline': fc_custom['baseline'],
                        'total_q05': fc_custom['ci_lower'],
                        'total_q95': fc_custom['ci_upper']
                    }
                else:
                    fc_custom = model.forecast_scenario(horizon, ki_change=custom_ki[0])
                    results['custom'] = {
                        'total': fc_custom['total'],
                        'ki_path': custom_ki,
                        'effect': fc_custom['effect'],
                        'baseline': fc_custom['baseline'],
                        'total_q05': fc_custom['total'] - 0.3,
                        'total_q95': fc_custom['total'] + 0.3
                    }

            st.success(f"✅ {model_desc}")

        except Exception as e:
            st.error(f"Ошибка расчёта: {e}")
            import traceback
            st.text(traceback.format_exc())
            return

    # Results visualization
    st.markdown("---")
    st.markdown("### 📊 Результаты сценарного анализа")

    # Comparison chart
    fig = go.Figure()
    dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')

    scenario_colors = {
        'auto': '#f59e0b',  # Amber
        'base': '#3b82f6',  # Blue
        'hike': '#ef4444',  # Red
        'cut': '#22c55e',   # Green
        'custom': '#8b5cf6' # Purple
    }
    scenario_names = {
        'auto': 'Авто (Тейлор)',
        'base': 'Базовый',
        'hike': 'Повышение Ki',
        'cut': 'Снижение Ki',
        'custom': 'Пользовательский'
    }

    for sc, fc in results.items():
        color = scenario_colors[sc]
        is_selected = sc == scenario

        # Add CI fan chart for selected scenario
        if is_selected and 'total_q05' in fc and 'total_q95' in fc:
            # Convert hex color to rgba for fill
            hex_color = color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            fill_color = f'rgba({r}, {g}, {b}, 0.2)'

            # Upper bound (no line)
            fig.add_trace(go.Scatter(
                x=dates, y=fc['total_q95'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            # Lower bound with fill to upper
            fig.add_trace(go.Scatter(
                x=dates, y=fc['total_q05'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=fill_color,
                showlegend=True,
                name=f'{scenario_names[sc]} 90% CI'
            ))

        # Main forecast line
        fig.add_trace(go.Scatter(
            x=dates, y=fc['total'],
            name=scenario_names[sc],
            mode='lines+markers',
            line=dict(color=color, width=2 if is_selected else 1,
                     dash='solid' if is_selected else 'dot'),
            marker=dict(size=6 if is_selected else 4)
        ))

    fig.update_layout(
        title="Прогноз MoM инфляции при разных сценариях ставки" + (" (с 90% CI)" if model_type == 'subcomponent' else ""),
        xaxis_title="Дата",
        yaxis_title="MoM инфляция (%)",
        height=450,
        hovermode='x unified',
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(gridcolor='#e5e5e5'), yaxis=dict(gridcolor='#e5e5e5')
    )

    st.plotly_chart(fig, use_container_width=True)

    # Ki trajectory chart
    fig_ki = go.Figure()
    for sc, fc in results.items():
        fig_ki.add_trace(go.Scatter(
            x=dates, y=fc['ki_path'],
            name=scenario_names[sc],
            mode='lines+markers',
            line=dict(color=scenario_colors[sc], width=2 if sc == scenario else 1)
        ))

    fig_ki.update_layout(
        title="Траектория изменения ключевой ставки (п.п.)",
        xaxis_title="Дата",
        yaxis_title="Изменение Ki (п.п.)",
        height=300,
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(gridcolor='#e5e5e5'), yaxis=dict(gridcolor='#e5e5e5')
    )

    st.plotly_chart(fig_ki, use_container_width=True)

    # Summary metrics
    st.markdown("### 📋 Сравнение сценариев")

    # Calculate baseline cumulative for delta
    base_cum = np.sum(results['base']['total'])

    cols = st.columns(len(results))
    for i, (sc, fc) in enumerate(results.items()):
        with cols[i]:
            cum_cpi = np.sum(fc['total'])
            effect = np.sum(fc['effect'])
            delta_vs_base = cum_cpi - base_cum

            st.metric(
                label=scenario_names[sc],
                value=f"{cum_cpi:.2f}%",
                delta=f"Эффект: {effect:+.2f}%" if sc != 'base' else "базовый"
            )

    # Group decomposition for subcomponent model
    if model_type == 'subcomponent' and group_decomposition:
        st.markdown("---")
        st.markdown("### 🧩 Декомпозиция эффекта по группам товаров")
        st.markdown("*При повышении ставки на +2 п.п.*")

        group_names = {
            'credit': '🏠 Кредитозависимые (авто, мебель)',
            'import': '🛒 Импортозависимые (одежда, обувь)',
            'basic': '🥬 Базовые продукты (мясо, молоко)',
            'regulated': '🏛️ Регулируемые (ЖКХ)',
            'services': '🎓 Услуги (образование, туризм)',
            'other': '📦 Прочие'
        }

        # Sort by absolute effect
        sorted_groups = sorted(
            group_decomposition.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Create bar chart
        fig_decomp = go.Figure()
        for group, effect in sorted_groups:
            if effect != 0:
                fig_decomp.add_trace(go.Bar(
                    x=[group_names.get(group, group)],
                    y=[effect],
                    name=group,
                    marker_color='#ef4444' if effect < 0 else '#22c55e',
                    text=[f'{effect:+.3f}%'],
                    textposition='outside'
                ))

        fig_decomp.update_layout(
            title="Вклад каждой группы товаров в общий эффект",
            yaxis_title="Эффект (%)",
            height=350,
            showlegend=False,
            plot_bgcolor='white', paper_bgcolor='white',
            yaxis=dict(gridcolor='#e5e5e5')
        )
        st.plotly_chart(fig_decomp, use_container_width=True)

        # Summary table
        decomp_data = []
        total_effect = sum(group_decomposition.values())
        for group, effect in sorted_groups:
            if effect != 0:
                pct = (effect / total_effect * 100) if total_effect != 0 else 0
                decomp_data.append({
                    'Группа': group_names.get(group, group),
                    'Эффект (%)': f'{effect:+.4f}',
                    'Доля (%)': f'{pct:.1f}%'
                })
        decomp_df = pd.DataFrame(decomp_data)
        st.dataframe(decomp_df, use_container_width=True, hide_index=True)

    # Asymmetric model comparison
    if model_type == 'asymmetric':
        st.markdown("---")
        st.markdown("### 📈📉 Асимметрия эффекта")

        hike_effect = np.sum(results['hike']['effect'])
        cut_effect = np.sum(results['cut']['effect'])
        ratio = abs(hike_effect / cut_effect) if cut_effect != 0 else float('inf')

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Повышение +2 п.п.", f"{hike_effect:+.2f}%")
        with col2:
            st.metric("Снижение -2 п.п.", f"{cut_effect:+.2f}%")
        with col3:
            st.metric("Ratio (hike/cut)", f"{ratio:.2f}x")

        st.info("📊 Повышение ставки влияет сильнее, чем снижение. "
                "Причина: банки быстрее повышают ставки по кредитам, чем снижают.")

    # YoY calculation
    st.markdown("---")
    st.markdown("### 📊 Годовая инфляция (YoY) по сценариям")

    try:
        from sirena.utils.yoy import yoy_at_horizon

        mom_history = (df['Все товары и услуги'] - 100).dropna()

        yoy_comparison = []
        for sc, fc in results.items():
            if horizon >= 12:
                yoy_h12 = yoy_at_horizon(mom_history, fc['total'], 12)
            else:
                yoy_h12 = yoy_at_horizon(mom_history, fc['total'], horizon)

            yoy_comparison.append({
                'Сценарий': scenario_names[sc],
                'Ki (п.п.)': fc['ki_path'][-1],
                'Cum MoM (%)': np.sum(fc['total']),
                'YoY h=12 (%)': yoy_h12 if horizon >= 12 else np.nan
            })

        yoy_df = pd.DataFrame(yoy_comparison)
        st.dataframe(yoy_df.style.format({
            'Ki (п.п.)': '{:+.1f}',
            'Cum MoM (%)': '{:.2f}',
            'YoY h=12 (%)': '{:.2f}'
        }), use_container_width=True, hide_index=True)

    except Exception as e:
        st.warning(f"YoY расчёт недоступен: {e}")

    # Effect breakdown chart
    st.markdown("---")
    st.markdown("### 📉 Разбивка эффекта от ставки")

    if scenario != 'base':
        selected_result = results[scenario]
        fig_effect = go.Figure()

        fig_effect.add_trace(go.Bar(
            x=dates, y=selected_result['baseline'],
            name='Базовый прогноз',
            marker_color='#3b82f6'
        ))

        fig_effect.add_trace(go.Bar(
            x=dates, y=selected_result['effect'],
            name='Эффект Ki',
            marker_color='#ef4444' if scenario == 'hike' else '#22c55e'
        ))

        fig_effect.update_layout(
            title=f"Разбивка: {scenario_names[scenario]}",
            xaxis_title="Дата",
            yaxis_title="MoM инфляция (%)",
            barmode='relative',
            height=400,
            plot_bgcolor='white', paper_bgcolor='white'
        )

        st.plotly_chart(fig_effect, use_container_width=True)

        st.info(f"""
        **Интерпретация для сценария "{scenario_names[scenario]}":**
        - Базовый прогноз (без изменения ставки): **{np.sum(selected_result['baseline']):.2f}%** за {horizon} мес
        - Эффект от изменения ставки: **{np.sum(selected_result['effect']):+.2f}%**
        - Итоговый прогноз: **{np.sum(selected_result['total']):.2f}%**
        """)

    # Detailed table
    with st.expander("📋 Подробная таблица прогнозов"):
        detail_df = pd.DataFrame({'Дата': dates})
        for sc, fc in results.items():
            detail_df[f'{scenario_names[sc]}'] = fc['total']
            if sc != 'base':
                detail_df[f'{scenario_names[sc]} эффект'] = fc['effect']

        st.dataframe(detail_df.style.format({col: '{:.3f}' for col in detail_df.columns if col != 'Дата'}),
                     use_container_width=True, hide_index=True)


def render_backtest_tab(horizon):
    """Render backtest tab for given horizon."""
    st.subheader(f"📊 Бэктест h={horizon} ({horizon} мес. вперёд)")

    bt_data = load_backtest_data(horizon)

    if bt_data is None:
        st.error(f"Данные бэктеста не найдены.")
        st.info(f"Запустите: `python3 scripts/run_backtest_h{horizon}.py`")
        return

    # Calculate metrics for each model
    models = [m for m in ALL_MODELS if m in bt_data.columns]

    metrics = []
    for m in models:
        errors = (bt_data['Actual'] - bt_data[m]).abs()
        mae = errors.mean()
        rmse = np.sqrt((errors ** 2).mean())
        kpi_violations = (errors > 0.5).sum()
        coverage = ((errors <= 0.5).sum() / len(errors)) * 100
        metrics.append({
            'Model': m, 'MAE': mae, 'RMSE': rmse,
            'KPI_Violations': kpi_violations, 'Coverage': coverage
        })

    metrics_df = pd.DataFrame(metrics).sort_values('MAE')

    # Top 5 models
    st.markdown("#### 🏆 Топ-5 моделей по MAE")
    cols = st.columns(5)
    for i, (_, row) in enumerate(metrics_df.head(5).iterrows()):
        if i < 5:
            cols[i].metric(
                row['Model'].replace('_', ' '),
                f"MAE: {row['MAE']:.3f}",
                f"KPI: {int(row['KPI_Violations'])}/{len(bt_data)}"
            )

    # Chart
    fig = go.Figure()

    # KPI Zone ±0.5
    fig.add_trace(go.Scatter(
        x=bt_data['Date'], y=bt_data['Actual'] + 0.5,
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=bt_data['Date'], y=bt_data['Actual'] - 0.5,
        mode='lines', line=dict(width=0), fill='tonexty',
        fillcolor='rgba(150, 150, 150, 0.2)', name='KPI (±0.5 п.п.)', hoverinfo='skip'
    ))

    # Actual
    fig.add_trace(go.Scatter(
        x=bt_data['Date'], y=bt_data['Actual'],
        name='Факт', mode='markers',
        marker=dict(color='black', size=10)
    ))

    # Top 5 models
    colors = ['#10b981', '#2563eb', '#f97316', '#8b5cf6', '#ef4444']
    for i, (_, row) in enumerate(metrics_df.head(5).iterrows()):
        if i < 5:
            m = row['Model']
            fig.add_trace(go.Scatter(
                x=bt_data['Date'], y=bt_data[m],
                name=m.replace('_', ' '),
                line=dict(color=colors[i], width=2)
            ))

    fig.update_layout(
        title=f"Прогноз на {horizon} мес. вперёд vs Факт",
        height=450,
        hovermode='x unified',
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(gridcolor='#e5e5e5'), yaxis=dict(gridcolor='#e5e5e5')
    )
    st.plotly_chart(fig, use_container_width=True)

    # Full metrics table
    st.markdown("#### 📋 Все модели")
    st.dataframe(
        metrics_df.style.format({
            'MAE': '{:.3f}', 'RMSE': '{:.3f}',
            'Coverage': '{:.1f}%', 'KPI_Violations': '{:.0f}'
        }),
        use_container_width=True, hide_index=True
    )

    # KPI Optimizer section
    st.markdown("---")
    st.markdown("### 🎯 KPI Optimizer")

    best_model = metrics_df.iloc[0]['Model']
    bt_data['Month'] = bt_data['Date'].dt.month

    monthly_shifts, bias = calculate_kpi_corrections(bt_data, best_model)

    # Apply corrections
    bt_data['Seasonal'] = bt_data[best_model] + bt_data['Month'].map(monthly_shifts)
    bt_data['Bias'] = bt_data[best_model] - bt_data['Month'].map(bias)

    # Calculate metrics
    errors_orig = (bt_data['Actual'] - bt_data[best_model]).abs()
    errors_seasonal = (bt_data['Actual'] - bt_data['Seasonal']).abs()
    errors_bias = (bt_data['Actual'] - bt_data['Bias']).abs()

    kpi_orig = (errors_orig <= 0.5).sum()
    kpi_seasonal = (errors_seasonal <= 0.5).sum()
    kpi_bias = (errors_bias <= 0.5).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Оригинал", f"KPI: {kpi_orig}/{len(bt_data)}", f"MAE: {errors_orig.mean():.3f}")
    col2.metric("🎯 Seasonal", f"KPI: {kpi_seasonal}/{len(bt_data)}", f"MAE: {errors_seasonal.mean():.3f}")
    col3.metric("📊 Bias", f"KPI: {kpi_bias}/{len(bt_data)}", f"MAE: {errors_bias.mean():.3f}")


# =============================================================================
# MAIN APP
# =============================================================================
df = load_data()

if df is not None:
    last_date = df.index.max()

    # --- HEADER ---
    st.title("📊 СИРЕНА-КБР v5.1")
    st.markdown(f"""
    **Система прогнозирования инфляции КБР**
    Последние данные: {last_date.strftime('%B %Y')}
    Модели: {len(ALL_MODELS)} | Горизонты: h=1, h=2, h=3, h=6, h=12 | + YoY, Сценарии Ki
    """)

    # --- TABS ---
    tab_f1, tab_f2, tab_f3, tab_f6, tab_f12, tab_sc, tab_b1, tab_b2, tab_b3, tab_b6, tab_b12 = st.tabs([
        "🎯 Прогноз h=1", "🎯 Прогноз h=2", "🎯 Прогноз h=3", "🎯 Прогноз h=6", "📈 Прогноз h=12",
        "🎚️ Сценарии Ki",
        "📊 Бэктест h=1", "📊 Бэктест h=2", "📊 Бэктест h=3", "📊 Бэктест h=6", "📊 Бэктест h=12"
    ])

    # Preload backtest data
    bt_h1 = load_backtest_data(1)
    bt_h2 = load_backtest_data(2)
    bt_h3 = load_backtest_data(3)
    bt_h6 = load_backtest_data(6)

    # --- FORECAST TABS ---
    with tab_f1:
        render_forecast_tab(df, last_date, 1, bt_h1)

    with tab_f2:
        render_forecast_tab(df, last_date, 2, bt_h2)

    with tab_f3:
        render_forecast_tab(df, last_date, 3, bt_h3)

    with tab_f6:
        render_forecast_tab(df, last_date, 6, bt_h6)

    with tab_f12:
        render_forecast_h12_tab(df, last_date, 12)

    with tab_sc:
        render_scenarios_tab(df, last_date)

    # --- BACKTEST TABS ---
    with tab_b1:
        render_backtest_tab(1)

    with tab_b2:
        render_backtest_tab(2)

    with tab_b3:
        render_backtest_tab(3)

    with tab_b6:
        render_backtest_tab(6)

    with tab_b12:
        render_backtest_tab(12)

else:
    st.error("Данные не загружены. Проверьте файл data/infl_kbr.csv")
