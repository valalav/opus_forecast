"""
СИРЕНА-КБР v5.0: Dashboard
10 вкладок: 5 прогнозов (h=1,2,3,6,12) + 5 бэктестов
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
    'NGBoost', 'NGBoost_Shock', 'BVAR', 'SARIMA', 'LightGBM', 'Prophet',
    'ETS', 'EBM', 'CatBoost', 'Subcomp', 'Subcomp_Multi', 'Micro', 'Ensemble'
]

MODEL_COLORS = {
    'Ridge': '#1f77b4', 'Ridge_Ext': '#aec7e8', 'Bayes_Ridge': '#ff7f0e',
    'ElasticNet': '#ffbb78', 'Huber': '#2ca02c', 'Ridge_Shock': '#98df8a',
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
    page_title="СИРЕНА-КБР v5.0",
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

        # Try to get forecasts from multiple models
        models_to_try = ['Ridge', 'Prophet', 'BVAR', 'ETS']

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

            except Exception as e:
                pass  # Model unavailable

    if not forecasts:
        st.error("Не удалось рассчитать ни одной модели")
        return

    # Create chart
    fig = go.Figure()

    # Historical data (last 12 months)
    hist = df['Все товары и услуги'].iloc[-12:] - 100
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        name='Факт', mode='lines+markers',
        line=dict(color='black', width=3)
    ))

    # Forecasts
    for model_name, fc_vals in forecasts.items():
        color = MODEL_COLORS.get(model_name, '#888888')
        fig.add_trace(go.Scatter(
            x=dates, y=fc_vals,
            name=model_name, mode='lines+markers',
            line=dict(color=color, width=2)
        ))

    # Ensemble (simple average)
    if len(forecasts) > 1:
        ensemble = np.mean([forecasts[m] for m in forecasts], axis=0)
        fig.add_trace(go.Scatter(
            x=dates, y=ensemble,
            name='Ансамбль', mode='lines+markers',
            line=dict(color='#000000', width=4, dash='dash')
        ))

    fig.update_layout(
        title="Прогноз MoM инфляции на 12 месяцев",
        xaxis_title="Дата",
        yaxis_title="MoM инфляция (%)",
        height=500,
        hovermode='x unified'
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
        hovermode='x unified'
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
    st.title("📊 СИРЕНА-КБР v5.0")
    st.markdown(f"""
    **Система прогнозирования инфляции КБР**
    Последние данные: {last_date.strftime('%B %Y')}
    Модели: {len(ALL_MODELS)} | Горизонты: h=1, h=2, h=3, h=6, h=12
    """)

    # --- TABS ---
    tab_f1, tab_f2, tab_f3, tab_f6, tab_f12, tab_b1, tab_b2, tab_b3, tab_b6, tab_b12 = st.tabs([
        "🎯 Прогноз h=1", "🎯 Прогноз h=2", "🎯 Прогноз h=3", "🎯 Прогноз h=6", "📈 Прогноз h=12",
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
