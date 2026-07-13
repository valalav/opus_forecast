"""
Compare page functions for SIRENA-KBR Dashboard.

Allows side-by-side comparison of model forecasts with delta visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


# =============================================================================
# COMPARE FUNCTIONS
# =============================================================================


@st.cache_data
def load_precomputed_forecasts():
    """Load precomputed forecasts from JSON file."""
    try:
        with open("data/precomputed_forecasts.json", "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Ошибка загрузки прогнозов: {e}")
        return None


def render_compare_tab(df, last_date, ALL_MODELS, MONTH_NAMES_RU, MODEL_COLORS):
    """
    Render the Compare Models tab.

    Args:
        df: Historical data DataFrame
        last_date: Last date in data
        ALL_MODELS: List of all model names
        MONTH_NAMES_RU: List of month names in Russian
        MODEL_COLORS: Dictionary mapping model names to colors
    """
    st.subheader("🔍 Сравнение моделей")

    # Load precomputed forecasts
    forecast_data = load_precomputed_forecasts()
    if forecast_data is None:
        return

    # Get available models from forecasts
    available_models = forecast_data.get("forecasts", {}).keys()
    available_models = sorted([m for m in available_models if m in ALL_MODELS])

    if len(available_models) < 2:
        st.warning("Недостаточно моделей для сравнения. Требуется минимум 2 модели.")
        return

    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        model1 = st.selectbox(
            "Выберите модель 1", options=available_models, index=0, key="compare_model1"
        )
    with col2:
        # Start from index 1 to avoid selecting the same model
        idx2 = min(1, len(available_models) - 1)
        model2 = st.selectbox(
            "Выберите модель 2",
            options=available_models,
            index=idx2,
            key="compare_model2",
        )

    if model1 == model2:
        st.warning("Выберите две разные модели для сравнения.")
        return

    # Get forecast data
    forecasts1 = forecast_data["forecasts"].get(model1, [])
    forecasts2 = forecast_data["forecasts"].get(model2, [])

    if not forecasts1 or not forecasts2:
        st.error("Нет данных прогноза для выбранных моделей.")
        return

    # Create date range for forecasts
    horizon = len(forecasts1)
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS"
    )

    # Create DataFrame for comparison
    compare_df = pd.DataFrame(
        {
            "Date": forecast_dates,
            model1: forecasts1,
            model2: forecasts2,
            "Delta": np.array(forecasts1) - np.array(forecasts2),
        }
    )
    compare_df["Month"] = compare_df["Date"].dt.month

    # Display comparison table
    st.markdown("### 📋 Таблица сравнения")
    display_df = compare_df.copy()
    display_df["Месяц"] = display_df["Month"].apply(
        lambda x: MONTH_NAMES_RU[x - 1] if 0 < x <= 12 else str(x)
    )
    display_df = display_df[["Date", "Месяц", model1, model2, "Delta"]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Create side-by-side plot
    st.markdown("### 📊 График прогнозов")

    fig = go.Figure()

    # Historical data
    hist_df = df.tail(24).reset_index()
    if "Date" in hist_df.columns:
        hist_df = hist_df.rename(columns={"Date": "date_col"})
    else:
        hist_df = hist_df.reset_index()
        hist_df.columns = ["date_col", "Все товары и услуги"]

    fig.add_trace(
        go.Scatter(
            x=hist_df["date_col"],
            y=hist_df["Все товары и услуги"],
            name="История",
            line=dict(color="#999999", dash="solid", width=2),
            hovertemplate="%{x}<br>Инфляция: %{y:.2f}%",
        )
    )

    # Model 1 forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecasts1,
            name=model1,
            line=dict(color=MODEL_COLORS.get(model1, "#1f77b4"), width=3),
            hovertemplate="%{x}<br>{model1}: %{y:.2f}%".replace("{model1}", model1),
        )
    )

    # Model 2 forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecasts2,
            name=model2,
            line=dict(color=MODEL_COLORS.get(model2, "#ff7f0e"), width=3),
            hovertemplate="%{x}<br>{model2}: %{y:.2f}%".replace("{model2}", model2),
        )
    )

    fig.update_layout(
        title=f"Сравнение прогнозов: {model1} vs {model2}",
        xaxis_title="Дата",
        yaxis_title="Инфляция (% MoM)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Delta visualization
    st.markdown("### 📈 Разница (Delta)")
    st.markdown(f"**Положительные значения:** {model1} выше {model2}")
    st.markdown(f"**Отрицательные значения:** {model2} выше {model1}")

    fig_delta = go.Figure()

    fig_delta.add_trace(
        go.Bar(
            x=forecast_dates,
            y=compare_df["Delta"],
            name="Delta",
            marker_color=compare_df["Delta"].apply(
                lambda x: "#d62728" if x > 0 else "#2ca02c"
            ),
            hovertemplate="%{x}<br>Delta: %{y:.2f}%",
        )
    )

    # Add zero line
    fig_delta.add_hline(y=0, line_dash="dash", line_color="black")

    fig_delta.update_layout(
        title=f"Разница между прогнозами ({model1} - {model2})",
        xaxis_title="Дата",
        yaxis_title="Разница (%)",
        hovermode="x",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_delta, use_container_width=True)

    # Summary statistics
    st.markdown("### 📊 Статистика")
    col_stats1, col_stats2, col_stats3 = st.columns(3)

    with col_stats1:
        st.metric(label=f"Средний {model1}", value=f"{np.mean(forecasts1):.3f}%")

    with col_stats2:
        st.metric(label=f"Средний {model2}", value=f"{np.mean(forecasts2):.3f}%")

    with col_stats3:
        st.metric(
            label="Средняя Delta",
            value=f"{np.mean(compare_df['Delta']):.3f}%",
            delta=f"{np.mean(compare_df['Delta']):.3f}%",
        )

    # Volatility comparison
    st.markdown("### 📉 Волатильность прогнозов")
    col_vol1, col_vol2 = st.columns(2)

    with col_vol1:
        st.metric(label=f"Ст. отклонение {model1}", value=f"{np.std(forecasts1):.3f}%")

    with col_vol2:
        st.metric(label=f"Ст. отклонение {model2}", value=f"{np.std(forecasts2):.3f}%")
