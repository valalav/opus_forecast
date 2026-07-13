"""Backtest tab functions for dashboard refactoring."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


@st.cache_data
def load_backtest_data(horizon):
    """Load backtest predictions for given horizon."""
    filepath = f"archive/results/backtest_h{horizon}_predictions.csv"
    try:
        df = pd.read_csv(filepath)
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception as e:
        return None


def render_backtest_tab(horizon, ALL_MODELS):
    """Render backtest tab for given horizon.

    Args:
        horizon: Forecast horizon (1, 2, 3, 6, or 12)
        ALL_MODELS: List of model names
    """
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
        errors = (bt_data["Actual"] - bt_data[m]).abs()
        mae = errors.mean()
        rmse = np.sqrt((errors**2).mean())
        kpi_violations = (errors > 0.5).sum()
        coverage = ((errors <= 0.5).sum() / len(errors)) * 100
        metrics.append(
            {
                "Model": m,
                "MAE": mae,
                "RMSE": rmse,
                "KPI_Violations": kpi_violations,
                "Coverage": coverage,
            }
        )

    metrics_df = pd.DataFrame(metrics).sort_values("MAE")

    # Top 5 models
    st.markdown("#### 🏆 Топ-5 моделей по MAE")
    cols = st.columns(5)
    for i, (_, row) in enumerate(metrics_df.head(5).iterrows()):
        if i < 5:
            cols[i].metric(
                row["Model"].replace("_", " "),
                f"MAE: {row['MAE']:.3f}",
                f"KPI: {int(row['KPI_Violations'])}/{len(bt_data)}",
            )

    # Chart
    fig = go.Figure()

    # KPI Zone ±0.5
    fig.add_trace(
        go.Scatter(
            x=bt_data["Date"],
            y=bt_data["Actual"] + 0.5,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bt_data["Date"],
            y=bt_data["Actual"] - 0.5,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(150, 150, 150, 0.2)",
            name="KPI (±0.5 п.п.)",
            hoverinfo="skip",
        )
    )

    # Actual
    fig.add_trace(
        go.Scatter(
            x=bt_data["Date"],
            y=bt_data["Actual"],
            name="Факт",
            mode="markers",
            marker=dict(color="black", size=10),
        )
    )

    # Top 5 models
    colors = ["#10b981", "#2563eb", "#f97316", "#8b5cf6", "#ef4444"]
    for i, (_, row) in enumerate(metrics_df.head(5).iterrows()):
        if i < 5:
            m = row["Model"]
            fig.add_trace(
                go.Scatter(
                    x=bt_data["Date"],
                    y=bt_data[m],
                    name=m.replace("_", " "),
                    line=dict(color=colors[i], width=2),
                )
            )

    fig.update_layout(
        title=f"Прогноз на {horizon} мес. вперёд vs Факт",
        height=450,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#e5e5e5"),
        yaxis=dict(gridcolor="#e5e5e5"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Full metrics table
    st.markdown("#### 📋 Все модели")
    st.dataframe(
        metrics_df.style.format(
            {
                "MAE": "{:.3f}",
                "RMSE": "{:.3f}",
                "Coverage": "{:.1f}%",
                "KPI_Violations": "{:.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # KPI Optimizer section
    st.markdown("---")
    st.markdown("### 🎯 KPI Optimizer")

    best_model = metrics_df.iloc[0]["Model"]
    bt_data["Month"] = bt_data["Date"].dt.month

    # Import calculate_kpi_corrections from forecast tab
    from pages import calculate_kpi_corrections

    monthly_shifts, bias = calculate_kpi_corrections(bt_data, best_model)

    # Apply corrections
    bt_data["Seasonal"] = bt_data[best_model] + bt_data["Month"].map(monthly_shifts)
    bt_data["Bias"] = bt_data[best_model] - bt_data["Month"].map(bias)

    # Calculate metrics
    errors_orig = (bt_data["Actual"] - bt_data[best_model]).abs()
    errors_seasonal = (bt_data["Actual"] - bt_data["Seasonal"]).abs()
    errors_bias = (bt_data["Actual"] - bt_data["Bias"]).abs()

    kpi_orig = (errors_orig <= 0.5).sum()
    kpi_seasonal = (errors_seasonal <= 0.5).sum()
    kpi_bias = (errors_bias <= 0.5).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Оригинал", f"KPI: {kpi_orig}/{len(bt_data)}", f"MAE: {errors_orig.mean():.3f}"
    )
    col2.metric(
        "🎯 Seasonal",
        f"KPI: {kpi_seasonal}/{len(bt_data)}",
        f"MAE: {errors_seasonal.mean():.3f}",
    )
    col3.metric(
        "📊 Bias", f"KPI: {kpi_bias}/{len(bt_data)}", f"MAE: {errors_bias.mean():.3f}"
    )
