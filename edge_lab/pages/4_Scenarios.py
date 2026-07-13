"""
Scenarios Tab - Page 4
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import dashboard_utils


def render_scenarios_tab(df, last_date):
    """Render rate scenarios tab with calibrated transmission model v2.0."""
    st.subheader("🎚️ Сценарии ключевой ставки")

    # Model selection
    model_type = st.radio(
        "Модель трансмиссии",
        options=["simple", "subcomponent", "asymmetric"],
        format_func=lambda x: {
            "simple": "📊 Базовая (симметричный IRF)",
            "subcomponent": "🧩 Субкомпонентная (с декомпозицией)",
            "asymmetric": "📈📉 Асимметричная (hike ≠ cut)",
        }[x],
        horizontal=True,
    )

    # Model description
    if model_type == "simple":
        st.markdown("""
        **Базовая модель трансмиссии** — калиброванные параметры из литературы.
        - Пиковый эффект: **-0.08%** на 1 п.п. ставки
        - Лаг пикового эффекта: **6 месяцев**
        """)
    elif model_type == "subcomponent":
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
            options=["hawk", "neutral", "dove", "custom"],
            format_func=lambda x: {
                "hawk": "🦅 Hawk (жесткий)",
                "neutral": "⚖️ Neutral (базовый)",
                "dove": "🕊️ Dove (мягкий)",
                "custom": "✏️ Custom",
            }[x],
        )

        if scenario == "custom":
            st.markdown("**Custom Ki trajectory:**")
            start_ki = st.number_input(
                "Start Ki (%)", min_value=5.0, max_value=25.0, value=16.0, step=0.5
            )
            end_ki = st.number_input(
                "End Ki (%)", min_value=5.0, max_value=25.0, value=13.0, step=0.5
            )

    with col2:
        if scenario == "hawk":
            st.info(
                "🦅 **Hawk:** Ключевая ставка остается высокой (16-18% для борьбы с инфляцией)"
            )
        elif scenario == "neutral":
            st.success("⚖️ **Neutral:** Постепенное снижение (16→13% за 12 месяцев)")
        elif scenario == "dove":
            st.warning(
                "🕊️ **Dove:** Агрессивное снижение (16→10% для стимулирования экономики)"
            )
        else:
            st.info(f"✏️ **Custom:** {start_ki}% → {end_ki}%")

    # Generate scenario
    horizon = 12
    if st.button("🚀 Сгенерировать сценарий"):
        try:
            import sirena.exog.ki_trajectory as ki_traj

            if scenario == "custom":
                ki_forecast = ki_traj.generate_linear_trajectory(
                    start_ki, end_ki, horizon
                )
            else:
                ki_forecast = ki_traj.generate_scenario(scenario, horizon)

            # Calculate inflation impact
            if model_type == "simple":
                from sirena.models.exog_forecaster import SimpleTransmissionModel

                model = SimpleTransmissionModel()
            elif model_type == "subcomponent":
                from sirena.models.exog_forecaster import SubcomponentTransmissionModel

                model = SubcomponentTransmissionModel()
            else:
                from sirena.models.exog_forecaster import AsymmetricTransmissionModel

                model = AsymmetricTransmissionModel()

            impact = model.calculate_impact(ki_forecast, df, last_date)

            # Display results
            st.markdown("---")
            st.markdown("### 📊 Результаты сценария")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**Траектория Ki:**")
                fig_ki = go.Figure()
                fig_ki.add_trace(
                    go.Scatter(
                        x=pd.date_range(
                            start=last_date + pd.DateOffset(months=1),
                            periods=horizon,
                            freq="MS",
                        ),
                        y=ki_forecast,
                        name="Ki",
                        line=dict(color="red", width=3),
                    )
                )
                fig_ki.update_layout(title="Ключевая ставка", height=300)
                st.plotly_chart(fig_ki, use_container_width=True)

            with col2:
                st.markdown("**Влияние на инфляцию:**")
                fig_impact = go.Figure()
                fig_impact.add_trace(
                    go.Scatter(
                        x=pd.date_range(
                            start=last_date + pd.DateOffset(months=1),
                            periods=horizon,
                            freq="MS",
                        ),
                        y=impact,
                        name="Impact",
                        line=dict(color="blue", width=3),
                    )
                )
                fig_impact.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_impact.update_layout(title="Влияние на инфляцию (п.п.)", height=300)
                st.plotly_chart(fig_impact, use_container_width=True)

        except Exception as e:
            st.error(f"Ошибка: {e}")
            import traceback

            st.code(traceback.format_exc())


# For standalone testing
if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sirena.data.kbr_loader import load_inflation_data

    df = load_inflation_data()
    last_date = df["Date"].max()
    render_scenarios_tab(df, last_date)
