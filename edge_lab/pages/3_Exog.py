"""
Exogenous Forecast Tab - Page 3
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

# Import required modules
try:
    from sirena.models.exog_loader import (
        load_exog_data,
        get_exog_source,
        copy_forecast_to_manual,
        clear_manual_file,
        save_manual_data,
        manual_file_exists,
    )
except ImportError:
    pass


def render_exog_forecast_tab():
    """Render exogenous variables forecast tab."""
    st.subheader("📊 Прогноз экзогенных переменных")

    exog_csv_path = "data/exog_forecast.csv"
    exog_json_path = "data/exog_forecast.json"
    manual_path = "data/exog_manual.csv"

    try:
        import json
        import traceback

        # Check source
        from sirena.models.exog_loader import get_exog_source

        source = get_exog_source()

        # === SOURCE SELECTOR ===
        st.markdown("### Источник данных")

        col_src1, col_src2, col_src3 = st.columns([2, 1, 1])
        with col_src1:
            if source == "manual":
                st.info("📝 **Используются ручные данные** (exog_manual.csv)")
            elif source == "forecast":
                st.success("🤖 **Используется автопрогноз** (exog_forecast.csv)")
            else:
                st.warning("⚠️ Нет данных. Сгенерируйте прогноз.")

        with col_src2:
            if source == "forecast":
                if st.button(
                    "📋 Скопировать в ручной",
                    help="Скопировать автопрогноз для редактирования",
                ):
                    from sirena.models.exog_loader import copy_forecast_to_manual

                    if copy_forecast_to_manual(overwrite=False):
                        st.success("Скопировано!")
                        st.rerun()
                    else:
                        st.error("Ошибка копирования")

        with col_src3:
            if source == "manual":
                if st.button("🗑 Очистить"):
                    from sirena.models.exog_loader import clear_manual_file

                    if clear_manual_file():
                        st.success("Очищено!")
                        st.rerun()

        # === MANUAL DATA ENTRY ===
        if source == "manual" and manual_file_exists():
            st.markdown("---")
            st.markdown("### ✏️ Редактирование данных")

            df_manual = load_exog_data(manual_path)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**Ключевая ставка (Ki, %)**")
                ki_values = {}
                for i in range(12):
                    date_str = (
                        df_manual["Date"].iloc[0] + pd.DateOffset(months=i)
                    ).strftime("%Y-%m")
                    ki_values[date_str] = st.number_input(
                        f"{date_str}",
                        value=df_manual[df_manual["Date"] == date_str]["Ki"].values[0],
                        step=0.25,
                        key=f"ki_{date_str}",
                    )

            with col2:
                st.markdown("**Курс USD/RUB**")
                usd_values = {}
                for i in range(12):
                    date_str = (
                        df_manual["Date"].iloc[0] + pd.DateOffset(months=i)
                    ).strftime("%Y-%m")
                    usd_values[date_str] = st.number_input(
                        f"{date_str}",
                        value=df_manual[df_manual["Date"] == date_str][
                            "USD_nom_i"
                        ].values[0],
                        step=1.0,
                        key=f"usd_{date_str}",
                    )

            if st.button("💾 Сохранить"):
                # Update dataframe
                for i, row in df_manual.iterrows():
                    date_str = row["Date"]
                    df_manual.at[i, "Ki"] = ki_values[date_str]
                    df_manual.at[i, "USD_nom_i"] = usd_values[date_str]

                save_manual_data(df_manual)
                st.success("Данные сохранены!")
                st.rerun()

        # === AUTO FORECAST ===
        if source != "forecast":
            st.markdown("---")
            st.markdown("### 🤖 Генерация прогноза")

            if st.button("🚀 Сгенерировать прогноз"):
                try:
                    from sirena.models.exog_loader import generate_forecast

                    generate_forecast()
                    st.success("Прогноз сгенерирован!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка: {e}")

    except Exception as e:
        import traceback

        st.error(f"Ошибка загрузки прогноза: {e}")
        st.code(traceback.format_exc())


# For standalone testing
if __name__ == "__main__":
    render_exog_forecast_tab()
