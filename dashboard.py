"""
СИРЕНА-КБР v5.3: Dashboard
13 вкладок: 5 прогнозов + Сценарии Ki + Экзогенные + Weekly + 5 бэктестов
Новое в v5.3: Weekly Prices tab - nowcasting, volatility alerts, price trends
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

# Import dashboard utilities
import dashboard_utils

# Import forecast, backtest, weekly, research, compare tab functions, and constants from pages module
from pages import (
    get_best_model_for_horizon,
    forecast_with_model,
    calculate_kpi_corrections,
    render_forecast_tab,
    render_forecast_h12_tab,
    load_backtest_data,
    render_backtest_tab,
    render_alert_panel,
    render_weekly_tab,
    render_nowcast_tab,
    render_seasonality_tab,
    render_macro_tab,
    render_regime_indicator,
    render_compare_tab,
    ALL_MODELS,
    MODEL_COLORS,
    MONTH_NAMES_RU,
)

# --- SETUP ---
st.set_page_config(
    page_title="СИРЕНА-КБР v5.3", layout="wide", initial_sidebar_state="collapsed"
)
warnings.filterwarnings("ignore")


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_data():
    """Load main inflation data."""
    try:
        df_raw = pd.read_csv("data/inflation_data.csv", sep=";", decimal=",")
        df_raw["Date"] = pd.to_datetime(
            df_raw["Date"], format="%d.%m.%Y", errors="coerce"
        ).dt.to_period("M").dt.to_timestamp()

        source_columns = {
            "mom": "Все товары и услуги",
            "Prod": "Продовольственные товары",
            "Nonprod": "Непродовольственные товары",
            "Serv": "Услуги",
        }
        missing = [column for column in source_columns if column not in df_raw]
        if missing:
            raise KeyError(
                "Missing canonical inflation columns: " + ", ".join(missing)
            )

        df = df_raw.set_index("Date")[list(source_columns)].rename(
            columns=source_columns
        )
        for column in df.columns:
            df[column] = pd.to_numeric(
                df[column].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
        return df.dropna(how="all").sort_index()
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return None


# =============================================================================
# FORECAST FUNCTIONS
# =============================================================================
def _render_exog_generation_form():
    """Helper to render exog forecast generation form."""
    st.markdown("### Генерация прогноза")
    st.markdown("Введите текущие значения для генерации автопрогноза:")
    col_ki, col_usd, col_brent = st.columns(3)
    with col_ki:
        gen_ki = st.number_input(
            "Ключевая ставка (%)",
            min_value=4.0,
            max_value=25.0,
            value=16.0,
            step=0.25,
            key="gen_ki",
        )
    with col_usd:
        gen_usd = st.number_input(
            "Курс USD (руб.)",
            min_value=50.0,
            max_value=150.0,
            value=78.0,
            step=1.0,
            key="gen_usd",
        )
    with col_brent:
        gen_brent = st.number_input(
            "Brent ($)",
            min_value=30.0,
            max_value=150.0,
            value=60.0,
            step=1.0,
            key="gen_brent",
        )
    if st.button("🔄 Сгенерировать прогноз", type="primary", key="gen_exog_btn"):
        with st.spinner("Генерация..."):
            try:
                infl_df = pd.read_csv("data/inflation_data.csv", sep=";", decimal=",")
                for col in infl_df.columns:
                    if col != "Date" and infl_df[col].dtype == object:
                        infl_df[col] = infl_df[col].astype(str).str.replace(",", ".")
                        infl_df[col] = pd.to_numeric(infl_df[col], errors="coerce")
                infl_df["Date"] = pd.to_datetime(infl_df["Date"], format="%d.%m.%Y")
                infl_df = infl_df.set_index("Date").sort_index()
                from sirena.models.exog_forecaster import ExogForecaster

                ef = ExogForecaster(
                    ki_method="adaptive",
                    usd_method="adaptive",
                    brent_method="ar1",
                    current_ki=gen_ki,
                    current_usd_abs=gen_usd,
                    current_ruonia=gen_ki - 0.25,
                    current_brent=gen_brent,
                )
                ef.fit(infl_df)
                ef.save_forecast(
                    "data/exog_forecast.csv", horizon=12, include_history=6
                )
                ef.save_forecast_json("data/exog_forecast.json", horizon=12)
                st.success("Прогноз сгенерирован!")
                st.rerun()
            except Exception as e:
                st.error(f"Ошибка: {e}")


def render_exog_forecast_tab():
    """Render exogenous variables forecast tab."""
    st.subheader("📊 Прогноз экзогенных переменных")
    # Import loader
    from sirena.models.exog_loader import (
        load_exog_data,
        get_exog_source,
        copy_forecast_to_manual,
        clear_manual_file,
        save_manual_data,
    )

    exog_csv_path = "data/exog_forecast.csv"
    exog_json_path = "data/exog_forecast.json"
    manual_path = "data/exog_manual.csv"
    try:
        import json
        import traceback

        # Check source
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
                    if copy_forecast_to_manual(overwrite=False):
                        st.success("Скопировано!")
                        st.rerun()
                    else:
                        st.warning("Ручной файл уже существует")
        with col_src3:
            if source == "manual":
                if st.button("🗑️ Удалить ручной", help="Вернуться к автопрогнозу"):
                    if clear_manual_file():
                        st.success("Удалено!")
                        st.rerun()
        # Load data based on source
        if source == "manual":
            exog_df = pd.read_csv(manual_path)
            exog_df["Date"] = pd.to_datetime(exog_df["Date"])
        elif os.path.exists(exog_csv_path):
            exog_df = pd.read_csv(exog_csv_path)
            exog_df["Date"] = pd.to_datetime(exog_df["Date"])
        else:
            exog_df = None
        # Load metadata from forecast
        exog_meta = None
        if os.path.exists(exog_json_path):
            with open(exog_json_path, "r", encoding="utf-8") as f:
                exog_meta = json.load(f)
        if exog_df is None:
            st.warning("Нет данных. Сгенерируйте прогноз ниже.")
            # Show generation form
            _render_exog_generation_form()
            return
        # Display metadata
        if exog_meta:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Последние данные", exog_meta["last_data_date"])
            with col2:
                st.metric("Горизонт прогноза", f"{exog_meta['horizon']} мес.")
            with col3:
                methods_str = ", ".join(
                    [f"{k}: {v}" for k, v in exog_meta["methods"].items()]
                )
                st.metric("Методы", methods_str)
        # Split into history and forecast
        hist_df = exog_df[exog_df["Type"] == "History"].copy()
        fc_df = exog_df[exog_df["Type"] != "History"].copy()
        # --- Table ---
        st.markdown("### Таблица данных")
        # If manual mode, show editable table
        if source == "manual":
            st.markdown(
                "📝 **Режим редактирования** — измените значения и нажмите 'Сохранить'"
            )
            # Prepare editable dataframe
            edit_df = exog_df.copy()
            edit_df["Date"] = edit_df["Date"].dt.strftime("%Y-%m-%d")
            # Only keep editable columns
            edit_cols = ["Date", "Ki", "Ruonia", "USD_ABS", "Brent", "Type"]
            edit_df = edit_df[[c for c in edit_cols if c in edit_df.columns]]
            # Use data_editor for editing
            edited_df = st.data_editor(
                edit_df,
                use_container_width=True,
                height=400,
                num_rows="fixed",
                column_config={
                    "Date": st.column_config.TextColumn("Дата", disabled=True),
                    "Ki": st.column_config.NumberColumn(
                        "Ki (%)", min_value=4.0, max_value=25.0, step=0.25
                    ),
                    "Ruonia": st.column_config.NumberColumn(
                        "Ruonia (%)", min_value=4.0, max_value=25.0, step=0.25
                    ),
                    "USD_ABS": st.column_config.NumberColumn(
                        "USD (руб.)", min_value=50.0, max_value=150.0, step=0.5
                    ),
                    "Brent": st.column_config.NumberColumn(
                        "Brent ($)", min_value=30.0, max_value=150.0, step=0.5
                    ),
                    "Type": st.column_config.TextColumn("Тип", disabled=True),
                },
                key="exog_editor",
            )
            # Save button
            if st.button("💾 Сохранить изменения", type="primary"):
                edited_df["Date"] = pd.to_datetime(edited_df["Date"])
                if save_manual_data(edited_df):
                    st.success("Сохранено!")
                    st.rerun()
                else:
                    st.error("Ошибка сохранения")
        else:
            # Read-only display
            display_df = exog_df.copy()
            display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m")
            # Rename columns for Russian
            col_rename = {
                "Date": "Дата",
                "Ki": "Ключевая ставка (%)",
                "Ruonia": "RUONIA (%)",
                "USD_ABS": "Курс USD (руб.)",
                "usd_nom_i": "Индекс USD",
                "Brent": "Brent ($)",
                "Type": "Тип",
            }
            display_df = display_df.rename(columns=col_rename)

            # Style the dataframe
            def highlight_forecast(row):
                if "Тип" in row.index and row["Тип"] == "Forecast":
                    return ["background-color: #e6f3ff"] * len(row)
                return [""] * len(row)

            # Format numeric columns
            format_dict = {}
            if "Ключевая ставка (%)" in display_df.columns:
                format_dict["Ключевая ставка (%)"] = "{:.2f}"
            if "RUONIA (%)" in display_df.columns:
                format_dict["RUONIA (%)"] = "{:.2f}"
            if "Курс USD (руб.)" in display_df.columns:
                format_dict["Курс USD (руб.)"] = "{:.2f}"
            if "Индекс USD" in display_df.columns:
                format_dict["Индекс USD"] = "{:.2f}"
            if "Brent ($)" in display_df.columns:
                format_dict["Brent ($)"] = "{:.2f}"
            styled_df = display_df.style.apply(highlight_forecast, axis=1)
            if format_dict:
                styled_df = styled_df.format(format_dict, na_rep="-")
            st.dataframe(styled_df, use_container_width=True, height=400)
        # --- Chart ---
        st.markdown("### Траектории экзогенных")
        # Create figure with secondary y-axis
        fig = go.Figure()
        # Ki trajectory
        fig.add_trace(
            go.Scatter(
                x=hist_df["Date"],
                y=hist_df["Ki"],
                name="Ki (история)",
                mode="lines+markers",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=6),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc_df["Date"],
                y=fc_df["Ki"],
                name="Ki (прогноз)",
                mode="lines+markers",
                line=dict(color="#1f77b4", width=2, dash="dash"),
                marker=dict(size=6, symbol="circle-open"),
            )
        )
        # Ruonia trajectory
        fig.add_trace(
            go.Scatter(
                x=hist_df["Date"],
                y=hist_df["Ruonia"],
                name="Ruonia (история)",
                mode="lines+markers",
                line=dict(color="#ff7f0e", width=2),
                marker=dict(size=6),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc_df["Date"],
                y=fc_df["Ruonia"],
                name="Ruonia (прогноз)",
                mode="lines+markers",
                line=dict(color="#ff7f0e", width=2, dash="dash"),
                marker=dict(size=6, symbol="circle-open"),
            )
        )

        # Add vertical line at forecast start
        if len(fc_df) > 0:
            fc_start = fc_df["Date"].iloc[0]
            fig.add_vline(x=fc_start, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="Прогноз ключевой ставки и RUONIA",
            xaxis_title="Дата",
            yaxis_title="Ставка (%)",
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(gridcolor="#e5e5e5"),
            yaxis=dict(gridcolor="#e5e5e5"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        st.plotly_chart(fig, use_container_width=True)
        # USD and Brent charts side by side
        col_usd, col_brent = st.columns(2)
        with col_usd:
            if "USD_ABS" in exog_df.columns and not exog_df["USD_ABS"].isna().all():
                fig_usd = go.Figure()
                fig_usd.add_trace(
                    go.Scatter(
                        x=hist_df["Date"],
                        y=hist_df["USD_ABS"],
                        name="USD (история)",
                        mode="lines+markers",
                        line=dict(color="#9467bd", width=2),
                        marker=dict(size=6),
                    )
                )
                fig_usd.add_trace(
                    go.Scatter(
                        x=fc_df["Date"],
                        y=fc_df["USD_ABS"],
                        name="USD (прогноз)",
                        mode="lines+markers",
                        line=dict(color="#9467bd", width=2, dash="dash"),
                        marker=dict(size=6, symbol="circle-open"),
                    )
                )
                if len(fc_df) > 0:
                    fig_usd.add_vline(
                        x=fc_df["Date"].iloc[0], line_dash="dot", line_color="gray"
                    )
                fig_usd.update_layout(
                    title="Прогноз курса USD/RUB",
                    xaxis_title="Дата",
                    yaxis_title="Курс (руб.)",
                    height=350,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    xaxis=dict(gridcolor="#e5e5e5"),
                    yaxis=dict(gridcolor="#e5e5e5"),
                )
                st.plotly_chart(fig_usd, use_container_width=True)
        with col_brent:
            if "Brent" in exog_df.columns and not exog_df["Brent"].isna().all():
                fig_brent = go.Figure()
                fig_brent.add_trace(
                    go.Scatter(
                        x=hist_df["Date"],
                        y=hist_df["Brent"],
                        name="Brent (история)",
                        mode="lines+markers",
                        line=dict(color="#2ca02c", width=2),
                        marker=dict(size=6),
                    )
                )
                fig_brent.add_trace(
                    go.Scatter(
                        x=fc_df["Date"],
                        y=fc_df["Brent"],
                        name="Brent (прогноз)",
                        mode="lines+markers",
                        line=dict(color="#2ca02c", width=2, dash="dash"),
                        marker=dict(size=6, symbol="circle-open"),
                    )
                )
                if len(fc_df) > 0:
                    fig_brent.add_vline(
                        x=fc_df["Date"].iloc[0], line_dash="dot", line_color="gray"
                    )
                fig_brent.update_layout(
                    title="Прогноз цены нефти Brent",
                    xaxis_title="Дата",
                    yaxis_title="Цена ($)",
                    height=350,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    xaxis=dict(gridcolor="#e5e5e5"),
                    yaxis=dict(gridcolor="#e5e5e5"),
                )

                st.plotly_chart(fig_brent, use_container_width=True)
        # --- Methodology ---
        with st.expander("📖 Методология прогноза"):
            st.markdown("""
            **Методы прогноза экзогенных переменных:**
            | Переменная | Метод | Описание |
            |------------|-------|----------|
            | **Ki (Ключевая ставка)** | Adaptive | Продолжение тренда с затуханием (0.7^t). Если Ki менялся в последние 6 мес — тренд продолжается |
            | **Ruonia** | Ki - спред | Ruonia = Ki - 0.25 п.п. (исторический спред) |
            | **USD** | Adaptive | Продолжение тренда с затуханием (0.8^t). Анализ изменений за 6 мес. |
            | **Brent** | AR(1) | Возврат к долгосрочному среднему $70 со скоростью 3%/мес |
            **Валидация методов (бэктест 2023-2025):**
            - Adaptive для Ki: **MAE -24%** vs naive на h=1
            - Adaptive для USD: учитывает тренды укрепления/ослабления рубля
            - AR(1) для Brent: адекватно отражает mean reversion
            **Интеграция:**
            Прогнозные траектории используются в SubcompMulti v2.7 для rate-sensitive features.
            """)
        # --- Update section ---
        st.markdown("### Обновить прогноз")
        st.markdown("Введите актуальные значения для пересчёта прогноза:")
        col_ki, col_usd, col_brent = st.columns(3)
        with col_ki:
            new_ki = st.number_input(
                "Ключевая ставка (%)",
                min_value=4.0,
                max_value=25.0,
                value=float(exog_meta["last_values"].get("Ki", 16.0)),
                step=0.25,
                key="exog_ki",
            )
        with col_usd:
            new_usd = st.number_input(
                "Курс USD (руб.)",
                min_value=50.0,
                max_value=150.0,
                value=float(exog_meta["last_values"].get("USD_ABS", 78.0)),
                step=1.0,
                key="exog_usd",
            )
        with col_brent:
            new_brent = st.number_input(
                "Brent ($)",
                min_value=30.0,
                max_value=150.0,
                value=float(exog_meta["last_values"].get("Brent", 60.0)),
                step=1.0,
                key="exog_brent",
            )
        if st.button("🔄 Пересчитать прогноз", type="primary"):
            with st.spinner("Генерация нового прогноза..."):
                try:
                    # Load macro data
                    infl_df = pd.read_csv(
                        "data/inflation_data.csv", sep=";", decimal=","
                    )
                    for col in infl_df.columns:
                        if col != "Date" and infl_df[col].dtype == object:
                            infl_df[col] = (
                                infl_df[col].astype(str).str.replace(",", ".")
                            )
                            infl_df[col] = pd.to_numeric(infl_df[col], errors="coerce")
                    infl_df["Date"] = pd.to_datetime(infl_df["Date"], format="%d.%m.%Y")
                    infl_df = infl_df.set_index("Date").sort_index()

                    from sirena.models.exog_forecaster import ExogForecaster

                    ef = ExogForecaster(
                        ki_method="adaptive",
                        usd_method="adaptive",
                        brent_method="ar1",
                        current_ki=new_ki,
                        current_usd_abs=new_usd,
                        current_ruonia=new_ki - 0.25,
                        current_brent=new_brent,
                    )
                    ef.fit(infl_df)
                    ef.save_forecast(
                        "data/exog_forecast.csv", horizon=12, include_history=6
                    )
                    ef.save_forecast_json("data/exog_forecast.json", horizon=12)

                    st.success("Прогноз обновлён!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка обновления: {e}")
    except FileNotFoundError:
        st.warning(
            "Файл прогноза не найден. Введите текущие значения и нажмите кнопку для генерации."
        )
        col_ki, col_usd, col_brent = st.columns(3)
        with col_ki:
            init_ki = st.number_input(
                "Ключевая ставка (%)",
                min_value=4.0,
                max_value=25.0,
                value=16.0,
                step=0.25,
                key="init_ki",
            )
        with col_usd:
            init_usd = st.number_input(
                "Курс USD (руб.)",
                min_value=50.0,
                max_value=150.0,
                value=78.0,
                step=1.0,
                key="init_usd",
            )
        with col_brent:
            init_brent = st.number_input(
                "Brent ($)",
                min_value=30.0,
                max_value=150.0,
                value=60.0,
                step=1.0,
                key="init_brent",
            )
        if st.button("🔄 Сгенерировать прогноз", type="primary"):
            with st.spinner("Генерация прогноза экзогенных..."):
                try:
                    infl_df = pd.read_csv(
                        "data/inflation_data.csv", sep=";", decimal=","
                    )
                    for col in infl_df.columns:
                        if col != "Date" and infl_df[col].dtype == object:
                            infl_df[col] = (
                                infl_df[col].astype(str).str.replace(",", ".")
                            )
                            infl_df[col] = pd.to_numeric(infl_df[col], errors="coerce")
                    infl_df["Date"] = pd.to_datetime(infl_df["Date"], format="%d.%m.%Y")
                    infl_df = infl_df.set_index("Date").sort_index()

                    from sirena.models.exog_forecaster import ExogForecaster

                    ef = ExogForecaster(
                        ki_method="adaptive",
                        usd_method="adaptive",
                        brent_method="ar1",
                        current_ki=init_ki,
                        current_usd_abs=init_usd,
                        current_ruonia=init_ki - 0.25,
                        current_brent=init_brent,
                    )
                    ef.fit(infl_df)
                    ef.save_forecast(
                        "data/exog_forecast.csv", horizon=12, include_history=6
                    )
                    ef.save_forecast_json("data/exog_forecast.json", horizon=12)
                    st.success("Прогноз сгенерирован!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Ошибка генерации: {e}")
    except Exception as e:
        import traceback

        st.error(f"Ошибка загрузки прогноза: {e}")
        st.code(traceback.format_exc())


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
            options=["auto", "base", "hike", "cut", "custom"],
            format_func=lambda x: {
                "auto": "🤖 Авто (по правилу Тейлора)",
                "base": "📊 Базовый (без изменений)",
                "hike": "📈 Повышение (+2 п.п. за 6 мес)",
                "cut": "📉 Снижение (-2 п.п. за 6 мес)",
                "custom": "⚙️ Пользовательский",
            }[x],
        )
    with col2:
        horizon = st.slider(
            "Горизонт прогноза", min_value=6, max_value=24, value=12, step=3
        )
    # Custom scenario input
    custom_ki = None
    if scenario == "custom":
        st.markdown("**Введите траекторию изменения Ki (п.п. от текущего уровня):**")
        cols = st.columns(min(6, horizon))
        custom_vals = []
        for i, col in enumerate(cols):
            if i < horizon:
                val = col.number_input(
                    f"Месяц {i + 1}", value=0.0, step=0.5, key=f"ki_{i}"
                )
                custom_vals.append(val)
        # Fill remaining months with last value
        while len(custom_vals) < horizon:
            custom_vals.append(custom_vals[-1] if custom_vals else 0.0)
        custom_ki = np.array(custom_vals)
    # Run model
    with st.spinner("Расчёт модели трансмиссии..."):
        try:
            # Load inflation_data.csv
            infl_df = pd.read_csv(
                "data/inflation_data.csv", sep=";", decimal=",", encoding="utf-8-sig"
            )
            for col in infl_df.columns:
                if col != "Date" and infl_df[col].dtype == object:
                    infl_df[col] = infl_df[col].astype(str).str.replace(",", ".")
                    infl_df[col] = pd.to_numeric(infl_df[col], errors="coerce")
            infl_df["Date"] = pd.to_datetime(
                infl_df["Date"], format="%d.%m.%Y", errors="coerce"
            )
            infl_df = infl_df.set_index("Date").sort_index()

            # Display current regime
            try:
                from sirena.models.regime_detector import detect_regime, MacroRegime

                regime, regime_diag = detect_regime(infl_df)
                regime_emoji = {"normal": "🟢", "shock": "🔴", "high_inflation": "🟠"}[
                    regime.value
                ]
                regime_label = {
                    "normal": "Нормальный",
                    "shock": "Шок",
                    "high_inflation": "Высокая инфляция",
                }[regime.value]
                st.info(
                    f"**Текущий режим:** {regime_emoji} {regime_label} | ΔKi(3м): {regime_diag['ki_change']:+.1f} п.п. | ΔRuonia(3м): {regime_diag['ruonia_change']:+.1f} п.п."
                )
            except Exception:
                pass
            # Auto scenario using Taylor rule
            auto_ki_trajectory = None
            if scenario == "auto":
                try:
                    from sirena.models.ki_trajectory import KiTrajectoryForecaster
                    from sirena.models.subcomponent_multi import (
                        SubcomponentMultiForecaster,
                    )

                    # Get baseline inflation forecast
                    base_model = SubcomponentMultiForecaster(horizon=1)
                    base_model.fit(infl_df, "mom")
                    baseline_fc = base_model.forecast(horizon)
                    # Get Ki trajectory from Taylor rule
                    ki_model = KiTrajectoryForecaster()
                    ki_model.fit(infl_df)
                    auto_ki_trajectory = ki_model.forecast_trajectory(
                        horizon, baseline_fc
                    )
                    current_ki = (
                        infl_df["Ki"].iloc[-1] if "Ki" in infl_df.columns else 21.0
                    )
                    ki_change_auto = auto_ki_trajectory[-1] - current_ki
                    st.success(
                        f"**Авто-траектория Ki:** {current_ki:.1f}% → {auto_ki_trajectory[-1]:.1f}% (Δ = {ki_change_auto:+.1f} п.п.)"
                    )
                except Exception as e:
                    st.warning(f"Авто-сценарий недоступен: {e}")
                    scenario = "base"  # Fallback

            # Initialize model based on type
            group_decomposition = None
            if model_type == "subcomponent":
                from sirena.models.subcomponent_scenario import (
                    SubcomponentScenarioForecaster,
                )

                model = SubcomponentScenarioForecaster()
                model.fit(infl_df)
                model_desc = "Субкомпонентная модель с декомпозицией по группам товаров"
            elif model_type == "asymmetric":
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
                "base": 0.0,
                "hike": 2.0,  # +2 п.п.
                "cut": -2.0,  # -2 п.п.
            }
            # Add auto scenario if trajectory is available
            if auto_ki_trajectory is not None:
                current_ki = infl_df["Ki"].iloc[-1] if "Ki" in infl_df.columns else 21.0
                ki_changes["auto"] = auto_ki_trajectory[-1] - current_ki
            # Get forecasts for all scenarios
            results = {}
            for sc, ki_change in ki_changes.items():
                # Use forecast_scenario_with_ci for subcomponent model
                if model_type == "subcomponent" and hasattr(
                    model, "forecast_scenario_with_ci"
                ):
                    fc = model.forecast_scenario_with_ci(
                        horizon, ki_change=ki_change, alpha=0.1
                    )
                    results[sc] = {
                        "total": fc["mean"],
                        "ki_path": np.full(horizon, ki_change),
                        "effect": fc["effect"],
                        "baseline": fc["baseline"],
                        "total_q05": fc["ci_lower"],  # Real 90% CI
                        "total_q95": fc["ci_upper"],
                    }
                else:
                    fc = model.forecast_scenario(horizon, ki_change=ki_change)
                    results[sc] = {
                        "total": fc["total"],
                        "ki_path": np.full(horizon, ki_change),
                        "effect": fc["effect"],
                        "baseline": fc["baseline"],
                        "total_q05": fc["total"] - 0.3,  # Approximate CI
                        "total_q95": fc["total"] + 0.3,
                    }

                # Get group decomposition for subcomponent model
                if model_type == "subcomponent" and sc == "hike":
                    fc_full = model.forecast_scenario(horizon, ki_change=ki_change)
                    group_decomposition = fc_full.get("group_decomposition")
            # Custom scenario
            if custom_ki is not None:
                if model_type == "subcomponent" and hasattr(
                    model, "forecast_scenario_with_ci"
                ):
                    fc_custom = model.forecast_scenario_with_ci(
                        horizon, ki_change=custom_ki[0], alpha=0.1
                    )
                    results["custom"] = {
                        "total": fc_custom["mean"],
                        "ki_path": custom_ki,
                        "effect": fc_custom["effect"],
                        "baseline": fc_custom["baseline"],
                        "total_q05": fc_custom["ci_lower"],
                        "total_q95": fc_custom["ci_upper"],
                    }
                else:
                    fc_custom = model.forecast_scenario(horizon, ki_change=custom_ki[0])
                    results["custom"] = {
                        "total": fc_custom["total"],
                        "ki_path": custom_ki,
                        "effect": fc_custom["effect"],
                        "baseline": fc_custom["baseline"],
                        "total_q05": fc_custom["total"] - 0.3,
                        "total_q95": fc_custom["total"] + 0.3,
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
    dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS"
    )
    scenario_colors = {
        "auto": "#f59e0b",  # Amber
        "base": "#3b82f6",  # Blue
        "hike": "#ef4444",  # Red
        "cut": "#22c55e",  # Green
        "custom": "#8b5cf6",  # Purple
    }
    scenario_names = {
        "auto": "Авто (Тейлор)",
        "base": "Базовый",
        "hike": "Повышение Ki",
        "cut": "Снижение Ki",
        "custom": "Пользовательский",
    }

    for sc, fc in results.items():
        color = scenario_colors[sc]
        is_selected = sc == scenario
        # Add CI fan chart for selected scenario
        if is_selected and "total_q05" in fc and "total_q95" in fc:
            # Convert hex color to rgba for fill
            hex_color = color.lstrip("#")
            r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            fill_color = f"rgba({r}, {g}, {b}, 0.2)"
            # Upper bound (no line)
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=fc["total_q95"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Lower bound with fill to upper
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=fc["total_q05"],
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=fill_color,
                    showlegend=True,
                    name=f"{scenario_names[sc]} 90% CI",
                )
            )
        # Main forecast line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=fc["total"],
                name=scenario_names[sc],
                mode="lines+markers",
                line=dict(
                    color=color,
                    width=2 if is_selected else 1,
                    dash="solid" if is_selected else "dot",
                ),
                marker=dict(size=6 if is_selected else 4),
            )
        )
    fig.update_layout(
        title="Прогноз MoM инфляции при разных сценариях ставки"
        + (" (с 90% CI)" if model_type == "subcomponent" else ""),
        xaxis_title="Дата",
        yaxis_title="MoM инфляция (%)",
        height=450,
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#e5e5e5"),
        yaxis=dict(gridcolor="#e5e5e5"),
    )

    st.plotly_chart(fig, use_container_width=True)
    # Ki trajectory chart
    fig_ki = go.Figure()
    for sc, fc in results.items():
        fig_ki.add_trace(
            go.Scatter(
                x=dates,
                y=fc["ki_path"],
                name=scenario_names[sc],
                mode="lines+markers",
                line=dict(color=scenario_colors[sc], width=2 if sc == scenario else 1),
            )
        )
    fig_ki.update_layout(
        title="Траектория изменения ключевой ставки (п.п.)",
        xaxis_title="Дата",
        yaxis_title="Изменение Ki (п.п.)",
        height=300,
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#e5e5e5"),
        yaxis=dict(gridcolor="#e5e5e5"),
    )
    st.plotly_chart(fig_ki, use_container_width=True)
    # Summary metrics
    st.markdown("### 📋 Сравнение сценариев")
    # Calculate baseline cumulative for delta
    base_cum = np.sum(results["base"]["total"])
    cols = st.columns(len(results))
    for i, (sc, fc) in enumerate(results.items()):
        with cols[i]:
            cum_cpi = np.sum(fc["total"])
            effect = np.sum(fc["effect"])
            delta_vs_base = cum_cpi - base_cum
            st.metric(
                label=scenario_names[sc],
                value=f"{cum_cpi:.2f}%",
                delta=f"Эффект: {effect:+.2f}%" if sc != "base" else "базовый",
            )
    # Group decomposition for subcomponent model
    if model_type == "subcomponent" and group_decomposition:
        st.markdown("---")
        st.markdown("### 🧩 Декомпозиция эффекта по группам товаров")
        st.markdown("*При повышении ставки на +2 п.п.*")
        group_names = {
            "credit": "🏠 Кредитозависимые (авто, мебель)",
            "import": "🛒 Импортозависимые (одежда, обувь)",
            "basic": "🥬 Базовые продукты (мясо, молоко)",
            "regulated": "🏛️ Регулируемые (ЖКХ)",
            "services": "🎓 Услуги (образование, туризм)",
            "other": "📦 Прочие",
        }
        # Sort by absolute effect
        sorted_groups = sorted(
            group_decomposition.items(), key=lambda x: abs(x[1]), reverse=True
        )
        # Create bar chart
        fig_decomp = go.Figure()
        for group, effect in sorted_groups:
            if effect != 0:
                fig_decomp.add_trace(
                    go.Bar(
                        x=[group_names.get(group, group)],
                        y=[effect],
                        name=group,
                        marker_color="#ef4444" if effect < 0 else "#22c55e",
                        text=[f"{effect:+.3f}%"],
                        textposition="outside",
                    )
                )
        fig_decomp.update_layout(
            title="Вклад каждой группы товаров в общий эффект",
            yaxis_title="Эффект (%)",
            height=350,
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(gridcolor="#e5e5e5"),
        )
        st.plotly_chart(fig_decomp, use_container_width=True)
        # Summary table
        decomp_data = []
        total_effect = sum(group_decomposition.values())
        for group, effect in sorted_groups:
            if effect != 0:
                pct = (effect / total_effect * 100) if total_effect != 0 else 0
                decomp_data.append(
                    {
                        "Группа": group_names.get(group, group),
                        "Эффект (%)": f"{effect:+.4f}",
                        "Доля (%)": f"{pct:.1f}%",
                    }
                )
        decomp_df = pd.DataFrame(decomp_data)
        st.dataframe(decomp_df, use_container_width=True, hide_index=True)

    # Asymmetric model comparison
    if model_type == "asymmetric":
        st.markdown("---")
        st.markdown("### 📈📉 Асимметрия эффекта")

        hike_effect = np.sum(results["hike"]["effect"])
        cut_effect = np.sum(results["cut"]["effect"])
        ratio = abs(hike_effect / cut_effect) if cut_effect != 0 else float("inf")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Повышение +2 п.п.", f"{hike_effect:+.2f}%")
        with col2:
            st.metric("Снижение -2 п.п.", f"{cut_effect:+.2f}%")
        with col3:
            st.metric("Ratio (hike/cut)", f"{ratio:.2f}x")
        st.info(
            "📊 Повышение ставки влияет сильнее, чем снижение. "
            "Причина: банки быстрее повышают ставки по кредитам, чем снижают."
        )
    # YoY calculation
    st.markdown("---")
    st.markdown("### 📊 Годовая инфляция (YoY) по сценариям")
    try:
        from sirena.utils.yoy import yoy_at_horizon

        mom_history = (df["Все товары и услуги"] - 100).dropna()
        yoy_comparison = []
        for sc, fc in results.items():
            if horizon >= 12:
                yoy_h12 = yoy_at_horizon(mom_history, fc["total"], 12)
            else:
                yoy_h12 = yoy_at_horizon(mom_history, fc["total"], horizon)
            yoy_comparison.append(
                {
                    "Сценарий": scenario_names[sc],
                    "Ki (п.п.)": fc["ki_path"][-1],
                    "Cum MoM (%)": np.sum(fc["total"]),
                    "YoY h=12 (%)": yoy_h12 if horizon >= 12 else np.nan,
                }
            )
        yoy_df = pd.DataFrame(yoy_comparison)
        st.dataframe(
            yoy_df.style.format(
                {
                    "Ki (п.п.)": "{:+.1f}",
                    "Cum MoM (%)": "{:.2f}",
                    "YoY h=12 (%)": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    except Exception as e:
        st.warning(f"YoY расчёт недоступен: {e}")
    # Effect breakdown chart
    st.markdown("---")
    st.markdown("### 📉 Разбивка эффекта от ставки")
    if scenario != "base":
        selected_result = results[scenario]
        fig_effect = go.Figure()
        fig_effect.add_trace(
            go.Bar(
                x=dates,
                y=selected_result["baseline"],
                name="Базовый прогноз",
                marker_color="#3b82f6",
            )
        )
        fig_effect.add_trace(
            go.Bar(
                x=dates,
                y=selected_result["effect"],
                name="Эффект Ki",
                marker_color="#ef4444" if scenario == "hike" else "#22c55e",
            )
        )
        fig_effect.update_layout(
            title=f"Разбивка: {scenario_names[scenario]}",
            xaxis_title="Дата",
            yaxis_title="MoM инфляция (%)",
            barmode="relative",
            height=400,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )

        st.plotly_chart(fig_effect, use_container_width=True)
        st.info(f"""
        **Интерпретация для сценария "{scenario_names[scenario]}":**
        - Базовый прогноз (без изменения ставки): **{np.sum(selected_result["baseline"]):.2f}%** за {horizon} мес
        - Эффект от изменения ставки: **{np.sum(selected_result["effect"]):+.2f}%**
        - Итоговый прогноз: **{np.sum(selected_result["total"]):.2f}%**
        """)
    # Detailed table
    with st.expander("📋 Подробная таблица прогнозов"):
        detail_df = pd.DataFrame({"Дата": [d.strftime("%m.%y") for d in dates]})
        for sc, fc in results.items():
            detail_df[f"{scenario_names[sc]}"] = fc["total"]
            if sc != "base":
                detail_df[f"{scenario_names[sc]} эффект"] = fc["effect"]
        st.dataframe(
            detail_df.style.format(
                {col: "{:.3f}" for col in detail_df.columns if col != "Дата"}
            ),
            use_container_width=True,
            hide_index=True,
        )


# REGIME INDICATOR WIDGET
# =============================================================================
def render_regime_indicator(df):
    """
    Render regime indicator widget in Streamlit sidebar.
    Args:
        df: DataFrame with inflation data (must contain 'Все товары и услуги')
    Displays:
        - Colored badge showing current regime
        - Tooltip with explanation
        - Diagnostic metrics (Ki, Ruonia, Inflation changes)
        - Regime history timeline chart
    Supported regime types: normal, shock, high_inflation
    """
    try:
        from sirena.models.regime_detector import (
            detect_regime,
            get_regime_history,
            MacroRegime,
        )

        # Get current regime
        regime, regime_diag = detect_regime(df)
        # Get regime history
        history = get_regime_history(df)
        # Regime styling
        regime_info = {
            "value": regime.value,
            "emoji": {"normal": "🟢", "shock": "🔴", "high_inflation": "🟠"}[
                regime.value
            ],
            "label": {
                "normal": "Нормальный",
                "shock": "Шок",
                "high_inflation": "Высокая инфляция",
            }[regime.value],
            "color": {
                "normal": "#2ecc71",
                "shock": "#e74c3c",
                "high_inflation": "#f39c12",
            }[regime.value],
            "description": {
                "normal": "Стандартные макроэкономические условия",
                "shock": "Резкие изменения ставок (кризис)",
                "high_inflation": "Ускорение инфляции >1.5% YoY",
            }[regime.value],
            "diagnostics": regime_diag,
        }

        regime_history = history
    except Exception as e:
        return
        # Display in sidebar (assumes called from st.sidebar context)
        st.markdown("---")
        st.subheader("📊 Монитор режимов")
        # Regime badge with tooltip
        badge_html = f"""
        <div style="
            background-color: {regime_info["color"]};
            color: white;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="font-size: 24px; margin-bottom: 5px;">
                {regime_info["emoji"]} {regime_info["label"]}
            </div>
        </div>
        """
        st.markdown(badge_html, unsafe_allow_html=True)
        # Tooltip/explanation
        st.caption(f"💡 {regime_info['description']}")
        st.markdown("**Диагностика:**")
        st.caption(f"ΔKi (3м): {regime_info['diagnostics']['ki_change']:+.1f} п.п.")
        st.caption(
            f"ΔRuonia (3м): {regime_info['diagnostics']['ruonia_change']:+.1f} п.п."
        )
        st.caption(
            f"ΔИнфляция YoY: {regime_info['diagnostics']['yoy_change']:+.1f} п.п."
        )
        # Regime history timeline
        if not regime_history.empty:
            st.markdown("**История режимов:**")

            # Timeline chart
            fig_timeline = go.Figure()
            # Create regime timeline bars
            for i, (date, row) in enumerate(regime_history.iterrows()):
                regime_colors = {
                    "normal": "#2ecc71",
                    "shock": "#e74c3c",
                    "high_inflation": "#f39c12",
                }
                fig_timeline.add_trace(
                    go.Scatter(
                        x=[date, date],
                        y=[0, 1],
                        mode="lines",
                        line=dict(
                            color=regime_colors.get(row["regime"], "#95a5a6"),
                            width=8,
                        ),
                        showlegend=False,
                        hovertemplate=f"<b>{date.strftime('%Y-%m')}</b><br>"
                        f"Режим: {row['regime']}<br>"
                        f"ΔKi: {row['ki_change']:+.1f} п.п.<br>"
                        f"ΔRuonia: {row['ruonia_change']:+.1f} п.п.<extra></extra>",
                    )
                )
            # Add regime labels on timeline
            for regime in ["normal", "shock", "high_inflation"]:
                regime_data = regime_history[regime_history["regime"] == regime]
                if not regime_data.empty:
                    fig_timeline.add_trace(
                        go.Scatter(
                            x=[regime_data.index[0]],
                            y=[1.2],
                            mode="markers+text",
                            marker=dict(
                                color={
                                    "normal": "#2ecc71",
                                    "shock": "#e74c3c",
                                    "high_inflation": "#f39c12",
                                }[regime],
                                size=10,
                            ),
                            text={
                                regime: {
                                    "normal": "🟢",
                                    "shock": "🔴",
                                    "high_inflation": "🟠",
                                }[regime]
                            },
                            textposition="top center",
                            showlegend=False,
                        )
                    )
            fig_timeline.update_layout(
                title="История режимов",
                xaxis_title="Дата",
                yaxis=dict(showgrid=False, showticklabels=False, range=[-0.1, 1.5]),
                height=200,
                margin=dict(l=0, r=0, t=40, b=40),
                template="plotly_white",
            )

            st.plotly_chart(fig_timeline, use_container_width=True)
    except Exception as e:
        pass  # Silently fail if sidebar widget has issues


# =============================================================================
# MAIN APP
# =============================================================================
df = load_data()
if df is not None:
    last_date = df.index.max()
    # --- SIDEBAR: Regime Monitor ---
    try:
        with st.sidebar:
            render_regime_indicator(df)
    except Exception as e:
        pass  # Silently fail if sidebar widget has issues
    # --- HEADER ---
    st.title("📊 СИРЕНА-КБР v5.3")
    st.markdown(f"""
    **Система прогнозирования инфляции КБР**
    Последние данные: {last_date.strftime("%B %Y")}
    Модели: {len(ALL_MODELS)} | Горизонты: h=1,2,3,6,12 | Weekly, Экзогенные, Сценарии Ki
    """)

    # --- TABS ---
    (
        tab_f1,
        tab_f2,
        tab_f3,
        tab_f6,
        tab_f12,
        tab_seasonality,
        tab_macro,
        tab_sc,
        tab_exog,
        tab_weekly,
        tab_nowcast,
        tab_compare,
        tab_b1,
        tab_b2,
        tab_b3,
        tab_b6,
        tab_b12,
    ) = st.tabs(
        [
            "🎯 Прогноз h=1",
            "🎯 Прогноз h=2",
            "🎯 Прогноз h=3",
            "🎯 Прогноз h=6",
            "📈 Прогноз h=12",
            "📊 Сезонность",
            "🔍 Макро",
            "🎚️ Сценарии Ki",
            "📉 Экзогенные",
            "📈 Weekly",
            "📊 Nowcast",
            "🔍 Сравнение",
            "📊 Бэктест h=1",
            "📊 Бэктест h=2",
            "📊 Бэктест h=3",
            "📊 Бэктест h=6",
            "📊 Бэктест h=12",
        ]
    )
    # Preload backtest data
    bt_h1 = load_backtest_data(1)
    bt_h2 = load_backtest_data(2)
    bt_h3 = load_backtest_data(3)
    bt_h6 = load_backtest_data(6)
    # --- FORECAST TABS ---
    with tab_f1:
        render_forecast_tab(
            df, last_date, 1, bt_h1, ALL_MODELS, MONTH_NAMES_RU, MODEL_COLORS
        )
    with tab_f2:
        render_forecast_tab(
            df, last_date, 2, bt_h2, ALL_MODELS, MONTH_NAMES_RU, MODEL_COLORS
        )
    with tab_f3:
        render_forecast_tab(
            df, last_date, 3, bt_h3, ALL_MODELS, MONTH_NAMES_RU, MODEL_COLORS
        )
    with tab_f6:
        render_forecast_tab(
            df, last_date, 6, bt_h6, ALL_MODELS, MONTH_NAMES_RU, MODEL_COLORS
        )
    with tab_f12:
        render_forecast_h12_tab(df, last_date, MODEL_COLORS, load_backtest_data, 12)
    # --- RESEARCH TABS ---
    with tab_seasonality:
        render_seasonality_tab()

    with tab_macro:
        render_macro_tab()
    # --- SCENARIO & EXOGENOUS TABS ---
    with tab_sc:
        render_scenarios_tab(df, last_date)
    with tab_exog:
        render_exog_forecast_tab()
    with tab_weekly:
        render_weekly_tab()
    with tab_nowcast:
        render_nowcast_tab()
    # --- COMPARE TAB ---
    with tab_compare:
        render_compare_tab(df, last_date, ALL_MODELS, MONTH_NAMES_RU, MODEL_COLORS)
    # --- BACKTEST TABS ---
    with tab_b1:
        render_backtest_tab(1, ALL_MODELS)

    with tab_b2:
        render_backtest_tab(2, ALL_MODELS)
    with tab_b3:
        render_backtest_tab(3, ALL_MODELS)
    with tab_b6:
        render_backtest_tab(6, ALL_MODELS)
    with tab_b12:
        render_backtest_tab(12, ALL_MODELS)

else:
    st.error("Данные не загружены. Проверьте файл data/infl_kbr.csv")
