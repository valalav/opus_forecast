"""
Dashboard Utilities for SIRENA-KBR.

Extracted utility functions for dashboard.py to improve code organization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

ALL_MODELS = [
    "Ridge",
    "Ridge_Ext",
    "Bayes_Ridge",
    "ElasticNet",
    "Huber",
    "Ridge_Shock",
    "Ridge_Macro",
    "NGBoost",
    "NGBoost_Shock",
    "BVAR",
    "SARIMA",
    "LightGBM",
    "Prophet",
    "ETS",
    "EBM",
    "CatBoost",
    "Subcomp",
    "Subcomp_Multi",
    "Micro",
    "Ensemble",
]

MODEL_COLORS = {
    "Ridge": "#1f77b4",
    "Ridge_Ext": "#aec7e8",
    "Bayes_Ridge": "#ff7f0e",
    "ElasticNet": "#ffbb78",
    "Huber": "#2ca02c",
    "Ridge_Shock": "#98df8a",
    "Ridge_Macro": "#2ecc71",
    "NGBoost": "#d62728",
    "NGBoost_Shock": "#ff9896",
    "BVAR": "#9467bd",
    "SARIMA": "#c5b0d5",
    "LightGBM": "#8c564b",
    "Prophet": "#c49c94",
    "ETS": "#e377c2",
    "EBM": "#f7b6d2",
    "CatBoost": "#7f7f7f",
    "Subcomp": "#c7c7c7",
    "Subcomp_Multi": "#bcbd22",
    "Micro": "#17becf",
    "Ensemble": "#000000",
    "Actual": "#000000",
    "Факт": "#000000",
}

MONTH_NAMES_RU = [
    "Январь", "Февраль", "Март", "Апрель",
    "Май", "Июнь", "Июль", "Август",
    "Сентябрь", "Октябрь", "Ноябрь", "Декабрь",
]


# =============================================================================
# DATA UTILITIES
# =============================================================================

def load_inflation_data(filepath: str = "data/inflation_data.csv") -> pd.DataFrame:
    """Load inflation data from CSV file."""
    df = pd.read_csv(filepath)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    return df


def calculate_mom_change(series: pd.Series) -> pd.Series:
    """Calculate month-over-month change."""
    return series.diff()


def calculate_yoy_change(series: pd.Series) -> pd.Series:
    """Calculate year-over-year change."""
    return series.diff(12)


def get_last_n_months(df: pd.DataFrame, n: int = 12) -> pd.DataFrame:
    """Get last N months of data."""
    return df.tail(n)


# =============================================================================
# METRICS CALCULATIONS
# =============================================================================

def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(actual[mask] - predicted[mask]))


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    mask = ~(np.isnan(actual) | np.isnan(predicted)) & (actual != 0)
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def calculate_kpi_violations(errors: np.ndarray, threshold: float = 0.5) -> int:
    """Count KPI violations (errors exceeding threshold)."""
    mask = ~np.isnan(errors)
    return int(np.sum(np.abs(errors[mask]) > threshold))


def calculate_coverage(errors: np.ndarray, threshold: float = 0.5) -> float:
    """Calculate coverage rate (% of errors within threshold)."""
    mask = ~np.isnan(errors)
    if mask.sum() == 0:
        return np.nan
    return (np.abs(errors[mask]) <= threshold).mean() * 100


# =============================================================================
# FORMATTING UTILITIES
# =============================================================================

def format_date_ru(date: datetime) -> str:
    """Format date in Russian locale."""
    if pd.isna(date):
        return ""
    month_idx = date.month - 1
    return f"{MONTH_NAMES_RU[month_idx]} {date.year}"


def format_percent(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}%"


def format_number(value: float, decimals: int = 3) -> str:
    """Format number with specified decimals."""
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


# =============================================================================
# MODEL RANKING
# =============================================================================

def rank_models_by_mae(
    predictions: pd.DataFrame,
    actual_col: str = "Actual",
    ascending: bool = True
) -> pd.DataFrame:
    """Rank models by MAE."""
    model_cols = [c for c in predictions.columns if c not in [actual_col, 'Date']]

    results = []
    for model in model_cols:
        mae = calculate_mae(
            predictions[actual_col].values,
            predictions[model].values
        )
        results.append({"model": model, "MAE": mae})

    df = pd.DataFrame(results)
    df = df.sort_values("MAE", ascending=ascending).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def get_model_color(model_name: str) -> str:
    """Get color for model, with fallback."""
    return MODEL_COLORS.get(model_name, "#808080")


# =============================================================================
# CHART HELPERS
# =============================================================================

def create_error_bands(
    center: np.ndarray,
    width: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Create upper and lower error bands."""
    return center + width, center - width


def interpolate_missing(series: pd.Series, method: str = "linear") -> pd.Series:
    """Interpolate missing values in series."""
    return series.interpolate(method=method)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_dataframe(
    df: pd.DataFrame,
    required_cols: List[str]
) -> Tuple[bool, List[str]]:
    """Validate DataFrame has required columns."""
    missing = [c for c in required_cols if c not in df.columns]
    return len(missing) == 0, missing


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Check data quality metrics."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_pct": (df.isna().sum().sum() / df.size) * 100,
        "duplicates": df.duplicated().sum(),
        "date_range": (df.index.min(), df.index.max()) if isinstance(df.index, pd.DatetimeIndex) else None,
    }


if __name__ == "__main__":
    # Quick test
    print("Dashboard Utils loaded successfully")
    print(f"Models: {len(ALL_MODELS)}")
    print(f"Colors: {len(MODEL_COLORS)}")
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
        manual_file_exists,
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


def render_feature_importance_tab(df):
    """Render Feature Importance tab."""
    st.subheader("🔍 Важность признаков (Feature Importance)")

    st.markdown("""
    Анализ важности признаков для различных моделей прогнозирования.
    Выберите модель, чтобы увидеть, какие факторы влияют на прогноз инфляции.
    """)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        selected_model = st.selectbox(
            "Выберите модель",
            ["Ridge", "Huber", "NGBoost"],
            index=0,
            label_visibility="collapsed",
        )

    with col2:
        st.markdown("**Важность:**")
        st.metric(
            "Всего признаков", "-", help="Количество признаков, используемых моделью"
        )

    with col3:
        st.markdown("**Тип:**")
        if selected_model == "Ridge":
            st.markdown("🔵 Линейная (Ridge)")
        elif selected_model == "Huber":
            st.markdown("🟡 Робастная (Huber)")
        else:
            st.markdown("🟣 Вероятностная (NGBoost)")

    st.markdown("---")

    # Train and get feature importance
    try:
        with st.spinner(
            f"Обучение модели {selected_model} и расчёт важности признаков..."
        ):
            from sklearn.inspection import permutation_importance
            from sklearn.linear_model import Ridge, HuberRegressor
            from sklearn.metrics import mean_absolute_error
            from sklearn.model_selection import train_test_split
            from sirena.models.ngboost_simple import NGBoostForecaster

            # Load and prepare data
            df_raw = pd.read_csv("data/infl_kbr.csv", sep=",")
            df_raw["Date"] = pd.to_datetime(df_raw["Date"])
            df_model = df_raw.set_index("Date")

            # Prepare features
            y = df_model["mom"].values
            X = pd.DataFrame(
                {
                    "y_lag1": df_model["mom"].shift(1),
                    "y_lag2": df_model["mom"].shift(2),
                    "y_lag3": df_model["mom"].shift(3),
                    "y_lag12": df_model["mom"].shift(12),
                    "month_sin": np.sin(2 * np.pi * df_model.index.month / 12),
                    "month_cos": np.cos(2 * np.pi * df_model.index.month / 12),
                }
            ).dropna()

            y = y[-len(X) :]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Fit selected model
            if selected_model == "Ridge":
                model = Ridge(alpha=0.1)
            elif selected_model == "Huber":
                model = HuberRegressor(epsilon=1.35, alpha=0.1)
            else:
                model = NGBoostForecaster(
                    n_estimators=100, learning_rate=0.01, random_state=42
                )

            model.fit(X_train, y_train)

            # Calculate permutation importance
            result = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=10,
                random_state=42,
                scoring="neg_mean_absolute_error",
            )

            # Importance is negative MAE decrease, take absolute for display
            importance_df = pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance": np.abs(result.importances_mean),
                    "std": result.importances_std,
                }
            ).sort_values("importance", ascending=False)

            importance_col = "importance"
            importance_label = "Важность"

        # Check if importance was computed
        if importance_df is None or len(importance_df) == 0:
            st.error("Модель не предоставила данные о важности признаков")
            return

        # Update metrics
        col2.metric("Всего признаков", len(importance_df))

        # Top 15 features bar chart (horizontal)
        top_features = importance_df.head(15)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=top_features[importance_col].values,
                y=top_features["feature"].values,
                orientation="h",
                marker=dict(
                    color=top_features[importance_col].values,
                    colorscale="Viridis",
                    colorbar=dict(title=importance_label),
                ),
                text=[f"{v:.4f}" for v in top_features[importance_col].values],
                textposition="outside",
            )
        )

        fig.update_layout(
            title=f"Топ-15 признаков по важности - {selected_model}",
            xaxis_title=importance_label,
            yaxis_title="Признак",
            height=500,
            yaxis=dict(autorange="reversed", gridcolor="#e5e5e5"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(gridcolor="#e5e5e5"),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Feature details table
        st.markdown("### 📋 Таблица важности всех признаков")

        with st.expander("Показать все признаки"):
            display_df = importance_df.copy()
            if "coefficient" in display_df.columns:
                display_df["Знак"] = display_df["coefficient"].apply(
                    lambda x: "Плюс (+)" if x > 0 else "Минус (-)"
                )
            if "is_macro" in display_df.columns:
                display_df["Тип"] = display_df["is_macro"].apply(
                    lambda x: "Макро" if x else "Базовый"
                )
            st.dataframe(
                display_df.style.background_gradient(
                    cmap="YlOrRd", subset=[importance_col]
                ).format({importance_col: "{:.4f}"}),
                use_container_width=True,
                height=300,
            )

        # Model info
        st.markdown("### ℹ️ Информация о модели")
        if hasattr(model, "get_model_info"):
            info = model.get_model_info()
            info_df = pd.DataFrame(
                {
                    "Параметр": list(info.keys()),
                    "Значение": [
                        str(v) if not isinstance(v, (list, dict)) else "..."
                        for v in info.values()
                    ],
                }
            )
            st.dataframe(info_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Ошибка при расчёте важности признаков: {e}")
        import traceback

        st.code(traceback.format_exc())


# =============================================================================
# MAIN APP
# =============================================================================
df = load_data()
df_macro = load_macro_data()

if df is not None:
    last_date = df.index.max()

    # --- SIDEBAR: REGIME MONITOR ---
    regime_indicator(df_macro, df)

    # --- SIDEBAR: ALERT PANEL ---
    alert_panel()

    # --- HEADER ---
    st.title("📊 СИРЕНА-КБР v5.2")
    st.markdown(f"""
    **Система прогнозирования инфляции КБР**
    Последние данные: {last_date.strftime("%B %Y")}
    Модели: {len(ALL_MODELS)} | Горизонты: h=1,2,3,6,12 | Экзогенные, Сценарии Ki
    """)

    # --- TABS ---
    (
        tab_f1,
        tab_f2,
        tab_f3,
        tab_f6,
        tab_f12,
        tab_sc,
        tab_exog,
        tab_fi,
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
            "🎚️ Сценарии Ki",
            "📉 Экзогенные",
            "🔍 Feature Importance",
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

    with tab_exog:
        render_exog_forecast_tab()

    with tab_fi:
        render_feature_importance_tab(df)

    # --- BACKTEST TABS ---
    with tab_b1:
        render_backtest_tab(1)

    with tab_b2:
        render_backtest_tab(2)

    with tab_b3:
        render_backtest_tab(3)

    with tab_b6:
        render_backtest_tab(6)

# =============================================================================
# DASHBOARD RENDER FUNCTIONS (moved from dashboard.py)
# =============================================================================

