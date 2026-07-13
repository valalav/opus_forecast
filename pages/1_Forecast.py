"""
Forecast page functions for SIRENA-KBR Dashboard.

Contains functions for rendering forecast tabs and helper functions
for generating predictions from various models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# FORECAST FUNCTIONS
# =============================================================================


def get_best_model_for_horizon(horizon):
    """Return best model name for each horizon based on backtests."""
    best_models = {
        1: "Huber",
        2: "NGBoost_Shock",
        3: "Micro",  # Based on h=3 backtest
        6: "Micro",  # Based on h=6 backtest
        12: "Prophet",
    }
    return best_models.get(horizon, "Ridge")


def forecast_with_model(df, target_date, model_name):
    """Get forecast from specific model."""
    try:
        if model_name == "Huber":
            from sirena.models.huber import HuberForecaster

            model = HuberForecaster()
            model.fit(df)
            df_ext = df.copy()
            df_ext.loc[target_date] = np.nan
            return model.predict(df_ext, target_date)["prediction"] - 100

        elif model_name == "NGBoost_Shock":
            from sirena.models.ngboost_shock import NGBoostShockForecaster

            model = NGBoostShockForecaster()
            model.fit(df, "Все товары и услуги")
            df_ext = df.copy()
            df_ext.loc[target_date] = np.nan
            return model.predict(df_ext, target_date)["prediction"] - 100

        elif model_name == "Micro":
            from sirena.models.microcomponent import MicrocomponentForecaster

            model = MicrocomponentForecaster(horizon=1, use_seasonal_adj=True)
            model.fit(df, "Все товары и услуги")
            result = model.predict(df, target_date)
            if result and "prediction" in result:
                month = target_date.month
                seasonal_adj = model.SEASONAL_ADJ.get(month, 0)
                return result["prediction"] - 100 + seasonal_adj
            return np.nan

        elif model_name == "Micro_SM":
            from sirena.models.micro_statsmodels_external import (
                MicroStatsmodelsExternalForecaster,
            )

            model = MicroStatsmodelsExternalForecaster(horizon=1)
            model.fit(df, "Все товары и услуги")
            result = model.predict(df, target_date)
            if result and "prediction" in result and not np.isnan(result["prediction"]):
                return result["prediction"] - 100
            return np.nan

        elif model_name == "Prophet":
            from sirena.models.prophet import ProphetForecaster

            model = ProphetForecaster()
            model.fit(df, "Все товары и услуги")
            fc = model.forecast(horizon=1)
            return fc[0] if len(fc) > 0 else np.nan

        elif model_name == "Ridge":
            from sirena.models.ridge_extended import RidgeExtendedForecaster

            model = RidgeExtendedForecaster()
            model.fit(df)
            df_ext = df.copy()
            df_ext.loc[target_date] = np.nan
            return model.predict(df_ext, target_date)["prediction"] - 100

        return np.nan
    except Exception as e:
        st.warning(f"Ошибка модели {model_name}: {e}")
        return np.nan


def calculate_kpi_corrections(bt_data, model_name):
    """Calculate seasonal shift and bias corrections."""
    bt_data = bt_data.copy()
    bt_data["Month"] = bt_data["Date"].dt.month

    # Seasonal Shift (optimize for KPI)
    monthly_shifts = {}
    for month in range(1, 13):
        month_data = bt_data[bt_data["Month"] == month]
        if len(month_data) == 0:
            monthly_shifts[month] = 0
            continue
        best_shift, best_hits = 0, 0
        for shift in np.arange(-0.5, 0.51, 0.01):
            shifted = month_data[model_name] + shift
            hits = ((month_data["Actual"] - shifted).abs() <= 0.5).sum()
            if hits > best_hits:
                best_hits = hits
                best_shift = shift
        monthly_shifts[month] = best_shift

    # Bias Correction
    bias = (
        bt_data.groupby("Month")
        .apply(lambda x: (x[model_name] - x["Actual"]).mean())
        .to_dict()
    )

    return monthly_shifts, bias


# =============================================================================
# TAB RENDERERS
# =============================================================================


def render_forecast_tab(
    df, last_date, horizon, bt_data, ALL_MODELS, MONTH_NAMES_RU, MODEL_COLORS
):
    """Render single-horizon forecast tab (h=1, 2, 3, 6)."""
    target_date = last_date + pd.DateOffset(months=horizon)
    target_month = target_date.month

    st.subheader(f"🎯 Прогноз на {horizon} мес. вперёд")
    st.markdown(
        f"**Прогнозный месяц:** {MONTH_NAMES_RU[target_month - 1]} {target_date.year}"
    )

    if bt_data is None:
        st.error(
            f"Данные бэктеста не найдены. Запустите: python3 scripts/run_backtest_h{horizon}.py"
        )
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
        errors = (bt_data["Actual"] - bt_data[m]).abs()
        model_mae[m] = errors.mean()
    top5_models = sorted(model_mae.keys(), key=lambda x: model_mae[x])[:5]

    st.markdown("#### 📊 Сравнение моделей")

    # Get predictions from top-5 models
    predictions = {}
    for model in top5_models:
        pred = forecast_with_model(df, target_date, model)
        predictions[model] = pred

    # Create comparison table
    comp_df = pd.DataFrame(
        [
            {
                "Model": m,
                "MAE": model_mae[m],
                "Prediction": predictions[m],
                "Seasonal Adj": predictions[m] + monthly_shifts.get(target_month, 0),
                "Bias Adj": predictions[m] - bias.get(target_month, 0),
            }
            for m in top5_models
        ]
    )
    comp_df = comp_df.sort_values("MAE")

    st.dataframe(
        comp_df.style.format(
            {
                "MAE": "{:.3f}",
                "Prediction": "{:.2f}%",
                "Seasonal Adj": "{:.2f}%",
                "Bias Adj": "{:.2f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Download button
    csv = comp_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Скачать прогноз (CSV)",
        data=csv,
        file_name=f"forecast_h{horizon}_{target_date.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    # Plot comparison
    fig = go.Figure()
    for model in top5_models[:3]:
        fig.add_trace(
            go.Scatter(
                x=[target_date],
                y=[predictions[model]],
                mode="markers",
                name=model,
                marker=dict(size=15, color=MODEL_COLORS.get(model, "#000000")),
            )
        )
    fig.update_layout(
        title=f"Прогноз на {horizon} мес. вперёд (Топ-3 модели)",
        xaxis_title="Дата",
        yaxis_title="Инфляция MoM (%)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_forecast_h12_tab(
    df, last_date, MODEL_COLORS, load_backtest_data, horizon=12
):
    """Render 12-month trajectory forecast tab."""
    st.subheader("📈 Прогноз траектории на 12 месяцев")

    # Load precomputed forecasts or calculate
    bt_h12 = load_backtest_data(12)

    if bt_h12 is None:
        st.error(
            "Данные бэктеста h=12 не найдены. Запустите: python3 scripts/run_backtest_h12.py"
        )
        return

    # Get ensemble forecast
    with st.spinner("Расчёт ансамбля моделей..."):
        forecasts = {}
        dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS"
        )

        # Try to get forecasts from multiple models (top performers)
        models_to_try = [
            "Ridge",
            "Prophet",
            "BVAR",
            "ETS",
            "Huber",
            "LightGBM",
            "SubcompMulti",
            "Micro_SM",
        ]

        for model_name in models_to_try:
            try:
                if model_name == "Ridge":
                    from sirena.models.ridge_extended import RidgeExtendedForecaster

                    model = RidgeExtendedForecaster()
                    model.fit(df)
                    fc_vals = []
                    for h in range(horizon):
                        target = dates[h]
                        df_ext = df.copy()
                        df_ext.loc[target] = np.nan
                        pred = model.predict(df_ext, target)["prediction"] - 100
                        fc_vals.append(pred)
                    forecasts["Ridge"] = fc_vals

                elif model_name == "Prophet":
                    from sirena.models.prophet import ProphetForecaster

                    model = ProphetForecaster()
                    model.fit(df, "Все товары и услуги")
                    fc = model.forecast(horizon=horizon)
                    forecasts["Prophet"] = list(fc)

                elif model_name == "BVAR":
                    from sirena.models.bvar import BayesianVAR

                    # Load BVAR data
                    bvar_df = pd.read_csv(
                        "data/inflation_data.csv", sep=";", decimal=","
                    )
                    for col in [
                        "mom",
                        "Prod",
                        "Nonprod",
                        "Serv",
                        "usd_nom_i",
                        "Ruonia",
                    ]:
                        if col in bvar_df.columns:
                            if bvar_df[col].dtype == object:
                                bvar_df[col] = (
                                    bvar_df[col].astype(str).str.replace(",", ".")
                                )
                            bvar_df[col] = pd.to_numeric(bvar_df[col], errors="coerce")
                    bvar_df["Date"] = pd.to_datetime(
                        bvar_df["Date"], format="%d.%m.%Y", errors="coerce"
                    )
                    bvar_df["Date"] = (
                        bvar_df["Date"].dt.to_period("M").dt.to_timestamp()
                    )
                    bvar_df = bvar_df.set_index("Date").sort_index()

                    data = pd.DataFrame()
                    data["CPI"] = bvar_df["mom"] - 100
                    data["Food"] = bvar_df["Prod"] - 100
                    data["NonFood"] = bvar_df["Nonprod"] - 100
                    data["Services"] = bvar_df["Serv"] - 100

                    model = BayesianVAR(lags=6)
                    model.fit(data)
                    fc = model.forecast(horizon=horizon)
                    forecasts["BVAR"] = (fc["CPI"] + 100).tolist()

                elif model_name == "ETS":
                    from sirena.models.ets import ETSForecaster

                    model = ETSForecaster()
                    model.fit(df, "Все товары и услуги")
                    fc = model.forecast(horizon=horizon)
                    forecasts["ETS"] = list(fc)

                elif model_name == "Huber":
                    from sirena.models.huber import HuberForecaster

                    model = HuberForecaster()
                    model.fit(df)
                    fc_vals = []
                    for h in range(horizon):
                        target = dates[h]
                        df_ext = df.copy()
                        df_ext.loc[target] = np.nan
                        pred = model.predict(df_ext, target)["prediction"] - 100
                        fc_vals.append(pred)
                    forecasts["Huber"] = fc_vals

                elif model_name == "LightGBM":
                    from sirena.models.lightgbm import LightGBMForecaster

                    model = LightGBMForecaster()
                    model.fit(df, "Все товары и услуги")
                    fc = model.forecast(horizon=horizon)
                    forecasts["LightGBM"] = list(fc)

                elif model_name == "SubcompMulti":
                    from sirena.models.subcomponent_scenario import (
                        UnifiedSubcomponentForecaster,
                    )

                    model = UnifiedSubcomponentForecaster(horizon=12)
                    model.fit(df)
                    result = model.predict(df, last_date + pd.DateOffset(months=12))
                    if "forecast" in result:
                        forecasts["SubcompMulti"] = result["forecast"]

                elif model_name == "Micro_SM":
                    from sirena.models.micro_statsmodels_external import (
                        MicroStatsmodelsExternalForecaster,
                    )

                    model = MicroStatsmodelsExternalForecaster(horizon=1)
                    model.fit(df)
                    fc = model.forecast(horizon=horizon)
                    forecasts["Micro_SM"] = [
                        value - 100 if value > 50 else value for value in fc
                    ]

            except Exception as e:
                st.warning(f"Не удалось получить прогноз от {model_name}: {e}")
                continue

    if not forecasts:
        st.error("Не удалось получить прогноз ни от одной модели")
        return

    # Create forecast dataframe
    forecast_df = pd.DataFrame(
        {"Date": dates, **{k: v for k, v in forecasts.items() if len(v) == horizon}}
    )

    # Calculate ensemble (simple average)
    forecast_cols = [c for c in forecast_df.columns if c != "Date"]
    forecast_df["Ensemble"] = forecast_df[forecast_cols].mean(axis=1)

    # Display metrics
    st.markdown("#### 📊 Метрики")
    col1, col2, col3 = st.columns(3)
    col1.metric("Количество моделей", len(forecast_cols))
    col2.metric("Горизонт", f"{horizon} мес.")
    col3.metric("Среднее (Ensemble)", f"{forecast_df['Ensemble'].mean():.2f}%")

    st.markdown("#### 📋 Прогнозные значения")
    st.dataframe(
        forecast_df.style.format(
            {c: "{:.2f}%" for c in forecast_df.columns if c != "Date"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Download button
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Скачать прогноз (CSV)",
        data=csv,
        file_name=f"forecast_h12_{last_date.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    # Plot forecast trajectory
    fig = go.Figure()

    # Historical data
    hist_months = 12
    hist_df = df.tail(hist_months).reset_index()
    fig.add_trace(
        go.Scatter(
            x=hist_df["Date"],
            y=hist_df["Все товары и услуги"],
            mode="lines",
            name="Факт",
            line=dict(color="#000000", width=2),
        )
    )

    # Forecast lines
    for model in ["Ensemble"] + forecast_cols[:5]:
        color = MODEL_COLORS.get(model, "#000000")
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df[model],
                mode="lines+markers",
                name=model,
                line=dict(color=color, width=2),
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        title="Прогноз MoM инфляции на 12 месяцев",
        xaxis_title="Дата",
        yaxis_title="Инфляция MoM (%)",
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)
