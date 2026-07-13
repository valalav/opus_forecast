"""
Forecast page functions for SIRENA-KBR Dashboard.

Contains functions for rendering forecast tabs and helper functions
for generating predictions from various models.
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# FORECAST FUNCTIONS
SEND_READY_POLICY_PATH = Path("data/send_ready_policy_trajectory.json")
SEND_READY_POLICY_COLUMN = "Отправочная траектория"


def load_send_ready_policy_trajectory(path, expected_dates):
    """Load a current expert policy path aligned to the cached production horizon."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    policy_dates = pd.to_datetime(payload["forecast_dates"])
    policy_values = np.asarray(payload["mom_pp"], dtype=float)
    expected_dates = pd.DatetimeIndex(expected_dates)

    if (
        policy_values.ndim != 1
        or len(policy_values) != len(expected_dates)
        or len(policy_dates) != len(expected_dates)
        or not np.array_equal(policy_dates.to_numpy(), expected_dates.to_numpy())
        or not np.isfinite(policy_values).all()
    ):
        raise ValueError(
            "send-ready policy trajectory does not match the production horizon"
        )
    return policy_values


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
    """Render the current send-ready path and cached production trajectory."""
    st.subheader("📈 Прогноз траектории на 12 месяцев")

    expected_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS"
    )
    try:
        cache_path = Path("data/precomputed_forecasts.json")
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached_dates = pd.to_datetime(cached["forecast_dates"])
        cached_forecasts = cached["forecasts"]
        if (
            len(cached_dates) != horizon
            or not np.array_equal(cached_dates.to_numpy(), expected_dates.to_numpy())
        ):
            raise ValueError(
                "cache dates do not match the latest official observation; "
                "run scripts/precompute_forecasts.py"
            )
        if not isinstance(cached_forecasts, dict):
            raise ValueError("forecast cache has no forecasts mapping")

        production_forecasts = {}
        for name, values in cached_forecasts.items():
            try:
                path = np.asarray(values, dtype=float)
            except (TypeError, ValueError):
                continue
            if len(path) == horizon and np.isfinite(path).all():
                production_forecasts[name] = path

        if "Ensemble" not in production_forecasts:
            raise ValueError("forecast cache has no finite production Ensemble")
        forecast_df = pd.DataFrame({"Date": cached_dates, **production_forecasts})
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        st.error(f"Не удалось загрузить production h=12 Ensemble: {exc}")
        return

    try:
        policy_values = load_send_ready_policy_trajectory(
            SEND_READY_POLICY_PATH, cached_dates
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        st.warning(
            "Текущая отправочная траектория недоступна или устарела; показан "
            f"только model Ensemble. Обновите `{SEND_READY_POLICY_PATH}`: {exc}"
        )
    else:
        forecast_df.insert(1, SEND_READY_POLICY_COLUMN, policy_values)

    if load_backtest_data(12) is None:
        st.warning(
            "Бэктест h=12 не загружен; показана текущая production-траектория "
            "из precomputed_forecasts.json."
        )

    forecast_cols = [column for column in forecast_df.columns if column != "Date"]
    diagnostic_cols = [
        column
        for column in forecast_cols
        if column not in {"Ensemble", SEND_READY_POLICY_COLUMN}
    ]
    has_send_ready_policy = SEND_READY_POLICY_COLUMN in forecast_df.columns
    first_period = forecast_df.loc[0, "Date"].strftime("%Y-%m")
    second_period = forecast_df.loc[1, "Date"].strftime("%Y-%m")

    st.markdown("#### 📊 Текущая отправочная траектория и модельный ориентир")
    col1, col2, col3, col4 = st.columns(4)
    if has_send_ready_policy:
        col1.metric(
            f"{first_period}: отправочная",
            f"{forecast_df.loc[0, SEND_READY_POLICY_COLUMN]:.2f}%",
        )
        col2.metric(
            f"{second_period}: отправочная",
            f"{forecast_df.loc[1, SEND_READY_POLICY_COLUMN]:.2f}%",
        )
        col3.metric(
            f"{first_period}: model Ensemble",
            f"{forecast_df.loc[0, 'Ensemble']:.2f}%",
        )
        st.caption(
            "Отправочная траектория — экспертная central policy-ветка с частичной "
            "реализацией топливного риска; она является текущим send-ready путём. "
            "Ensemble — отдельный сохранённый взвешенный модельный ориентир."
        )
    else:
        col1.metric(
            f"{first_period}: model Ensemble",
            f"{forecast_df.loc[0, 'Ensemble']:.2f}%",
        )
        col2.metric("Среднее: model Ensemble", f"{forecast_df['Ensemble'].mean():.2f}%")
        col3.metric("Горизонт", f"{horizon} мес.")
        st.caption(
            "Показан только сохранённый взвешенный model Ensemble; отправочная "
            "траектория требует актуализации."
        )
    col4.metric("Моделей в production cache", len(diagnostic_cols))
    st.caption(
        "BVAR, ETS, LightGBM и другие auxiliary-модели не пересчитываются и не "
        "усредняются на этой вкладке. VARPolicy использует обязательную "
        "SeasonalVAR траекторию после h=1."
    )

    st.markdown("#### 📋 Прогнозные значения")
    st.dataframe(
        forecast_df.style.format(
            {column: "{:.2f}%" for column in forecast_df.columns if column != "Date"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Скачать прогноз (CSV)",
        data=csv,
        file_name=f"forecast_h12_{last_date.strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

    fig = go.Figure()
    hist_df = df.tail(12).reset_index()
    history = pd.to_numeric(
        hist_df["Все товары и услуги"], errors="coerce"
    ).to_numpy(dtype=float)
    if np.nanmedian(history) > 50:
        history = history - 100
    fig.add_trace(
        go.Scatter(
            x=hist_df["Date"],
            y=history,
            mode="lines",
            name="Факт",
            line=dict(color="#000000", width=2),
        )
    )
    if has_send_ready_policy:
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df[SEND_READY_POLICY_COLUMN],
                mode="lines+markers",
                name=SEND_READY_POLICY_COLUMN,
                line=dict(color="#D62728", width=4),
                marker=dict(size=7),
            )
        )

    for model in [
        "Ensemble",
        "VARPolicy",
        "Ridge_ProdProxy",
        "Huber",
        "Micro",
    ]:
        if model not in forecast_df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df[model],
                mode="lines+markers",
                name=model,
                line=dict(color=MODEL_COLORS.get(model, "#000000"), width=2),
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        title=(
            "Отправочная и production-траектории MoM на 12 месяцев"
            if has_send_ready_policy
            else "Production-траектории MoM на 12 месяцев"
        ),
        xaxis_title="Дата",
        yaxis_title="Инфляция MoM (%)",
        hovermode="x unified",
        template="plotly_white",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)
