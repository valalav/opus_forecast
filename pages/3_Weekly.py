"""Weekly monitoring and nowcasting tab functions for dashboard refactoring."""

import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime


def _format_percent(value):
    if value is None:
        return "n/a"
    try:
        return f"{float(value):+.3f}%"
    except (TypeError, ValueError):
        return "n/a"


def _json_default(value):
    """Serialize scalar values produced by pandas and NumPy."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _render_weekly_bridge_diagnostics(key_prefix: str):
    """Render precomputed weekly bridge diagnostics when available."""
    data_path = Path("data/precomputed_forecasts.json")
    if not data_path.exists():
        return

    try:
        with data_path.open("r", encoding="utf-8") as f:
            precomputed = json.load(f)
    except Exception as exc:
        st.warning(f"Не удалось прочитать weekly bridge diagnostics: {exc}")
        return

    bridge = precomputed.get("diagnostics", {}).get("weekly_bridge")
    if not isinstance(bridge, dict):
        return

    by_month = bridge.get("by_month", {})
    months = sorted(
        month
        for month, payload in by_month.items()
        if isinstance(payload, dict) and payload.get("chain")
    )
    if not months:
        return

    default_month = max(
        months,
        key=lambda month: (
            by_month[month].get("chain", {}).get("weeks_count", 0),
            month,
        ),
    )
    selected_month = st.selectbox(
        "Месяц диагностики weekly bridge",
        months,
        index=months.index(default_month),
        key=f"{key_prefix}_weekly_bridge_month",
        help="Диагностический мост из свежего semicolon-файла; не входит в production Ensemble.",
    )
    payload = by_month[selected_month]
    chain = payload.get("chain", {})
    month_end = payload.get("month_end", {})
    blend = payload.get("nowcast_blend", {})

    st.markdown("---")
    st.subheader("🧭 Weekly bridge diagnostics")
    st.caption(
        "Свежий недельный мост рассчитывается из row-wise semicolon-файла, "
        "дедуплицирует повторные блоки и хранится отдельно от Ensemble."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Цепочка WoW", _format_percent(chain.get("mom")))
    col2.metric("Сигнал с экстраполяцией", _format_percent(chain.get("extrapolated_mom")))
    col3.metric("Month-end bridge", _format_percent(month_end.get("mom")))
    col4.metric("Aux Nowcast blend", _format_percent(blend.get("nowcast_mom")))

    with st.expander("Детали weekly bridge", expanded=False):
        st.markdown(
            f"**Источник:** `{bridge.get('source_file', 'unknown')}`  \n"
            f"**Строки:** raw={bridge.get('raw_rows')}, "
            f"deduped={bridge.get('deduped_rows')}, "
            f"duplicates_removed={bridge.get('duplicates_removed')}  \n"
            f"**Метод:** {bridge.get('method')}"
        )

        weeks = chain.get("weeks", [])
        if weeks:
            st.markdown("**Недельная цепочка**")
            st.dataframe(
                pd.DataFrame(weeks)[["date", "index", "mom", "cumulative_mom", "n_items"]],
                use_container_width=True,
                hide_index=True,
            )

        components = month_end.get("components", {})
        if components:
            st.markdown("**Компоненты month-end bridge**")
            component_rows = []
            labels = {
                "food": "Продовольствие",
                "nonfood": "Непродовольственные",
                "services": "Услуги",
            }
            for key, row in components.items():
                component_rows.append(
                    {
                        "component": labels.get(key, key),
                        "index": row.get("index"),
                        "mom": row.get("mom"),
                        "n_items": row.get("n_items"),
                    }
                )
            st.dataframe(pd.DataFrame(component_rows), use_container_width=True, hide_index=True)

        decreases = month_end.get("top_decreases", [])
        increases = month_end.get("top_increases", [])
        if decreases or increases:
            left, right = st.columns(2)
            with left:
                st.markdown("**Топ снижений**")
                st.dataframe(
                    pd.DataFrame(decreases[:8])[["product_name", "index", "mom", "approx_contribution_pp"]],
                    use_container_width=True,
                    hide_index=True,
                )
            with right:
                st.markdown("**Топ ростов**")
                st.dataframe(
                    pd.DataFrame(increases[:8])[["product_name", "index", "mom", "approx_contribution_pp"]],
                    use_container_width=True,
                    hide_index=True,
                )


def render_alert_panel(show_history: bool = True):
    """
    Render price anomaly alert panel.

    Displays current anomalies from VolatilityMonitor with 1.5σ threshold.
    Color-coded by severity (warning/critical). Shows alert history.

    Args:
        show_history: Whether to display alert history visualization
    """
    import json
    from pathlib import Path

    # Initialize VolatilityMonitor
    from sirena.models.volatility_monitor import VolatilityMonitor

    monitor = VolatilityMonitor()
    monitor.initialize()
    anomalies = monitor.check_anomalies()

    # Save current alerts to history
    alert_history_path = Path("data/alert_history.json")
    current_timestamp = datetime.now().isoformat()

    # Load existing history
    if alert_history_path.exists():
        try:
            with open(alert_history_path, "r", encoding="utf-8") as f:
                history = json.load(f)
            if not isinstance(history, list):
                raise ValueError("alert history must be a JSON list")
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            st.warning(f"История alert-панели повреждена и будет пересоздана: {exc}")
            history = []
    else:
        history = []

    # Add current snapshot to history
    snapshot = {
        "timestamp": current_timestamp,
        "total_anomalies": len(anomalies),
        "critical_count": len([a for a in anomalies if a["level"] == "critical"]),
        "warning_count": len([a for a in anomalies if a["level"] == "warning"]),
        "anomalies": anomalies,
    }
    history.append(snapshot)

    # Keep only last 30 days of history
    cutoff = datetime.fromisoformat(current_timestamp).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - pd.Timedelta(days=30)
    history = [h for h in history if datetime.fromisoformat(h["timestamp"]) >= cutoff]

    # Save updated history
    alert_history_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_history_path = alert_history_path.with_suffix(".json.tmp")
    with open(temporary_history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2, default=_json_default)
    temporary_history_path.replace(alert_history_path)

    # Display alerts using st.expander("Alerts")
    with st.expander("Alerts", expanded=True):
        # Summary metrics
        critical = [a for a in anomalies if a["level"] == "critical"]
        warnings_list = [a for a in anomalies if a["level"] == "warning"]

        col1, col2, col3 = st.columns(3)

        with col1:
            if critical:
                st.error(f"🚨 {len(critical)} критических")
            else:
                st.success("✅ 0 критических")

        with col2:
            if warnings_list:
                st.warning(f"⚠️ {len(warnings_list)} предупреждений")
            else:
                st.success("✅ 0 предупреждений")

        with col3:
            st.metric("Всего аномалий", len(anomalies), help="1.5σ threshold")

        # Display critical alerts
        if critical:
            st.markdown("### 🔴 Критические аномалии")
            for a in critical[:10]:
                st.markdown(
                    f"- **{a.get('product_name', 'Unknown')[:40]}**: "
                    f"z={a['z_score']:+.1f} ({a.get('direction', 'N/A')})"
                )

        # Display warning alerts
        if warnings_list:
            st.markdown("### ⚠️ Предупреждения")
            with st.expander("Показать предупреждения", expanded=False):
                for a in warnings_list[:10]:
                    st.markdown(
                        f"- {a.get('product_name', 'Unknown')[:40]}: "
                        f"z={a['z_score']:+.1f}"
                    )

        # No alerts message
        if not critical and not warnings_list:
            st.success("✅ Аномалий не обнаружено")

        # Alert history visualization
        if show_history and len(history) > 1:
            st.markdown("---")
            st.markdown("### 📜 История аномалий (30 дней)")

            # Create history DataFrame
            history_df = pd.DataFrame(
                [
                    {
                        "Timestamp": pd.to_datetime(h["timestamp"]),
                        "Всего": h["total_anomalies"],
                        "Критические": h["critical_count"],
                        "Предупреждения": h["warning_count"],
                    }
                    for h in history
                ]
            )
            history_df = history_df.sort_values("Timestamp")

            # Plot history
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=history_df["Timestamp"],
                    y=history_df["Всего"],
                    mode="lines+markers",
                    name="Всего",
                    line=dict(color="#333333"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=history_df["Timestamp"],
                    y=history_df["Критические"],
                    mode="lines+markers",
                    name="Критические",
                    line=dict(color="#d62728"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=history_df["Timestamp"],
                    y=history_df["Предупреждения"],
                    mode="lines+markers",
                    name="Предупреждения",
                    line=dict(color="#ffbb78"),
                )
            )

            fig.update_layout(
                title="Динамика аномалий",
                xaxis_title="Дата",
                yaxis_title="Количество",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0),
            )

            st.plotly_chart(fig, use_container_width=True)


def render_weekly_tab():
    """Render weekly prices monitoring tab."""
    st.header("📈 Недельные цены")
    st.markdown("Мониторинг недельных цен и nowcasting инфляции")
    _render_weekly_bridge_diagnostics("weekly_tab")

    try:
        from sirena.data.weekly_loader import (
            load_weekly_prices,
            compute_basket_signal,
            HIGH_QUALITY_PRODUCTS,
            detect_anomalies,
        )
        from sirena.models.weekly_prices import WeeklyPriceNowcaster
        from sirena.models.volatility_monitor import VolatilityMonitor

        # Load weekly data
        weekly_df = load_weekly_prices()

        # Nowcast model with regime switching
        model = WeeklyPriceNowcaster(
            use_macro=False, use_components=True, use_regime=True
        )
        model.fit()
        nowcast_result = model.nowcast()

        col1, col2, col3 = st.columns(3)

        # Current nowcast signal (using regime-specific weights)
        col1.metric(
            "Nowcast Signal",
            f"{nowcast_result['weekly_signal']:.3f}%",
            help=f"Взвешенный сигнал недельных цен (режим: {nowcast_result.get('regime', 'N/A')})",
        )
        col2.metric(
            "Food Signal",
            f"{nowcast_result['food_signal']:.3f}%",
            help="Продовольственные товары",
        )
        col3.metric(
            "Non-Food Signal",
            f"{nowcast_result['nonfood_signal']:.3f}%",
            help="Непродовольственные товары",
        )

        # Regime info
        regime_str = str(nowcast_result.get("regime", "N/A"))
        coverage_pct = float(nowcast_result.get("coverage", 0)) * 100
        st.caption(
            f"Текущий режим: {regime_str.upper()} | Coverage: {coverage_pct:.0f}%"
        )

        # Price trends chart
        st.markdown("---")
        st.subheader("📊 Динамика цен")

        # Select products to display
        hq_codes = list(HIGH_QUALITY_PRODUCTS.keys())
        recent_weeks = weekly_df[
            weekly_df["date"] >= weekly_df["date"].max() - pd.Timedelta(weeks=12)
        ]

        # Aggregate by week
        weekly_agg = recent_weeks.groupby("date")["wow_growth"].mean().reset_index()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=weekly_agg["date"],
                y=weekly_agg["wow_growth"],
                mode="lines+markers",
                name="Средний WoW Growth",
                line=dict(color="#1f77b4", width=2),
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="Средний недельный рост цен (22 продукта)",
            xaxis_title="Дата",
            yaxis_title="WoW Growth (%)",
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Data summary
        st.markdown("---")
        st.subheader("📋 Данные")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Всего строк", f"{len(weekly_df):,}")
        col2.metric("Продуктов", int(weekly_df["product_code"].nunique()))
        col3.metric("High Quality", len(HIGH_QUALITY_PRODUCTS))
        col4.metric("Coverage", f"{coverage_pct:.0f}%")

        # Recent data table
        with st.expander("Последние данные"):
            recent = weekly_df.nlargest(100, "date")[
                ["date", "product_name", "price", "wow_growth"]
            ]
            st.dataframe(recent, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Ошибка загрузки недельных данных: {e}")
        st.info("Убедитесь что файл data/kbr_weekly_prices_2008_2026.csv существует")


def render_nowcast_tab():
    """Render nowcasting tab with regime detection and volatility comparison."""
    st.header("📊 Nowcast - Недельный мониторинг")
    st.markdown("Nowcasting инфляции на основе недельных цен с детекцией режима")
    _render_weekly_bridge_diagnostics("nowcast_tab")

    try:
        from sirena.data.weekly_loader import (
            load_weekly_prices,
            compute_basket_signal,
            HIGH_QUALITY_PRODUCTS,
            detect_anomalies,
        )
        from sirena.models.weekly_prices import WeeklyPriceNowcaster
        from sirena.models.volatility_monitor import VolatilityMonitor
        from sirena.models.volatility_weighted_nowcaster import (
            VolatilityWeightedNowcaster,
        )
        from sirena.models.regime_adaptive_nowcaster import RegimeAdaptiveNowcaster
        from sirena.models.regime_detector import detect_regime, MacroRegime

        # Load data
        weekly_df = load_weekly_prices()

        # Load inflation data for regime detection
        try:
            infl_df = pd.read_csv(
                "data/inflation_data.csv",
                sep=";",
                decimal=",",
                encoding="utf-8-sig",
            )
            infl_df["Date"] = pd.to_datetime(
                infl_df["Date"], format="%d.%m.%Y", errors="raise"
            )
            for column in ("mom", "Ki", "Ruonia"):
                if column in infl_df.columns:
                    infl_df[column] = pd.to_numeric(infl_df[column], errors="coerce")
            if "mom" in infl_df.columns and infl_df["mom"].median() > 50:
                infl_df["mom"] = infl_df["mom"] - 100
            infl_df = infl_df.sort_values("Date")
        except Exception as e:
            st.warning(
                f"Не удалось загрузить inflation_data.csv для детекции режима: {e}"
            )
            infl_df = None

        # ========================================
        # REGIME INDICATOR BADGE (Requirement 4)
        # ========================================
        st.markdown("---")
        st.subheader("🎯 Индикатор режима")

        if infl_df is not None and len(infl_df) > 0:
            try:
                regime, diagnostics = detect_regime(infl_df)

                # Regime badge colors
                regime_colors = {
                    MacroRegime.NORMAL: "🟢",
                    MacroRegime.SHOCK: "🔴",
                    MacroRegime.HIGH_INFLATION: "🟠",
                }

                regime_names = {
                    MacroRegime.NORMAL: "Нормальный",
                    MacroRegime.SHOCK: "Шок",
                    MacroRegime.HIGH_INFLATION: "Высокая инфляция",
                }

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(
                        f"""
                    <div style="padding: 10px; border-radius: 5px; background-color: #f0f0f0; text-align: center;">
                        <h3 style="margin: 0;">{regime_colors.get(regime, "⚪")}</h3>
                        <p style="margin: 5px 0 0 0; font-weight: bold;">Текущий режим</p>
                        <p style="margin: 0;">{regime_names.get(regime, str(regime))}</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    st.metric(
                        "Δ Ключевая ставка",
                        f"{diagnostics.get('ki_change', 0):.2f} п.п.",
                        help="Изменение ключевой ставки за последний месяц",
                    )

                with col3:
                    st.metric(
                        "Δ Ruonia",
                        f"{diagnostics.get('ruonia_change', 0):.2f} п.п.",
                        help="Изменение ставки RUONIA за последний месяц",
                    )

                # Regime explanation
                with st.expander("ℹ️ Объяснение режимов"):
                    st.markdown("""
                    **Режимы определяются на основе изменений ставок и инфляции:**

                    - 🟢 **Нормальный (NORMAL)**: Стабильная ситуация, |ΔKi| ≤ 0.5 п.п.
                    - 🔴 **Шок (SHOCK)**: Резкие изменения ставок, |ΔKi| > 0.5 п.п. или |ΔRuonia| > 0.5 п.п.
                    - 🟠 **Высокая инфляция (HIGH_INFLATION)**: Ускорение YoY инфляции > 1.5 п.п.

                    **Влияние на nowcasting:**
                    - В режиме Шок используются равные веса для всех продуктов
                    - В нормальном режиме оптимизируются веса по исторической точности
                    - При высокой инфляции повышаются веса волатильных товаров
                    """)
            except Exception as e:
                st.warning(f"Не удалось определить режим: {e}")
        else:
            st.info("Данные для детекции режима не доступны")

        # ========================================
        # WEEKLY PRICE SIGNALS (Requirement 2)
        # ========================================
        st.markdown("---")
        st.subheader("📈 Сигналы текущей недели")

        # Standard weighted signal (basket weights)
        signal = compute_basket_signal(weekly_df, use_weights=True)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Сигнал (взвешенный)",
            f"{signal['signal']:.3f}%",
            help="Средневзвешенный WoW рост по корзине товаров",
        )
        col2.metric(
            "Food Signal",
            f"{signal['food_signal']:.3f}%",
            help="Продовольственные товары",
        )
        col3.metric(
            "Non-Food Signal",
            f"{signal['nonfood_signal']:.3f}%",
            help="Непродовольственные товары",
        )

        # ========================================
        # VOLATILITY-WEIGHTED vs EQUAL-WEIGHTED (Requirement 3)
        # ========================================
        st.markdown("---")
        st.subheader("⚖️ Сравнение методов взвешивания")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### 📊 Volatility-Weighted")
            st.markdown("""
            Веса пропорциональны обратной волатильности:
            $$w_i = \\frac{1/\\sigma_i}{\\sum(1/\\sigma_j)}$$

            Стабильные товары получают больший вес.
            """)

            try:
                # Calculate equal-weighted signal for comparison
                signal_equal = compute_basket_signal(weekly_df, use_weights=False)

                st.metric(
                    "Equal-Weighted Signal",
                    f"{signal_equal.get('signal', 0):.3f}%",
                    help="Равновзвешенный сигнал",
                )

                # Show volatility info from weekly data
                st.markdown("**Топ стабильные продукты (низкая волатильность):**")
                hq_products = list(HIGH_QUALITY_PRODUCTS.items())

                # Calculate volatility for each product from recent data
                recent_weeks = weekly_df[
                    weekly_df["date"]
                    >= weekly_df["date"].max() - pd.Timedelta(weeks=12)
                ]

                vol_by_product = []
                for code, metadata in hq_products:
                    prod_data = recent_weeks[recent_weeks["product_code"] == code]
                    if len(prod_data) > 0:
                        vol = prod_data["wow_growth"].std()
                        vol_by_product.append(
                            {
                                "code": code,
                                "name": metadata.get("name", str(code)),
                                "volatility": vol,
                            }
                        )

                if vol_by_product:
                    vol_df = (
                        pd.DataFrame(vol_by_product).sort_values("volatility").head(5)
                    )
                    for _, row in vol_df.iterrows():
                        st.markdown(f"- {row['name'][:35]}: σ={row['volatility']:.3f}%")

            except Exception as e:
                st.warning(f"Ошибка загрузки Volatility-Weighted модели: {e}")

        with col_right:
            st.markdown("### 🔹 Equal-Weighted")
            st.markdown("""
            Все продукты имеют равный вес:

            $$w_i = \\frac{1}{N}$$

            Рекомендуется в режиме Шок.
            """)

            # Calculate equal-weighted signal for comparison
            signal_equal = compute_basket_signal(weekly_df, use_weights=False)

            st.metric(
                "Equal-Weighted Signal",
                f"{signal_equal.get('signal', 0):.3f}%",
                help="Равновзвешенный сигнал",
            )

            # Difference between weighted and equal-weighted
            diff = signal_equal.get("signal", 0) - signal.get("signal", 0)
            diff_color = "🟢" if abs(diff) < 0.05 else "🟡" if abs(diff) < 0.1 else "🔴"
            st.markdown(f"""
            **Разница vs Basket-Weighted:** {diff_color} {diff:+.3f} п.п.

            {"" if abs(diff) < 0.05 else "⚠️ Существенное расхождение в методах взвешивания!"}
            """)

            # Equal-weighted signal
            signal_equal = compute_basket_signal(weekly_df, use_weights=False)

            st.metric(
                "Equal-Weighted Signal",
                f"{signal_equal['signal']:.3f}%",
                help="Равновзвешенный сигнал",
            )

            # Difference
            diff = signal_equal["signal"] - signal["signal"]
            diff_color = "🟢" if abs(diff) < 0.05 else "🟡" if abs(diff) < 0.1 else "🔴"
            st.markdown(f"""
            **Разница:** {diff_color} {diff:+.3f} п.п.

            {"" if abs(diff) < 0.05 else "⚠️ Существенное расхождение в методах взвешивания!"}
            """)

        # ========================================
        # NOWCAST MODEL PREDICTIONS
        # ========================================
        st.markdown("---")
        st.subheader("🔮 Nowcast модель")

        model = WeeklyPriceNowcaster(
            use_macro=False, use_components=True, use_regime=True
        )
        model.fit()
        nowcast = model.nowcast()

        legacy_target = pd.to_datetime(nowcast.get("target_date"), errors="coerce")
        target_label = (
            legacy_target.strftime("%Y-%m") if pd.notna(legacy_target) else "н/д"
        )
        latest_legacy_week = weekly_df["date"].max()
        col1.metric(
            f"Прогноз месяца {target_label}",
            f"{nowcast['prediction']:.2f}%",
            help=f"Coverage: {nowcast['coverage'] * 100:.0f}%",
        )
        latest_official_month = infl_df["Date"].max() if infl_df is not None else None
        if (
            latest_official_month is not None
            and pd.notna(legacy_target)
            and legacy_target.to_period("M")
            < latest_official_month.to_period("M")
        ):
            st.warning(
                "Историческая weekly-модель отстаёт: последние недельные данные "
                f"{latest_legacy_week:%Y-%m-%d}, тогда как официальный ряд уже "
                f"до {latest_official_month:%Y-%m}. Для текущего месяца используйте "
                "fresh weekly bridge выше."
            )
        col2.metric(
            "Продуктов использовано",
            nowcast.get("n_products", 0),
            help="Количество продуктов в nowcast",
        )
        mae = nowcast.get("mae")
        mae_value = (
            f"{float(mae):.4f}"
            if isinstance(mae, (int, float, np.number)) and pd.notna(mae)
            else "н/д"
        )
        col3.metric(
            "Точность (MAE)",
            mae_value,
            help="Нужен отдельный исторический бэктест; текущий nowcast его не рассчитывает.",
        )

        # ========================================
        # VOLATILITY ALERTS
        # ========================================
        render_alert_panel(show_history=True)

        # ========================================
        # PRICE TRENDS CHART
        # ========================================
        st.markdown("---")
        st.subheader("📈 Динамика цен (последние 12 недель)")

        recent_weeks = weekly_df[
            weekly_df["date"] >= weekly_df["date"].max() - pd.Timedelta(weeks=12)
        ]
        weekly_agg = recent_weeks.groupby("date")["wow_growth"].mean().reset_index()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=weekly_agg["date"],
                y=weekly_agg["wow_growth"],
                mode="lines+markers",
                name="Средний WoW Growth",
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=6),
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")

        # Add current signal line
        if signal["signal"] is not None and not pd.isna(signal["signal"]):
            fig.add_hline(
                y=signal["signal"],
                line_dash="solid",
                line_color="green",
                annotation_text=f"Текущий сигнал: {signal['signal']:.2f}%",
                annotation_position="top right",
            )

        fig.update_layout(
            title="Средний недельный рост цен",
            xaxis_title="Дата",
            yaxis_title="WoW Growth (%)",
            template="plotly_white",
            height=400,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ========================================
        # DATA SUMMARY
        # ========================================
        st.markdown("---")
        st.subheader("📋 Сводка данных")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Всего строк", f"{len(weekly_df):,}")
        col2.metric("Продуктов", int(weekly_df["product_code"].nunique()))
        col3.metric("High Quality", len(HIGH_QUALITY_PRODUCTS))
        col4.metric("Coverage", f"{signal['coverage'] * 100:.0f}%")

        # Recent data table
        with st.expander("📄 Последние данные"):
            recent = weekly_df.nlargest(50, "date")[
                ["date", "product_name", "price", "wow_growth"]
            ]
            st.dataframe(recent, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Ошибка загрузки Nowcast данных: {e}")
        import traceback

        st.text(traceback.format_exc())
