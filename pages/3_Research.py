"""
Research page functions for SIRENA-KBR Dashboard.

Contains functions for rendering research tabs (Seasonality, Macro analysis).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings

warnings.filterwarnings("ignore")


# =============================================================================
# SEASONALITY TAB
# =============================================================================


def render_seasonality_tab():
    """Render seasonality analysis tab."""
    st.header("📈 Сезонность")
    st.markdown("Анализ эволюции сезонных паттернов инфляции")

    try:
        # Load seasonal indices data
        seasonal_df = pd.read_csv("data/seasonal_indices_by_era.csv")

        # Product selection (placeholder for future multi-product support)
        st.subheader("🔍 Выбор параметров")
        product_select = st.selectbox(
            "Товар",
            ["Все товары и услуги"],
            disabled=True,
            help="В данный момент доступен только общий ИПЦ",
        )

        # Era selection
        era_options = seasonal_df["Era"].unique().tolist()
        era_select = st.multiselect(
            "Выберите периоды для сравнения",
            era_options,
            default=["2010-2014 (Pre-Crimea)", "2020-2024 (COVID & Recovery)"],
            help="Сравните сезонные паттерны между разными эпохами",
        )

        if not era_select:
            st.warning("Выберите хотя бы один период")
            return

        # Filter selected eras
        df_filtered = seasonal_df[seasonal_df["Era"].isin(era_select)].copy()

        # Month names for display
        month_names = [
            "Янв",
            "Фев",
            "Мар",
            "Апр",
            "Май",
            "Июн",
            "Июл",
            "Авг",
            "Сен",
            "Окт",
            "Ноя",
            "Дек",
        ]

        # Create comparison plot
        st.markdown("---")
        st.subheader("📊 Сезонные индексы по эпохам")

        fig = go.Figure()

        # Colors for different eras
        era_colors = {
            "2010-2014 (Pre-Crimea)": "#1f77b4",
            "2015-2019 (Post-Crimea)": "#ff7f0e",
            "2020-2024 (COVID & Recovery)": "#2ca02c",
        }

        # Add traces for each era
        for era in era_select:
            era_df = df_filtered[df_filtered["Era"] == era]
            fig.add_trace(
                go.Scatter(
                    x=month_names,
                    y=era_df["Mean_Index"],
                    mode="lines+markers",
                    name=era,
                    line=dict(color=era_colors.get(era, "#7f7f7f"), width=2),
                    marker=dict(size=8),
                    hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
                )
            )

        fig.update_layout(
            title="Сезонные индексы инфляции (Mean Index)",
            xaxis_title="Месяц",
            yaxis_title="Средняя инфляция (%)",
            template="plotly_white",
            height=500,
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Comparison metrics
        st.markdown("---")
        st.subheader("📋 Сравнительная статистика")

        # Create pivot table for comparison
        pivot_df = df_filtered.pivot(index="Month", columns="Era", values="Mean_Index")
        pivot_df.index = month_names

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Средняя за период (2010-2014)",
                f"{pivot_df['2010-2014 (Pre-Crimea)'].mean():.2f}%",
                help="Средняя сезонность за 2010-2014",
            )

        with col2:
            if "2015-2019 (Post-Crimea)" in pivot_df.columns:
                st.metric(
                    "Средняя за период (2015-2019)",
                    f"{pivot_df['2015-2019 (Post-Crimea)'].mean():.2f}%",
                    help="Средняя сезонность за 2015-2019",
                )

        with col3:
            st.metric(
                "Средняя за период (2020-2024)",
                f"{pivot_df['2020-2024 (COVID & Recovery)'].mean():.2f}%",
                help="Средняя сезонность за 2020-2024",
            )

        # Volatility comparison (Std)
        st.markdown("### Волатильность сезонности (Std)")

        std_pivot = df_filtered.pivot(index="Month", columns="Era", values="Std")
        std_pivot.index = month_names

        fig_std = go.Figure()

        for era in era_select:
            era_std_df = df_filtered[df_filtered["Era"] == era]
            fig_std.add_trace(
                go.Bar(
                    x=month_names,
                    y=era_std_df["Std"],
                    name=era,
                    marker_color=era_colors.get(era, "#7f7f7f"),
                    hovertemplate="%{x}: Std = %{y:.3f}<extra></extra>",
                )
            )

        fig_std.update_layout(
            title="Стандартное отклонение сезонной инфляции",
            xaxis_title="Месяц",
            yaxis_title="Std",
            template="plotly_white",
            height=400,
            barmode="group",
            hovermode="x unified",
        )

        st.plotly_chart(fig_std, use_container_width=True)

        # Key months analysis
        st.markdown("---")
        st.subheader("🗓️ Ключевые месяцы")

        key_months = {
            1: "Январь (Новый год/Тарифы)",
            7: "Июль (Индексация тарифов)",
            12: "Декабрь (Предпраздничный)",
        }

        for month_num, description in key_months.items():
            month_data = df_filtered[df_filtered["Month"] == month_num].copy()
            month_data = month_data.sort_values("Era")

            with st.expander(f"{description}"):
                for _, row in month_data.iterrows():
                    st.write(
                        f"**{row['Era']}**: {row['Mean_Index']:.2f}% (Std: {row['Std']:.3f})"
                    )

        # Data table
        st.markdown("---")
        st.subheader("📊 Полные данные")

        display_df = df_filtered[
            ["Month", "Era", "Mean_Index", "Median_Index", "Std", "Year_Count"]
        ].copy()
        display_df["Month_Name"] = display_df["Month"].apply(
            lambda x: month_names[x - 1]
        )
        display_df = display_df[
            ["Month_Name", "Era", "Mean_Index", "Median_Index", "Std", "Year_Count"]
        ]
        display_df.columns = [
            "Месяц",
            "Эпоха",
            "Mean Index",
            "Median Index",
            "Std",
            "Кол-во лет",
        ]

        st.dataframe(display_df, use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.error("Файл data/seasonal_indices_by_era.csv не найден")
        st.info(
            "Запустите скрипт python3 scripts/seasonal_evolution.py для генерации данных"
        )
    except Exception as e:
        st.error(f"Ошибка загрузки данных сезонности: {e}")
        import traceback

        st.text(traceback.format_exc())


# =============================================================================
# MACRO ANALYSIS TAB
# =============================================================================


def render_macro_tab():
    """Render macro analysis tab showing Fed Rate, Brent, USD correlations."""
    st.header("🔍 Макро-анализ")
    st.markdown("Анализ влияния федеральных и внешних факторов на инфляцию КБР")

    try:
        # Load Fed transmission results
        fed_results = pd.read_csv("data/fed_transmission_results.csv")

        # Load main inflation data with macro indicators
        try:
            df_macro = pd.read_csv("data/inflation_data.csv", sep=";", decimal=",")
            df_macro["Date"] = pd.to_datetime(df_macro["Date"], format="%d.%m.%Y")
            df_macro = df_macro.set_index("Date")

            # Extract CPI (mom)
            cpi = df_macro["mom"].values
            usd = df_macro.get("usd_nom_i", pd.Series([np.nan] * len(df_macro))).values
            brent_available = "brent" in df_macro.columns
        except:
            cpi = usd = None
            brent_available = False

        # Parse coefficients from string format
        def parse_coefficients(coeff_str):
            import ast

            try:
                return ast.literal_eval(coeff_str)
            except:
                return {}

        fed_results["coefficients_dict"] = fed_results["coefficients"].apply(
            parse_coefficients
        )

        st.markdown("---")

        # Plot 1: Fed Rate Lag Analysis (Coefficients)
        st.subheader("📊 Анализ лагов: Ставка ФРС vs ИПЦ")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig1 = go.Figure()

            # Filter Simple model (no controls)
            simple_df = fed_results[fed_results["model"] == "Simple"]

            fig1.add_trace(
                go.Bar(
                    x=simple_df["lag"],
                    y=simple_df["fed_coefficient"],
                    name="Коэффициент",
                    marker_color=simple_df["significant"].apply(
                        lambda x: "#2ca02c" if x else "#d62728"
                    ),
                    text=simple_df["fed_coefficient"].round(4),
                    textposition="outside",
                    hovertemplate=(
                        "Лаг: %{x} мес<br>"
                        "Коэффициент: %{y:.4f}<br>"
                        "Значимость: %{customdata}<extra></extra>"
                    ),
                    customdata=simple_df["significant"].apply(
                        lambda x: "Да" if x else "Нет"
                    ),
                )
            )

            fig1.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

            fig1.update_layout(
                title="Коэффициент регрессии по лагу (Simple Model)",
                xaxis_title="Лаг (месяцев)",
                yaxis_title="Коэффициент",
                template="plotly_white",
                height=400,
                showlegend=False,
            )

            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("### 📋 Интерпретация")
            st.markdown("""
            **Положительный коэффициент** → Ставка ФРС ↑ → Инфляция ↑
            **Отрицательный коэффициент** → Ставка ФРС ↑ → Инфляция ↓

            **Значимость** (зеленый):
            - p < 0.05 → статистически значимая связь
            - Все лаги: НЕ значимы (p > 0.1)
            """)

            st.markdown("#### R² по моделям")
            for model in ["Simple", "Controlled", "Full"]:
                model_df = fed_results[fed_results["model"] == model]
                best_r2 = model_df["r2"].max()
                best_lag = model_df.loc[model_df["r2"].idxmax(), "lag"]
                st.markdown(f"- **{model}**: R² = {best_r2:.4f} (лаг {best_lag})")

        st.markdown("---")

        # Plot 2: R² Comparison
        st.subheader("📈 Качество моделей по лагам")

        fig2 = go.Figure()

        for model in ["Simple", "Controlled", "Full"]:
            model_df = fed_results[fed_results["model"] == model]
            fig2.add_trace(
                go.Scatter(
                    x=model_df["lag"],
                    y=model_df["r2"],
                    mode="lines+markers",
                    name=model,
                    line=dict(width=2),
                    marker=dict(size=8),
                    hovertemplate="Лаг: %{x}<br>R²: %{y:.4f}<extra></extra>",
                )
            )

        fig2.update_layout(
            title="R² по лагам для разных моделей",
            xaxis_title="Лаг (месяцев)",
            yaxis_title="R² (объясняющая способность)",
            template="plotly_white",
            height=450,
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5
            ),
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        **Модели:**
        - **Simple**: Только ставка ФРС
        - **Controlled**: Ставка ФРС + изменение курса USD
        - **Full**: Ставка ФРС + USD + Brent
        """)

        st.markdown("---")

        # Plot 3: Correlation Heatmap
        if cpi is not None and usd is not None:
            st.subheader("🔗 Корреляция макро-индикаторов")

            # Calculate correlations
            corr_data = {
                "ИПЦ (CPI)": cpi,
                "USD (номинал)": usd,
            }

            if brent_available and "brent" in df_macro.columns:
                corr_data["Brent Oil"] = df_macro["brent"].values

            corr_df = pd.DataFrame(corr_data)
            corr_matrix = corr_df.corr()

            fig3 = go.Figure(
                data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale="RdBu",
                    zmid=0,
                    text=corr_matrix.values.round(3),
                    texttemplate="%{text}",
                    textfont={"size": 14},
                    hovertemplate=(
                        "X: %{x}<br>Y: %{y}<br>Корреляция: %{z:.3f}<extra></extra>"
                    ),
                )
            )

            fig3.update_layout(
                title="Корреляционная матрица",
                template="plotly_white",
                height=400,
            )

            st.plotly_chart(fig3, use_container_width=True)

        st.markdown("---")

        # Detailed results table
        st.subheader("📋 Детальные результаты регрессии")

        # Filter to show significant results or all results
        show_all = st.checkbox(
            "Показать все результаты (включая незначимые)", value=False
        )

        if show_all:
            display_df = fed_results.copy()
        else:
            display_df = fed_results[fed_results["significant"] == True].copy()

        if display_df.empty:
            st.warning("Нет статистически значимых результатов")
        else:
            # Format for display
            display_df["Коэффициент (ФРС)"] = display_df["fed_coefficient"].round(4)
            display_df["p-value"] = display_df["fed_p_value"].round(4)
            display_df["R²"] = display_df["r2"].round(4)

            columns_to_show = [
                "model",
                "lag",
                "Коэффициент (ФРС)",
                "p-value",
                "R²",
                "significant",
            ]

            st.dataframe(
                display_df[columns_to_show].rename(
                    columns={
                        "model": "Модель",
                        "lag": "Лаг",
                        "significant": "Значимость",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")

        # Key insights
        st.subheader("💡 Ключевые выводы")

        st.markdown("""
        **На основе анализа (Task 524):**

        1. **Прямая передача ставки ФРС**: Коэффициенты положительные, но НЕ значимы (p > 0.1)

        2. **Лаг 1 месяц имеет самый сильный эффект**:
           - Simple модель: R² = 0.0037 при лаге 1
           - Добавление контролей (USD, Brent) улучшает R² до 0.144

        3. **Механизм передачи**:
           - Через курс USD: Ставка ФРС ↑ → USD ↓ → Инфляция ↓ (отрицательный эффект)
           - Через цены нефти: Ставка ФРС ↑ → Нефть ↓ → Инфляция ↓

        4. **Полная модель (Full) показывает лучший R²**:
           - Лаг 4: R² = 0.0142
           - Но даже в полной модели эффект ставки ФРС НЕ значим

        **Вывод**: Ставка ФРС имеет слабое и статистически незначимое влияние на инфляцию КБР.
        Основной канал передачи — через обменный курс USD и цены нефти.
        """)

    except FileNotFoundError as e:
        st.error(f"Данные не найдены: {e}")
        st.info("Запустите `python3 scripts/fed_transmission.py` для генерации данных")
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")


# =============================================================================
# REGIME INDICATOR WIDGET
# =============================================================================


def render_regime_indicator(df):
    """Render regime indicator widget in sidebar."""
    try:
        from sirena.models.regime_detector import detect_regime, MacroRegime
        from sirena.exog.ki_trajectory import get_regime_history

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
    st.caption(f"ΔRuonia (3м): {regime_info['diagnostics']['ruonia_change']:+.1f} п.п.")
    st.caption(f"ΔИнфляция YoY: {regime_info['diagnostics']['yoy_change']:+.1f} п.п.")

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

        fig_timeline.update_layout(
            title="История режимов (последние 24 месяца)",
            xaxis_title="Дата",
            yaxis_showgrid=False,
            yaxis_zeroline=False,
            yaxis_showticklabels=False,
            template="plotly_white",
            height=250,
            hovermode="x unified",
        )

        st.plotly_chart(fig_timeline, use_container_width=True)
