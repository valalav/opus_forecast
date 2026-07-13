"""
Feature Importance Tab - Page 6
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import dashboard_utils


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

            # Prepare data
            target_col = "Все товары и услуги"

            if target_col not in df.columns:
                st.error("Целевая колонка не найдена в данных")
                return

            # Simple features
            features = [
                "Продовольственные товары",
                "Непродовольственные товары",
                "Услуги",
            ]
            available_features = [f for f in features if f in df.columns]

            if len(available_features) < 2:
                st.error("Недостаточно признаков для анализа")
                return

            X = df[available_features].dropna()
            y = df[target_col].loc[X.index]

            if len(X) < 20:
                st.error("Недостаточно данных для обучения")
                return

            # Train model
            if selected_model == "Ridge":
                model = Ridge(alpha=1.0)
            elif selected_model == "Huber":
                model = HuberRegressor(epsilon=1.35)
            else:  # NGBoost
                try:
                    from ngboost import NGBRegressor

                    model = NGBRegressor()
                except ImportError:
                    st.error("NGBoost не установлен. Установите: pip install ngboost")
                    return

            model.fit(X, y)

            # Calculate feature importance
            if selected_model in ["Ridge", "Huber"]:
                importances = np.abs(model.coef_)
                feature_names = available_features
            else:
                # For NGBoost, use permutation importance
                result = permutation_importance(
                    model, X, y, n_repeats=10, random_state=42, n_jobs=-1
                )
                importances = result.importances_mean
                feature_names = available_features

            # Display
            st.markdown("### 📊 Важность признаков")

            col_imp1, col_imp2 = st.columns([1, 2])

            with col_imp1:
                # Bar chart
                fig = go.Figure(
                    data=[
                        go.Bar(
                            x=importances,
                            y=feature_names,
                            orientation="h",
                            marker=dict(color="#2563eb"),
                        )
                    ]
                )
                fig.update_layout(
                    title="Важность признаков",
                    xaxis_title="Важность",
                    yaxis_title="Признак",
                    height=300,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_imp2:
                # Table
                importance_df = pd.DataFrame(
                    {"Признак": feature_names, "Важность": importances}
                ).sort_values("Важность", ascending=False)

                importance_df["Вклад"] = (
                    importance_df["Важность"] / importance_df["Важность"].sum() * 100
                ).round(1).astype(str) + "%"

                st.dataframe(
                    importance_df,
                    use_container_width=True,
                    hide_index=True,
                )

            # Interpretation
            st.markdown("---")
            st.markdown("### 💡 Интерпретация")

            top_feature = importance_df.iloc[0]["Признак"]
            top_importance = importance_df.iloc[0]["Вклад"]

            st.info(f"""
            **Главный драйвер:** {top_feature} ({top_importance} вклада в модель)

            Признаки с более высокой важностью имеют большее влияние на прогноз инфляции.
            """)

            # Feature analysis
            st.markdown("### 🔍 Анализ по признакам")

            for i, row in importance_df.iterrows():
                feat = row["Признак"]
                imp = row["Важность"]

                if feat == "Продовольственные товары":
                    st.markdown(f"🥬 **{feat}**: Важность = {imp:.4f}")
                    st.caption(
                        "Цены на продукты питания — основной компонент ИПЦ (~39% веса)"
                    )
                elif feat == "Непродовольственные товары":
                    st.markdown(f"🛍️ **{feat}**: Важность = {imp:.4f}")
                    st.caption(
                        "Товары длительного пользования, одежда, обувь (~37% веса)"
                    )
                elif feat == "Услуги":
                    st.markdown(f"🏛️ **{feat}**: Важность = {imp:.4f}")
                    st.caption("ЖКХ, транспорт, медицина, образование (~24% веса)")

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
    render_feature_importance_tab(df)
