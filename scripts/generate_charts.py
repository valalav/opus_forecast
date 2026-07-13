#!/usr/bin/env python3
"""
ГЕНЕРАТОР ГРАФИКОВ ИЗ БЭКТЕСТОВ
===============================

Читает уже рассчитанные CSV и генерирует интерактивные HTML графики.
Автоматически подхватывает ВСЕ модели из CSV — не нужно ничего менять при добавлении.

Запуск:
    python3 scripts/generate_charts.py           # Все графики
    python3 scripts/generate_charts.py --h1      # Только h=1
    python3 scripts/generate_charts.py --rank    # Ранжирование
    python3 scripts/generate_charts.py --open    # Открыть в браузере

Результат: assets/charts/*.html (интерактивные графики)

Автор: Claude Code
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Plotly для интерактивных графиков
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"Установите недостающие библиотеки: pip install {str(e).split()[-1]}")
    sys.exit(1)


# Директории (нужны для функций загрузки)
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "archive" / "results"
CHARTS_DIR = PROJECT_ROOT / "assets" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# Функции загрузки данных (нужны для nav_template)
def load_predictions(horizon: int) -> pd.DataFrame:
    """Загрузить прогнозы бэктеста"""
    csv_path = RESULTS_DIR / f"backtest_h{horizon}_predictions.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_metrics(horizon: int) -> pd.DataFrame:
    """Загрузить метрики бэктеста"""
    csv_path = RESULTS_DIR / f"backtest_h{horizon}_metrics.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


# HTML шаблон с навигацией (динамический — генерируется функцией)
def get_nav_template() -> str:
    """Генерирует навигацию с актуальными метриками MAE"""
    # Загрузить метрики для всех горизонтов
    mae_info = {}
    for h in [1, 2, 12]:
        df = load_metrics(h)
        if df is not None and len(df) > 0:
            best = df.sort_values("MAE").iloc[0]
            mae_info[h] = {"model": best["Model"], "mae": best["MAE"]}

    # Формируем строки для навигации
    h1_text = (
        f"h=1 <span style='color:#27ae60;font-size:11px;'>({mae_info[1]['model']} {mae_info[1]['mae']:.3f})</span>"
        if 1 in mae_info
        else "h=1"
    )
    h2_text = (
        f"h=2 <span style='color:#27ae60;font-size:11px;'>({mae_info[2]['model']} {mae_info[2]['mae']:.3f})</span>"
        if 2 in mae_info
        else "h=2"
    )
    h12_text = (
        f"h=12 <span style='color:#27ae60;font-size:11px;'>({mae_info[12]['model']} {mae_info[12]['mae']:.3f})</span>"
        if 12 in mae_info
        else "h=12"
    )

    return f"""
<div style="background:#f8f9fa;padding:8px 15px;border-bottom:1px solid #ddd;font-family:Arial,sans-serif;font-size:13px;position:fixed;top:0;left:0;right:0;z-index:1000;display:flex;gap:15px;align-items:center;flex-wrap:wrap;">
  <b style="color:#333;">СИРЕНА-КБР</b>
  <span style="color:#999;">|</span>
  <a href="forecasts.html" style="color:#e67e22;text-decoration:none;font-weight:bold;">Прогноз</a>
  <span style="color:#999;">|</span>
  <a href="backtest_h1_predictions.html" style="color:#1f77b4;text-decoration:none;">{h1_text}</a>
  <a href="backtest_h2_predictions.html" style="color:#1f77b4;text-decoration:none;">{h2_text}</a>
  <a href="backtest_h12_predictions.html" style="color:#1f77b4;text-decoration:none;">{h12_text}</a>
  <span style="color:#999;">|</span>
  <a href="backtest_h1_errors.html" style="color:#e74c3c;text-decoration:none;">Ошибки</a>
  <span style="color:#999;">|</span>
  <a href="ranking_heatmap.html" style="color:#27ae60;text-decoration:none;">Ранги</a>
  <a href="metrics_comparison.html" style="color:#27ae60;text-decoration:none;">MAE</a>
  <a href="last_month_errors.html" style="color:#9b59b6;text-decoration:none;">Последний</a>
  <a href="index.html" style="color:#999;text-decoration:none;margin-left:auto;">Главная</a>
</div>
<div style="height:45px;"></div>
"""


# Цвета для моделей (автоматически генерируются если не заданы)
MODEL_COLORS = {
    "Ridge": "#1f77b4",
    "Ridge_Ext": "#aec7e8",
    "Rolling_Ridge": "#d62728",
    "Ridge_Macro": "#2ecc71",
    "Ridge_Shock": "#98df8a",
    "Bayes_Ridge": "#ff7f0e",
    "ElasticNet": "#ffbb78",
    "Huber": "#2ca02c",
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
}


def get_color(model: str) -> str:
    """Получить цвет для модели (генерирует если не задан)"""
    if model in MODEL_COLORS:
        return MODEL_COLORS[model]
    # Генерируем цвет по хэшу имени
    hash_val = hash(model) % 360
    return f"hsl({hash_val}, 70%, 50%)"


def save_with_nav(fig, output_path: str):
    """Сохранить график с навигационной панелью"""
    html = fig.to_html(full_html=True, include_plotlyjs=True)
    # Вставить навигацию после <body>
    nav_html = get_nav_template()
    html = html.replace("<body>", f"<body>{nav_html}")
    Path(output_path).write_text(html)
    return output_path


def get_models(df: pd.DataFrame) -> list:
    """Получить список моделей из DataFrame (исключая Date и Actual)"""
    exclude = ["Date", "Actual", "date", "actual"]
    return [col for col in df.columns if col not in exclude]


def chart_backtest_predictions(horizon: int) -> str:
    """График прогнозов vs факт для одного горизонта"""
    df = load_predictions(horizon)
    if df is None:
        return None

    models = get_models(df)

    fig = go.Figure()

    # Факт — толстая черная линия
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Actual"],
            name="Факт",
            line=dict(color="black", width=3),
            mode="lines+markers",
        )
    )

    # Прогнозы моделей
    for model in models:
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df[model],
                name=model,
                line=dict(color=get_color(model), width=1.5),
                mode="lines",
                visible="legendonly",  # Скрыты по умолчанию, кликнуть чтобы показать
            )
        )

    # KPI коридор ±0.5
    fig.add_hrect(
        y0=df["Actual"].mean() - 0.5,
        y1=df["Actual"].mean() + 0.5,
        fillcolor="green",
        opacity=0.1,
        line_width=0,
        annotation_text="KPI ±0.5",
    )

    # Компактные кнопки справа внизу
    n_traces = len(models) + 1
    fig.update_layout(
        title=f"Бэктест h={horizon}: Прогнозы vs Факт",
        xaxis_title="Дата",
        yaxis_title="MoM %",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600,
        margin=dict(t=80),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=1,
                y=-0.15,
                xanchor="right",
                showactive=True,
                bgcolor="#f0f0f0",
                borderwidth=0,
                buttons=[
                    dict(label="Все", method="restyle", args=[{"visible": True}]),
                    dict(
                        label="Скрыть",
                        method="restyle",
                        args=[{"visible": "legendonly"}, list(range(1, n_traces))],
                    ),
                ],
            )
        ],
    )

    output_path = CHARTS_DIR / f"backtest_h{horizon}_predictions.html"
    return save_with_nav(fig, str(output_path))


def chart_backtest_errors(horizon: int) -> str:
    """График ошибок прогнозов"""
    df = load_predictions(horizon)
    if df is None:
        return None

    models = get_models(df)
    default_visible = ["Ridge", "Rolling_Ridge", "Micro", "Subcomp", "Ridge_Macro"]

    fig = go.Figure()

    for model in models:
        errors = df[model] - df["Actual"]
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=errors,
                name=model,
                line=dict(color=get_color(model), width=1.5),
                mode="lines",
                visible=True if model in default_visible else "legendonly",
            )
        )

    # KPI границы
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="KPI +0.5")
    fig.add_hline(
        y=-0.5, line_dash="dash", line_color="red", annotation_text="KPI -0.5"
    )
    fig.add_hline(y=0, line_color="black", line_width=1)

    # Компактные кнопки справа внизу
    fig.update_layout(
        title=f"Бэктест h={horizon}: Ошибки (Прогноз - Факт)",
        xaxis_title="Дата",
        yaxis_title="Ошибка (п.п.)",
        hovermode="x unified",
        height=500,
        margin=dict(t=80),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=1,
                y=-0.15,
                xanchor="right",
                showactive=True,
                bgcolor="#f0f0f0",
                borderwidth=0,
                buttons=[
                    dict(label="Все", method="restyle", args=[{"visible": True}]),
                    dict(
                        label="Скрыть",
                        method="restyle",
                        args=[{"visible": "legendonly"}],
                    ),
                ],
            )
        ],
    )

    output_path = CHARTS_DIR / f"backtest_h{horizon}_errors.html"
    return save_with_nav(fig, str(output_path))


def chart_metrics_comparison() -> str:
    """Сравнение MAE по всем горизонтам"""
    metrics = {}
    for h in [1, 2, 12]:
        df = load_metrics(h)
        if df is not None:
            metrics[h] = df.set_index("Model")["MAE"].to_dict()

    if not metrics:
        return None

    # Собрать все модели
    all_models = set()
    for h_metrics in metrics.values():
        all_models.update(h_metrics.keys())
    all_models = sorted(all_models)

    fig = go.Figure()

    for h in [1, 2, 12]:
        if h in metrics:
            values = [metrics[h].get(m, np.nan) for m in all_models]
            fig.add_trace(
                go.Bar(
                    name=f"h={h}",
                    x=all_models,
                    y=values,
                    text=[f"{v:.3f}" if not np.isnan(v) else "" for v in values],
                    textposition="auto",
                )
            )

    fig.update_layout(
        title="MAE по моделям и горизонтам",
        xaxis_title="Модель",
        yaxis_title="MAE",
        barmode="group",
        height=600,
        margin=dict(t=80),
        xaxis_tickangle=-45,
    )

    output_path = CHARTS_DIR / "metrics_comparison.html"
    return save_with_nav(fig, str(output_path))


def chart_ranking_heatmap() -> str:
    """Тепловая карта рангов моделей"""
    ranks = {}

    for h in [1, 2, 12]:
        df = load_metrics(h)
        if df is not None:
            df_sorted = df.sort_values("MAE").reset_index(drop=True)
            df_sorted["Rank"] = range(1, len(df_sorted) + 1)
            for _, row in df_sorted.iterrows():
                model = row["Model"]
                if model not in ranks:
                    ranks[model] = {}
                ranks[model][f"h={h}"] = row["Rank"]

    if not ranks:
        return None

    # Вычислить средний ранг
    for model in ranks:
        model_ranks = list(ranks[model].values())
        ranks[model]["Avg"] = np.mean(model_ranks)

    # Сортировать по среднему рангу
    sorted_models = sorted(ranks.keys(), key=lambda m: ranks[m].get("Avg", 999))

    # Создать матрицу
    horizons = ["h=1", "h=2", "h=12"]
    z_data = []
    for model in sorted_models:
        row = [ranks[model].get(h, np.nan) for h in horizons]
        z_data.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=horizons,
            y=sorted_models,
            colorscale="RdYlGn_r",  # Зеленый = хорошо (низкий ранг)
            text=[
                [f"#{int(v)}" if not np.isnan(v) else "" for v in row] for row in z_data
            ],
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate="Модель: %{y}<br>Горизонт: %{x}<br>Ранг: %{z}<extra></extra>",
        )
    )

    # Добавить средний ранг
    avg_ranks = [f"{ranks[m]['Avg']:.1f}" for m in sorted_models]
    fig.add_trace(
        go.Scatter(
            x=["Avg"] * len(sorted_models),
            y=sorted_models,
            mode="text",
            text=avg_ranks,
            textposition="middle center",
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Ранжирование моделей (зеленый = лучше)",
        xaxis_title="Горизонт",
        yaxis_title="Модель",
        height=max(400, len(sorted_models) * 25),
        margin=dict(t=80),
        yaxis=dict(autorange="reversed"),
    )

    output_path = CHARTS_DIR / "ranking_heatmap.html"
    return save_with_nav(fig, str(output_path))


def chart_model_detail(model: str) -> str:
    """Детальный график для одной модели по всем горизонтам"""
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=["h=1 (1 месяц)", "h=2 (2 месяца)", "h=12 (12 месяцев)"],
        shared_xaxes=True,
        vertical_spacing=0.08,
    )

    for i, h in enumerate([1, 2, 12], 1):
        df = load_predictions(h)
        if df is None or model not in df.columns:
            continue

        # Факт
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Actual"],
                name="Факт" if i == 1 else None,
                line=dict(color="black", width=2),
                showlegend=(i == 1),
            ),
            row=i,
            col=1,
        )

        # Прогноз модели
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df[model],
                name=model if i == 1 else None,
                line=dict(color=get_color(model), width=2),
                showlegend=(i == 1),
            ),
            row=i,
            col=1,
        )

        # Ошибки как заливка
        errors = df[model] - df["Actual"]
        colors = [
            "rgba(255,0,0,0.3)" if e > 0.5 or e < -0.5 else "rgba(0,255,0,0.2)"
            for e in errors
        ]

    fig.update_layout(
        title=f"Модель {model}: прогнозы на всех горизонтах",
        height=800,
        margin=dict(t=80),
        hovermode="x unified",
    )

    output_path = CHARTS_DIR / f"model_{model.lower()}_detail.html"
    return save_with_nav(fig, str(output_path))


def chart_last_month_errors() -> str:
    """Ошибки всех моделей за последний месяц"""
    df = load_predictions(1)
    if df is None:
        return None

    last_row = df.iloc[-1]
    actual = last_row["Actual"]
    date = last_row["Date"]

    models = get_models(df)
    errors = [(m, last_row[m] - actual) for m in models if pd.notna(last_row[m])]
    errors.sort(key=lambda x: abs(x[1]))

    model_names = [e[0] for e in errors]
    error_values = [e[1] for e in errors]
    colors = ["green" if abs(e) <= 0.5 else "red" for e in error_values]

    fig = go.Figure(
        go.Bar(
            x=error_values,
            y=model_names,
            orientation="h",
            marker_color=colors,
            text=[f"{e:+.3f}" for e in error_values],
            textposition="outside",
        )
    )

    fig.add_vline(x=0.5, line_dash="dash", line_color="red")
    fig.add_vline(x=-0.5, line_dash="dash", line_color="red")
    fig.add_vline(x=0, line_color="black")

    fig.update_layout(
        title=f"Ошибки h=1 за {date.strftime('%Y-%m')} (Факт: {actual:.3f})",
        xaxis_title="Ошибка (п.п.)",
        yaxis_title="Модель",
        height=max(400, len(models) * 25),
        margin=dict(t=80),
    )

    output_path = CHARTS_DIR / "last_month_errors.html"
    return save_with_nav(fig, str(output_path))


def chart_rolling_correlation() -> str:
    """Rolling 12-month correlation heatmap between inflation and macro variables"""
    csv_path = PROJECT_ROOT / "data" / "inflation_data.csv"
    if not csv_path.exists():
        print("  ⚠ inflation_data.csv not found, skipping rolling correlation")
        return None

    df = pd.read_csv(csv_path, sep=";", decimal=",", encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
    df = df.sort_values("Date")

    # Convert numeric columns
    numeric_cols = ["mom", "usd_nom_i", "Ki_i", "Ruonia", "Nonprod", "Prod", "Serv"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "."), errors="coerce"
            )

    # Variables to correlate with mom (inflation)
    variables = ["mom", "usd_nom_i", "Ki_i", "Ruonia", "Nonprod", "Prod", "Serv"]
    df_with_date = df[["Date"] + variables].dropna()

    if len(df_with_date) < 12:
        print("  ⚠ Not enough data for 12-month rolling correlation")
        return None

    # Calculate rolling 12-month correlations
    window = 12
    correlations = {}
    dates = []

    for i in range(len(df_with_date) - window + 1):
        window_data = df_with_date.iloc[i : i + window]
        date = df_with_date.iloc[i + window - 1]["Date"]
        dates.append(date)

        # Correlation of mom with other variables
        for var in ["usd_nom_i", "Ki_i", "Ruonia", "Nonprod", "Prod", "Serv"]:
            if var not in correlations:
                correlations[var] = []
            corr = window_data["mom"].corr(window_data[var])
            correlations[var].append(corr)

    # Create DataFrame for heatmap
    corr_df = pd.DataFrame(correlations, index=dates)
    corr_df.index.name = "Date"

    # Create heatmap using seaborn
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        corr_df.T,
        cmap="RdBu_r",
        center=0,
        cbar_kws={"label": "Correlation"},
        xticklabels=False,
        vmin=-1,
        vmax=1,
    )

    plt.title("12-Month Rolling Correlation with Inflation (mom)")
    plt.xlabel("Time (12-month windows)")
    plt.ylabel("Variable")
    plt.tight_layout()

    # Save as PNG
    output_path = CHARTS_DIR / "rolling_corr.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  ✓ rolling_corr.png saved ({len(dates)} windows)")
    return str(output_path)


def chart_forecasts() -> str:
    """График прогнозов на будущее"""
    csv_path = RESULTS_DIR / "forecasts_current.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Также загрузим историю для контекста
    hist_path = RESULTS_DIR / "backtest_h1_predictions.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        hist["Date"] = pd.to_datetime(hist["Date"])
        last_actual = hist[["Date", "Actual"]].tail(6)
    else:
        last_actual = None

    models = [col for col in df.columns if col != "Date"]

    fig = go.Figure()

    # История (если есть)
    if last_actual is not None:
        fig.add_trace(
            go.Scatter(
                x=last_actual["Date"],
                y=last_actual["Actual"],
                name="Факт (история)",
                line=dict(color="black", width=3),
                mode="lines+markers",
            )
        )

    # Прогнозы моделей
    for model in models:
        if df[model].isna().all():
            continue
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df[model],
                name=model,
                line=dict(color=get_color(model), width=2),
                mode="lines+markers",
                visible=True
                if model in ["Ridge_Macro", "Micro", "Subcomp"]
                else "legendonly",
            )
        )

    # Среднее по моделям
    model_cols = [c for c in models if not df[c].isna().all()]
    if model_cols:
        df["Ensemble_Mean"] = df[model_cols].mean(axis=1)
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Ensemble_Mean"],
                name="Среднее",
                line=dict(color="red", width=3, dash="dash"),
                mode="lines",
            )
        )

    n_traces = len([m for m in models if not df[m].isna().all()]) + 2
    fig.update_layout(
        title="Прогноз на 12 месяцев вперёд",
        xaxis_title="Дата",
        yaxis_title="MoM %",
        hovermode="x unified",
        height=600,
        margin=dict(t=80),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=1,
                y=-0.15,
                xanchor="right",
                showactive=True,
                bgcolor="#f0f0f0",
                borderwidth=0,
                buttons=[
                    dict(label="Все", method="restyle", args=[{"visible": True}]),
                    dict(
                        label="Скрыть",
                        method="restyle",
                        args=[{"visible": "legendonly"}],
                    ),
                ],
            )
        ],
    )

    output_path = CHARTS_DIR / "forecasts.html"
    return save_with_nav(fig, str(output_path))


def generate_all_charts(open_browser: bool = False):
    """Сгенерировать все графики"""
    print("Генерация графиков...")
    charts = []

    # Бэктесты по горизонтам
    for h in [1, 2, 12]:
        path = chart_backtest_predictions(h)
        if path:
            print(f"  ✓ backtest_h{h}_predictions.html")
            charts.append(path)

        path = chart_backtest_errors(h)
        if path:
            print(f"  ✓ backtest_h{h}_errors.html")
            charts.append(path)

    # Сравнение метрик
    path = chart_metrics_comparison()
    if path:
        print(f"  ✓ metrics_comparison.html")
        charts.append(path)

    # Тепловая карта рангов
    path = chart_ranking_heatmap()
    if path:
        print(f"  ✓ ranking_heatmap.html")
        charts.append(path)

    # Rolling correlations
    path = chart_rolling_correlation()
    if path:
        print(f"  ✓ rolling_corr.png")
        charts.append(path)

    # Ошибки последнего месяца
    path = chart_last_month_errors()
    if path:
        print(f"  ✓ last_month_errors.html")
        charts.append(path)

    # Generate Forecast Table (New)
    print("\n[8/8] Generating Forecast Table...")
    try:
        # Add scripts dir to path if needed or import directly if in same dir
        sys.path.append(str(PROJECT_ROOT / "scripts"))
        import generate_forecast_table

        data = generate_forecast_table.load_forecasts("data/precomputed_forecasts.json")
        forecast_table_path = CHARTS_DIR / "forecast_table.html"
        generate_forecast_table.generate_html_table(data, str(forecast_table_path))
        print(f"  ✓ forecast_table.html")
        charts.append(str(forecast_table_path))
    except Exception as e:
        print(f"Error generating table: {e}")

    # Прогнозы
    path = chart_forecasts()
    if path:
        print(f"  ✓ forecasts.html")
        charts.append(path)

    print(f"\nСоздано {len(charts)} графиков в {CHARTS_DIR}/")

    # Создать index.html
    create_index(charts)

    if open_browser and charts:
        import webbrowser

        index_path = CHARTS_DIR / "index.html"
        webbrowser.open(f"file://{index_path}")


def _make_summary_row(h: int, mae_info: dict) -> str:
    """Создать строку таблицы с метриками"""
    if h not in mae_info:
        return f"<tr><td>h={h}</td><td>N/A</td><td>N/A</td><td>N/A</td></tr>"

    info = mae_info[h]
    horizon_names = {1: "h=1 (1 мес)", 2: "h=2 (2 мес)", 12: "h=12 (год)"}
    h_name = horizon_names.get(h, f"h={h}")

    micro_cell = f"{info['micro_mae']:.3f}" if info.get("micro_mae") else "N/A"

    return f'<tr><td>{h_name}</td><td>{info["best_model"]}</td><td class="mae">{info["best_mae"]:.3f}</td><td>{micro_cell}</td></tr>'


def create_index(charts: list):
    """Создать index.html со ссылками на все графики и метриками MAE"""
    from datetime import datetime

    # Загрузить метрики для всех горизонтов
    mae_info = {}
    for h in [1, 2, 12]:
        df = load_metrics(h)
        if df is not None and len(df) > 0:
            best = df.sort_values("MAE").iloc[0]
            micro = df[df["Model"] == "Micro"]
            micro_mae = micro["MAE"].values[0] if len(micro) > 0 else None
            mae_info[h] = {
                "best_model": best["Model"],
                "best_mae": best["MAE"],
                "micro_mae": micro_mae,
            }

    # Формируем описания для карточек
    h1_desc = (
        f"Лучшая: {mae_info[1]['best_model']} (MAE {mae_info[1]['best_mae']:.3f})"
        if 1 in mae_info
        else "Главный КПЭ"
    )
    h2_desc = (
        f"Лучшая: {mae_info[2]['best_model']} (MAE {mae_info[2]['best_mae']:.3f})"
        if 2 in mae_info
        else "Среднесрочный"
    )
    h12_desc = (
        f"Лучшая: {mae_info[12]['best_model']} (MAE {mae_info[12]['best_mae']:.3f})"
        if 12 in mae_info
        else "Годовая траектория"
    )

    # Micro сравнение
    micro_text = ""
    if 1 in mae_info and mae_info[1]["micro_mae"]:
        micro_text = f"<p style='color:#e74c3c;'>Micro ARIMA: MAE h=1 {mae_info[1]['micro_mae']:.3f}"
        if 2 in mae_info and mae_info[2]["micro_mae"]:
            micro_text += f", h=2 {mae_info[2]['micro_mae']:.3f}"
        micro_text += "</p>"

    # Навигация с MAE
    nav_h1 = f"h=1 ({mae_info[1]['best_mae']:.2f})" if 1 in mae_info else "h=1"
    nav_h2 = f"h=2 ({mae_info[2]['best_mae']:.2f})" if 2 in mae_info else "h=2"
    nav_h12 = f"h=12 ({mae_info[12]['best_mae']:.2f})" if 12 in mae_info else "h=12"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>СИРЕНА-КБР: Графики</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; background: #f5f5f5; }}
        .nav {{ background:#f8f9fa;padding:8px 15px;border-bottom:1px solid #ddd;font-size:13px;display:flex;gap:15px;align-items:center;flex-wrap:wrap; }}
        .nav b {{ color:#333; }}
        .nav a {{ color:#1f77b4;text-decoration:none; }}
        .nav span {{ color:#999; }}
        .content {{ padding: 20px; }}
        h1 {{ color: #333; margin-top: 0; }}
        .summary {{ background: #e8f5e9; border-radius: 8px; padding: 15px; margin-bottom: 20px; }}
        .summary h2 {{ margin: 0 0 10px 0; font-size: 16px; color: #2e7d32; }}
        .summary table {{ border-collapse: collapse; width: 100%; }}
        .summary th, .summary td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #c8e6c9; }}
        .summary th {{ background: #c8e6c9; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 15px; }}
        .card {{ background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .card a {{ text-decoration: none; color: #1f77b4; font-weight: bold; font-size: 14px; }}
        .card a:hover {{ text-decoration: underline; }}
        .card p {{ color: #666; font-size: 12px; margin: 5px 0 0 0; }}
        .card .mae {{ color: #27ae60; font-weight: bold; }}
        .section {{ margin-bottom: 25px; }}
        .section-title {{ font-size: 14px; color: #666; margin-bottom: 10px; text-transform: uppercase; }}
        .timestamp {{ color: #999; font-size: 11px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="nav">
        <b>СИРЕНА-КБР</b>
        <span>|</span>
        <a href="backtest_h1_predictions.html">{nav_h1}</a>
        <a href="backtest_h2_predictions.html">{nav_h2}</a>
        <a href="backtest_h12_predictions.html">{nav_h12}</a>
        <span>|</span>
        <a href="ranking_heatmap.html">Ранги</a>
        <a href="metrics_comparison.html">MAE</a>
        <a href="last_month_errors.html">Последний месяц</a>
    </div>
    <div class="content">
        <h1>Графики СИРЕНА-КБР</h1>

        <div class="summary">
            <h2>📊 Сводка MAE по горизонтам</h2>
            <table>
                <tr><th>Горизонт</th><th>Лучшая модель</th><th>MAE</th><th>Micro ARIMA</th></tr>
                {_make_summary_row(1, mae_info)}
                {_make_summary_row(2, mae_info)}
                {_make_summary_row(12, mae_info)}
            </table>
        </div>

        <div class="section">
            <div class="section-title">Прогнозы</div>
            <div class="grid">
                <div class="card"><a href="forecasts.html" style="color:#e67e22;">Прогноз 12 мес</a><p>Все модели вперёд</p></div>
                <div class="card" style="background:#e8f8f5;"><a href="forecast_table.html" style="color:#16a085;">📋 Детальная таблица</a><p>Все модели + Nowcast</p></div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Бэктесты: Прогнозы vs Факт</div>
            <div class="grid">
                <div class="card"><a href="backtest_h1_predictions.html">h=1 (1 месяц)</a><p>{h1_desc}</p></div>
                <div class="card"><a href="backtest_h2_predictions.html">h=2 (2 месяца)</a><p>{h2_desc}</p></div>
                <div class="card"><a href="backtest_h12_predictions.html">h=12 (год)</a><p>{h12_desc}</p></div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Ошибки</div>
            <div class="grid">
                <div class="card"><a href="backtest_h1_errors.html">Ошибки h=1</a><p>Отклонения от факта</p></div>
                <div class="card"><a href="backtest_h2_errors.html">Ошибки h=2</a><p>Отклонения от факта</p></div>
                <div class="card"><a href="backtest_h12_errors.html">Ошибки h=12</a><p>Отклонения от факта</p></div>
            </div>
        </div>

        <div class="section">
            <div class="section-title">Аналитика</div>
            <div class="grid">
                <div class="card"><a href="ranking_heatmap.html">Ранжирование</a><p>Тепловая карта позиций</p></div>
                <div class="card"><a href="metrics_comparison.html">Сравнение MAE</a><p>По всем моделям</p></div>
                <div class="card"><a href="last_month_errors.html">Последний месяц</a><p>Ошибки всех моделей</p></div>
            </div>
        </div>

        <p class="timestamp">Обновлено: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
    </div>
</body>
</html>
"""

    index_path = CHARTS_DIR / "index.html"
    index_path.write_text(html)
    print(f"  ✓ index.html (главная страница)")


if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)

    args = sys.argv[1:]
    open_browser = "--open" in args

    if "--h1" in args:
        chart_backtest_predictions(1)
        chart_backtest_errors(1)
    elif "--h2" in args:
        chart_backtest_predictions(2)
        chart_backtest_errors(2)
    elif "--h12" in args:
        chart_backtest_predictions(12)
        chart_backtest_errors(12)
    elif "--rank" in args:
        chart_ranking_heatmap()
    elif "--metrics" in args:
        chart_metrics_comparison()
    else:
        generate_all_charts(open_browser)
