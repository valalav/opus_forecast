#!/usr/bin/env python3
"""
FAN CHART GENERATOR
===================

Generates fan chart visualization with confidence bands (50%, 80%, 95%).

Usage:
    python3 scripts/plot_fan_chart.py
    python3 scripts/plot_fan_chart.py --model Ridge
    python3 scripts/plot_fan_chart.py --horizon 12

Output: assets/charts/fan_chart.html

Author: Ralph Universal (Worker Agent)
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.io as pio
except ImportError as e:
    print(f"Error: Install plotly: pip install plotly")
    sys.exit(1)

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
CHARTS_DIR = PROJECT_ROOT / "assets" / "charts"
DATA_DIR = PROJECT_ROOT / "data"

CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Z-scores for confidence intervals
Z_SCORES = {
    "50": 0.674,  # 25% quantile
    "80": 1.282,  # 10% quantile
    "95": 1.960,  # 2.5% quantile
}


def load_inflation_data():
    """Load inflation data for context."""
    csv_path = DATA_DIR / "infl_kbr.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path, sep=";")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Get only "Все товары и услуги" and convert MoM to percentage
    df_total = df[df["Товар"] == "Все товары и услуги"].copy()

    # Convert MoM index to percentage (101.49 -> 1.49%)
    df_total["MoM"] = df_total["MoM"] - 100

    # Keep only columns we need
    df_total = df_total[["Date", "MoM"]].dropna()

    return df_total


def estimate_std_from_history(historical_mom: pd.Series, horizon: int) -> np.ndarray:
    """
    Estimate standard deviation from historical volatility.

    Args:
        historical_mom: Historical MoM values
        horizon: Forecast horizon (months)

    Returns:
        Array of std values for each month ahead
    """
    if len(historical_mom) < 12:
        # Default std if not enough data
        return np.array([0.3 * np.sqrt(1 + i * 0.1) for i in range(horizon)])

    # Calculate rolling std (12-month window)
    base_std = historical_mom.rolling(12).std().iloc[-1]

    # Std increases with horizon (uncertainty grows)
    stds = np.array([base_std * np.sqrt(1 + i * 0.1) for i in range(horizon)])

    return stds


def load_precomputed_forecasts(model_name: str, horizon: int = 12) -> dict:
    """
    Load forecasts from precomputed data.

    Args:
        model_name: Model name
        horizon: Forecast horizon

    Returns:
        Dict with 'dates', 'predictions', 'std', 'historical'
    """
    # Load precomputed forecasts
    json_path = DATA_DIR / "precomputed_forecasts.json"
    if not json_path.exists():
        print(f"Error: precomputed_forecasts.json not found")
        return None

    import json

    with open(json_path, "r") as f:
        forecast_data = json.load(f)

    # Check if model exists
    if model_name not in forecast_data.get("forecasts", {}):
        print(f"Error: Model {model_name} not in precomputed forecasts")
        return None

    # Get predictions
    predictions = np.array(forecast_data["forecasts"][model_name][:horizon])

    # Get last data date
    last_date_str = forecast_data.get("last_data_date", "2025-12-01")
    last_date = pd.to_datetime(last_date_str)

    # Generate forecast dates
    forecast_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1), periods=horizon, freq="MS"
    )

    # Load historical data for std estimation
    hist_df = load_inflation_data()
    if hist_df is None:
        print("Error: Could not load historical data")
        return None

    # Estimate std from historical volatility
    std = estimate_std_from_history(hist_df["MoM"], horizon)

    return {
        "dates": forecast_dates,
        "predictions": predictions,
        "std": std,
        "historical": hist_df[["Date", "MoM"]].tail(6),  # Last 6 months for context
    }


def create_fan_chart(data: dict, model_name: str) -> go.Figure:
    """
    Create fan chart with confidence bands.

    Args:
        data: Dict with dates, predictions, std, historical
        model_name: Name of the model

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    # Historical data (last 6 months)
    if data.get("historical") is not None:
        hist = data["historical"]
        fig.add_trace(
            go.Scatter(
                x=hist["Date"],
                y=hist["MoM"],
                name="History",
                line=dict(color="black", width=2),
                mode="lines+markers",
            )
        )

    # Confidence bands (95%, 80%, 50%) - from widest to narrowest
    dates = data["dates"]
    predictions = data["predictions"]
    std = data["std"]

    # 95% CI (widest, lightest)
    ci95_upper = predictions + Z_SCORES["95"] * std
    ci95_lower = predictions - Z_SCORES["95"] * std

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ci95_upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ci95_lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 100, 80, 0.2)",  # Light teal
            name="95% CI",
            hoverinfo="skip",
        )
    )

    # 80% CI
    ci80_upper = predictions + Z_SCORES["80"] * std
    ci80_lower = predictions - Z_SCORES["80"] * std

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ci80_upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ci80_lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 150, 136, 0.4)",  # Medium teal
            name="80% CI",
            hoverinfo="skip",
        )
    )

    # 50% CI (narrowest, darkest)
    ci50_upper = predictions + Z_SCORES["50"] * std
    ci50_lower = predictions - Z_SCORES["50"] * std

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ci50_upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=ci50_lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 200, 180, 0.6)",  # Dark teal
            name="50% CI",
            hoverinfo="skip",
        )
    )

    # Central prediction (mean)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=predictions,
            name="Prediction",
            line=dict(color="black", width=3, dash="solid"),
            mode="lines+markers",
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Fan Chart: {model_name} Forecast (12 months)",
        xaxis_title="Date",
        yaxis_title="MoM Inflation (%)",
        hovermode="x unified",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80, b=60, l=60, r=40),
    )

    return fig


def save_with_nav(fig: go.Figure, output_path: str):
    """Save chart with navigation template."""
    html = fig.to_html(full_html=True, include_plotlyjs=True)

    # Simple nav template
    nav_html = """
    <div style="background:#f8f9fa;padding:8px 15px;border-bottom:1px solid #ddd;font-family:Arial,sans-serif;font-size:13px;position:fixed;top:0;left:0;right:0;z-index:1000;">
      <b>SIRENA-KBR</b>
      <span style="color:#999;">|</span>
      <a href="forecasts.html" style="color:#1f77b4;text-decoration:none;">Прогноз</a>
      <span style="color:#999;">|</span>
      <a href="fan_chart.html" style="color:#e67e22;text-decoration:none;font-weight:bold;">Fan Chart</a>
      <span style="color:#999;">|</span>
      <a href="index.html" style="color:#999;text-decoration:none;">Главная</a>
    </div>
    <div style="height:45px;"></div>
    """

    # Insert nav after <body>
    html = html.replace("<body>", f"<body>{nav_html}")
    Path(output_path).write_text(html)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate Fan Chart with confidence bands"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Ridge",
        help="Model name (Ridge, NGBoost, Huber, etc.)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=12,
        help="Forecast horizon in months (default: 12)",
    )

    args = parser.parse_args()

    print(f"Generating fan chart for {args.model} (h={args.horizon})...")

    # Load precomputed forecasts with estimated CI
    forecast_data = load_precomputed_forecasts(args.model, args.horizon)

    if forecast_data is None:
        print("Error: Failed to load forecasts")
        sys.exit(1)

    # Create fan chart
    fig = create_fan_chart(forecast_data, args.model)

    # Save HTML
    output_path = CHARTS_DIR / "fan_chart.html"
    save_with_nav(fig, str(output_path))

    print(f"✓ Fan chart saved to {output_path}")
    print(f"  - 50% CI: Mean ± {Z_SCORES['50']}σ")
    print(f"  - 80% CI: Mean ± {Z_SCORES['80']}σ")
    print(f"  - 95% CI: Mean ± {Z_SCORES['95']}σ")


if __name__ == "__main__":
    main()
