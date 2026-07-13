#!/usr/bin/env python3
"""
INTERACTIVE CORRELATION MATRIX VISUALIZER
=======================================

Creates an interactive Plotly correlation heatmap for inflation and macro variables.

Usage:
    python3 scripts/plot_interactive_corr.py
    python3 scripts/plot_interactive_corr.py --features mom,Prod,Nonprod,Serv

Output: assets/charts/corr_matrix_interactive.html

Author: Ralph Universal (Worker Agent)
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    import plotly.io as pio
except ImportError as e:
    print(f"Error: Install plotly: pip install plotly")
    sys.exit(1)

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
CHARTS_DIR = PROJECT_ROOT / "assets" / "charts"
DATA_DIR = PROJECT_ROOT / "data"

CHARTS_DIR.mkdir(parents=True, exist_ok=True)


# Default features to include in correlation matrix
DEFAULT_FEATURES = [
    "mom",
    "Prod",
    "Nonprod",
    "Serv",
    "usd_nom_i",
    "Ki_i",
    "Ruonia",
    "Ki",
    "fl_potrb_zad",
    "all_real",
]

# Feature names for display (Russian)
FEATURE_NAMES = {
    "mom": "Инфляция (ИПЦ)",
    "Prod": "Продовольствие",
    "Nonprod": "Непродовольствие",
    "Serv": "Услуги",
    "usd_nom_i": "Курс USD",
    "Ki_i": "Ключевая ставка",
    "Ruonia": "RUONIA",
    "Ki": "Ki (ставка)",
    "fl_potrb_zad": "Отгрузка промышленности",
    "all_real": "Реальные располагаемые доходы",
}


def load_inflation_data(features: list = None) -> pd.DataFrame:
    """
    Load inflation data from CSV.

    Args:
        features: List of features to include. If None, use all numeric.

    Returns:
        DataFrame with selected features
    """
    csv_path = DATA_DIR / "inflation_data.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return None

    # Load with semicolon separator
    df = pd.read_csv(csv_path, sep=";", decimal=",")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date")

    # Select features
    if features is None:
        # Use all numeric columns except Date
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        features = [col for col in numeric_cols if col in df.columns]

    # Convert to numeric (handle commas as decimals)
    for col in features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter to available features
    available_features = [col for col in features if col in df.columns]
    if not available_features:
        print(f"Error: No valid features found")
        return None

    df_subset = df[available_features].copy()

    return df_subset


def calculate_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Pearson correlation matrix.

    Args:
        df: DataFrame with numeric features

    Returns:
        Correlation matrix (N x N)
    """
    corr_matrix = df.corr(method="pearson")
    return corr_matrix


def create_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    """
    Create interactive Plotly heatmap.

    Args:
        corr_matrix: Correlation matrix DataFrame

    Returns:
        Plotly Figure
    """
    # Get feature names for display
    feature_labels = [FEATURE_NAMES.get(col, col) for col in corr_matrix.columns]

    # Create heatmap with custom hover template
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=feature_labels,
            y=feature_labels,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(
                title="Корреляция",
                tickmode="array",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1", "-0.5", "0", "0.5", "1"],
            ),
            hovertemplate=(
                "<b>%{x}</b> vs <b>%{y}</b><br>"
                "Корреляция: <b>%{z:.3f}</b><br>"
                "<extra></extra>"
            ),
            text=corr_matrix.values.round(3),
            texttemplate="%{text}",
            textfont={"size": 10},
        )
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text="Корреляционная матрица макроэкономических показателей",
            x=0.5,
            xanchor="center",
            font=dict(size=20),
        ),
        xaxis=dict(title="", tickangle=-45, tickfont=dict(size=11)),
        yaxis=dict(title="", tickfont=dict(size=11)),
        width=900,
        height=800,
        margin=dict(l=100, r=50, t=80, b=100),
        hovermode="closest",
    )

    # Add diagonal line (optional, for visual reference)
    n = len(corr_matrix)
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=n - 1,
        y1=n - 1,
        line=dict(color="white", width=2, dash="solid"),
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
      <a href="fan_chart.html" style="color:#1f77b4;text-decoration:none;">Fan Chart</a>
      <span style="color:#999;">|</span>
      <a href="corr_matrix_interactive.html" style="color:#e67e22;text-decoration:none;font-weight:bold;">Корреляции</a>
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
        description="Generate interactive correlation matrix heatmap"
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated list of features (e.g., mom,Prod,Nonprod,Serv)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include all numeric features from inflation_data.csv",
    )

    args = parser.parse_args()

    # Determine features to use
    if args.features:
        features = [f.strip() for f in args.features.split(",")]
    elif args.all:
        features = None  # Will use all numeric
    else:
        features = DEFAULT_FEATURES

    print(f"Loading data...")
    df = load_inflation_data(features)

    if df is None:
        print("Error: Failed to load data")
        sys.exit(1)

    print(f"Features: {', '.join(df.columns.tolist())}")
    print(f"Observations: {len(df)}")

    print("Calculating correlation matrix...")
    corr_matrix = calculate_correlation_matrix(df)

    print("Creating interactive heatmap...")
    fig = create_heatmap(corr_matrix)

    # Save HTML
    output_path = CHARTS_DIR / "corr_matrix_interactive.html"
    save_with_nav(fig, str(output_path))

    print(f"✓ Correlation matrix saved to {output_path}")
    print(f"  - Features: {len(corr_matrix)}")
    print(f"  - Color scale: Red-Blue (Red=Negative, Blue=Positive)")

    # Show some interesting correlations
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    top_corr = upper_triangle.unstack().abs().sort_values(ascending=False)

    print(f"\n📊 Top 5 correlations:")
    for idx, val in top_corr.head(5).items():
        feat1, feat2 = idx
        name1 = FEATURE_NAMES.get(feat1, feat1)
        name2 = FEATURE_NAMES.get(feat2, feat2)
        sign = "+" if corr_matrix.loc[feat1, feat2] >= 0 else "-"
        print(f"  {name1} ↔ {name2}: {sign}{val:.3f}")


if __name__ == "__main__":
    main()
