#!/usr/bin/env python3
"""
Generate Model Performance HTML Report
Creates visualization charts and an HTML report from backtest metrics.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Paths
BASE_DIR = "/home/valalav/_projects/sirena-kbr/edge_lab"
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
CHARTS_DIR = os.path.join(ASSETS_DIR, "charts")
REPORTS_DIR = os.path.join(ASSETS_DIR, "reports")
ARCHIVE_RESULTS_DIR = os.path.join(BASE_DIR, "archive", "results")

# Ensure directories exist
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Set matplotlib style
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def generate_mae_comparison():
    """Generate MAE comparison bar chart."""
    # Load consolidated metrics
    consolidated_path = os.path.join(DATA_DIR, "consolidated_metrics.csv")
    if not os.path.exists(consolidated_path):
        print(f"Warning: {consolidated_path} not found")
        return None

    df = pd.read_csv(consolidated_path)

    if df.empty:
        print("Warning: consolidated_metrics.csv is empty")
        return None

    # Create MAE comparison chart
    fig, ax = plt.subplots(figsize=(12, 6))

    models = df["Model"].values
    maes = [
        df["MAE_h1"].iloc[i] * 0.5
        + df["MAE_h2"].iloc[i] * 0.3
        + df["MAE_h12"].iloc[i] * 0.2
        for i in range(len(df))
    ]

    bars = ax.bar(
        models,
        maes,
        color=["#2ecc71", "#3498db", "#e74c3c", "#9b59b6", "#f39c12"][: len(models)],
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Weighted MAE")
    ax.set_title("Model Performance Comparison (Weighted MAE)")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    output_path = os.path.join(CHARTS_DIR, "mae_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Generated: {output_path}")
    return output_path


def generate_horizon_performance():
    """Generate horizon-wise performance line chart."""
    # Load all backtest metrics
    all_metrics_path = os.path.join(ARCHIVE_RESULTS_DIR, "backtest_all_metrics.csv")

    if not os.path.exists(all_metrics_path):
        print(f"Warning: {all_metrics_path} not found")
        return None

    df = pd.read_csv(all_metrics_path)

    if df.empty:
        print("Warning: backtest_all_metrics.csv is empty")
        return None

    # Group by model and horizon
    fig, ax = plt.subplots(figsize=(12, 6))

    for model in df["model"].unique():
        model_data = df[df["model"] == model].sort_values("horizon")
        ax.plot(
            model_data["horizon"],
            model_data["MAE"],
            marker="o",
            label=model,
            linewidth=2,
        )

    ax.set_xlabel("Forecast Horizon (months)")
    ax.set_ylabel("MAE")
    ax.set_title("MAE by Forecast Horizon")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks([1, 2, 12])

    plt.tight_layout()
    output_path = os.path.join(CHARTS_DIR, "forecast_trajectories.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Generated: {output_path}")
    return output_path


def generate_html_report(mae_chart_path, horizon_chart_path):
    """Generate the HTML report."""
    # Load consolidated metrics
    consolidated_path = os.path.join(DATA_DIR, "consolidated_metrics.csv")
    if os.path.exists(consolidated_path):
        df = pd.read_csv(consolidated_path)
    else:
        df = pd.DataFrame()

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Start building HTML
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Performance Report - Opus Edge Lab</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
        }
        
        h1 {
            font-size: 28px;
            margin-bottom: 10px;
        }
        
        .timestamp {
            font-size: 14px;
            opacity: 0.9;
        }
        
        .content {
            padding: 30px;
        }
        
        .section {
            margin-bottom: 40px;
        }
        
        h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 22px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .charts-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
        }
        
        .chart-wrapper {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .chart-wrapper img {
            width: 100%;
            height: auto;
            border-radius: 6px;
        }
        
        .chart-caption {
            text-align: center;
            margin-top: 10px;
            color: #6c757d;
            font-size: 14px;
        }
        
        .table-container {
            overflow-x: auto;
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        
        th {
            background: #667eea;
            color: white;
            padding: 12px 15px;
            text-align: left;
            font-weight: 600;
        }
        
        td {
            padding: 10px 15px;
            border-bottom: 1px solid #dee2e6;
        }
        
        tr:last-child td {
            border-bottom: none;
        }
        
        tr:nth-child(even) {
            background: rgba(102, 126, 234, 0.05);
        }
        
        tr:hover {
            background: rgba(102, 126, 234, 0.1);
        }
        
        .metric-value {
            font-weight: 600;
            color: #667eea;
        }
        
        .interpretation {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border-left: 4px solid #667eea;
            padding: 20px;
            border-radius: 6px;
            margin-bottom: 20px;
        }
        
        .interpretation h3 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 18px;
        }
        
        .interpretation p {
            color: #495057;
            line-height: 1.6;
        }
        
        footer {
            background: #f8f9fa;
            padding: 20px 30px;
            text-align: center;
            color: #6c757d;
            font-size: 13px;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .badge-h1 { background: #28a745; color: white; }
        .badge-h2 { background: #17a2b8; color: white; }
        .badge-h12 { background: #dc3545; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Model Performance Report</h1>
            <p class="timestamp">Generated: {timestamp}</p>
        </header>
        
        <div class="content">
            <div class="section">
                <div class="interpretation">
                    <h3>Executive Summary</h3>
                    <p>This report presents the comprehensive performance analysis of forecasting models in the Opus Edge Lab system. Metrics include Mean Absolute Error (MAE) across different forecast horizons (h=1, h=2, h=12 months) and a weighted composite score.</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <div class="charts-container">
"""

    # Add MAE comparison chart
    if mae_chart_path:
        html += f"""
                    <div class="chart-wrapper">
                        <img src="{"../" if REPORTS_DIR else ""}charts/mae_comparison.png" alt="MAE Comparison">
                        <p class="chart-caption">Model Performance Comparison (Weighted MAE)</p>
                    </div>
"""

    # Add horizon performance chart
    if horizon_chart_path:
        html += f"""
                    <div class="chart-wrapper">
                        <img src="{"../" if REPORTS_DIR else ""}charts/forecast_trajectories.png" alt="Forecast Trajectories">
                        <p class="chart-caption">MAE by Forecast Horizon</p>
                    </div>
"""

    html += """
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Metrics</h2>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Model</th>
                                <th>Weighted Score</th>
                                <th><span class="badge badge-h1">h=1</span> MAE</th>
                                <th><span class="badge badge-h2">h=2</span> MAE</th>
                                <th><span class="badge badge-h12">h=12</span> MAE</th>
                            </tr>
                        </thead>
                        <tbody>
"""

    # Add table rows from consolidated metrics
    if not df.empty:
        for _, row in df.iterrows():
            html += f"""
                            <tr>
                                <td>{row["Model"]}</td>
                                <td class="metric-value">{row["Weighted_Score"]:.4f}</td>
                                <td>{row["MAE_h1"]:.4f}</td>
                                <td>{row["MAE_h2"]:.4f}</td>
                                <td>{row["MAE_h12"]:.4f}</td>
                            </tr>
"""
    else:
        html += """
                            <tr>
                                <td colspan="5" style="text-align: center; color: #6c757d;">No data available</td>
                            </tr>
"""

    html += """
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="section">
                <div class="interpretation">
                    <h3>Interpretation</h3>
                    <p><strong>Weighted Score Calculation:</strong> The composite score is calculated as 50% h=1 MAE + 30% h=2 MAE + 20% h=12 MAE, prioritizing short-term accuracy while accounting for medium-term performance.</p>
                    <p><strong>Lower is Better:</strong> All MAE values represent average absolute errors in percentage points. A value of 0.236 means the model's predictions are, on average, within 0.236 percentage points of actual values.</p>
                </div>
            </div>
        </div>
        
        <footer>
            <p>Opus Edge Lab - Autonomous Economic Forecasting System | Generated by Ralph Universal Agent</p>
        </footer>
    </div>
</body>
</html>
"""

    # Write HTML file
    output_path = os.path.join(REPORTS_DIR, "model_performance.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Generated: {output_path}")
    return output_path


def main():
    """Main execution function."""
    print("=" * 60)
    print("Generating Model Performance Report")
    print("=" * 60)

    # Generate charts
    print("\n1. Generating visualization charts...")
    mae_chart = generate_mae_comparison()
    horizon_chart = generate_horizon_performance()

    # Generate HTML report
    print("\n2. Generating HTML report...")
    html_report = generate_html_report(mae_chart, horizon_chart)

    print("\n" + "=" * 60)
    print("Report Generation Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    if mae_chart:
        print(f"  - {mae_chart}")
    if horizon_chart:
        print(f"  - {horizon_chart}")
    if html_report:
        print(f"  - {html_report}")
    print(f"\nOpen the report in a browser:")
    print(f"  file://{html_report}")


if __name__ == "__main__":
    main()
