#!/usr/bin/env python3
"""Generate model performance HTML report with images and metrics table."""

import pandas as pd
from datetime import datetime
import os

# Paths
BASE_DIR = "/home/valalav/_projects/sirena-kbr/edge_lab"
HTML_PATH = os.path.join(BASE_DIR, "assets/reports/model_performance.html")
CSV_PATH = os.path.join(BASE_DIR, "data/consolidated_metrics.csv")


def generate_html():
    """Generate the HTML report."""
    # Read consolidated metrics
    df = pd.read_csv(CSV_PATH)

    # Current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate table rows
    table_rows = ""
    for _, row in df.iterrows():
        model = row["Model"]
        weighted = f"{row['Weighted_Score']:.4f}"
        mae_h1 = f"{row['MAE_h1']:.4f}"
        mae_h2 = f"{row['MAE_h2']:.4f}"
        mae_h12 = f"{row['MAE_h12']:.4f}"

        table_rows += f"""
                            <tr>
                                <td>{model}</td>
                                <td class="metric-value">{weighted}</td>
                                <td>{mae_h1}</td>
                                <td>{mae_h2}</td>
                                <td>{mae_h12}</td>
                            </tr>
"""

    # HTML template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Performance Report - Opus Edge Lab</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
        }}
        
        h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        
        .timestamp {{
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        h2 {{
            color: #667eea;
            margin-bottom: 20px;
            font-size: 22px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .charts-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 30px;
        }}
        
        .chart-wrapper {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .chart-wrapper img {{
            width: 100%;
            height: auto;
            border-radius: 6px;
        }}
        
        .chart-caption {{
            text-align: center;
            margin-top: 10px;
            color: #6c757d;
            font-size: 14px;
        }}
        
        .table-container {{
            overflow-x: auto;
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        
        th {{
            background: #667eea;
            color: white;
            padding: 12px 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 10px 15px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        tr:last-child td {{
            border-bottom: none;
        }}
        
        tr:nth-child(even) {{
            background: rgba(102, 126, 234, 0.05);
        }}
        
        tr:hover {{
            background: rgba(102, 126, 234, 0.1);
        }}
        
        .metric-value {{
            font-weight: 600;
            color: #667eea;
        }}
        
        .interpretation {{
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border-left: 4px solid #667eea;
            padding: 20px;
            border-radius: 6px;
            margin-bottom: 20px;
        }}
        
        .interpretation h3 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 18px;
        }}
        
        .interpretation p {{
            color: #495057;
            line-height: 1.6;
        }}
        
        footer {{
            background: #f8f9fa;
            padding: 20px 30px;
            text-align: center;
            color: #6c757d;
            font-size: 13px;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 600;
        }}
        
        .badge-h1 {{ background: #28a745; color: white; }}
        .badge-h2 {{ background: #17a2b8; color: white; }}
        .badge-h12 {{ background: #dc3545; color: white; }}
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

                    <div class="chart-wrapper">
                        <img src="../charts/mae_comparison.png" alt="MAE Comparison">
                        <p class="chart-caption">Model Performance Comparison (Weighted MAE)</p>
                    </div>

                    <div class="chart-wrapper">
                        <img src="../charts/forecast_trajectories.png" alt="Forecast Trajectories">
                        <p class="chart-caption">MAE by Forecast Horizon</p>
                    </div>

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
{table_rows}
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
    os.makedirs(os.path.dirname(HTML_PATH), exist_ok=True)
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML report generated: {HTML_PATH}")
    return True


if __name__ == "__main__":
    generate_html()
