# Seasonality Analysis

**Script**: `scripts/visualize_seasonality.py`
**Output**: `assets/charts/seasonality.html`

## Overview
Understanding seasonal patterns is crucial for inflation forecasting, especially for food products (harvest cycles) and services (tariff indexing). This tool decomposes the time series into Trend, Seasonal, and Residual components using an additive model.

## Methodology
- **Decomposition**: Additive (`Observed = Trend + Seasonal + Residual`)
- **Period**: 12 months
- **Libraries**: `statsmodels.tsa.seasonal.seasonal_decompose`, `plotly`

## Usage
Run the script to generate the HTML report:
```bash
python3 scripts/visualize_seasonality.py
```

The output file will be saved to `assets/charts/seasonality.html`. You can open this file in any web browser.

## Visualized Components
The report currently covers:
1.  **Headline Inflation (MoM)**: Overall seasonal pattern.
2.  **Food Products (Prod)**: Strong summer deflation / autumn inflation cycles.
3.  **Non-Food Products (Nonprod)**: Typically stable, with some FX-pass-through effects.
4.  **Services (Serv)**: Tariff shocks (usually July/December).

## Key Findings
- **Food**: Shows the most distinct and robust seasonal pattern using this method.
- **Services**: Seasonality is more "spiky" due to discrete tariff indexation events rather than smooth sine-waves.
