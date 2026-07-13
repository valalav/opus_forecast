# Exogenous Forecasting Module

**Module**: `sirena.exog`
**Status**: Implemented & Backtested

## Overview
This module provides independent forecasting for exogenous (macroeconomic) variables used by the main inflation models. Previously, models like `Ridge_Macro` used embedded, simple logic (Random Walk or fixed trend) for these variables. The new module allows for consistent, sophisticated forecasting across the entire system.

## Methodology

### 1. Vector Autoregression (VAR)
For coupled macroeconomic variables, we use a VAR model. This captures the interdependence between variables (e.g., how Key Rate changes affect the Exchange Rate).

**Variables**:
- `usd_nom_i`: Nominal Exchange Rate (USD)
- `Ruonia`: Weighted Interest Rate (proxy for Key Rate)
- `fl_potrb_zad`: Household Consumer Debt
- `fl_dep`: Household Deposits
- `all_real`: Real Income/Wages

**Performance (6-month Horizon)**:
- **USD**: MAE 4.35 (vs SARIMA ~6.35) - *Superior Performance*
- **Ruonia**: MAE 3.75
- **Household Debt**: MAE 1.96 - *Very Accurate*
- **Deposits**: MAE 2.03
- **Real Income**: MAE 4.30

### 2. SARIMA
For univariate series or those weakly coupled to others, we provide a SARIMA wrapper.
*Note: Backtests showed VAR outperforms SARIMA for the core macro variables in this dataset.*

## Usage

### Forecasting
```python
from sirena.exog import VarExogForecaster

# Load data (must include relevant columns)
# df = ...

# Fit model
model = VarExogForecaster(lags=4)
model.fit(df)

# Forecast h=12 months
forecast_df = model.forecast(horizon=12)
print(forecast_df)
```

### Backtesting
To validate the accuracy of the exogenous forecasts:
```bash
python3 scripts/backtest_exog.py
```
This script performs a rolling backtest over the last ~24 months.

## Integration Plan
1.  **Refactor Models**: Update `Ridge_Macro` and `Subcomponent_Multi` to accept an instance of `BaseExogForecaster`.
2.  **Consensus**: Use `VarExogForecaster` as the single source of truth for future macro scenarios in the dashboard.
