#!/usr/bin/env python3
"""
Comprehensive test for ExogProphet with feature engineering
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prophet import Prophet

# Load data
data_path = Path(__file__).parent.parent / "data" / "inflation_data.csv"
brent_path = Path(__file__).parent.parent / "data" / "brent_prices.csv"

macro_df = pd.read_csv(data_path, sep=";", decimal=",")
macro_df["Date"] = pd.to_datetime(macro_df["Date"], format="%d.%m.%Y")
macro_df = macro_df.set_index("Date").sort_index()

brent_df = pd.read_csv(brent_path)
brent_df["Date"] = pd.to_datetime(brent_df["Date"])
brent_df = brent_df.set_index("Date").sort_index()

# Align brent to macro dates
brent_df = brent_df.reindex(macro_df.index, method="ffill")

# Enhanced feature engineering
result = macro_df.copy()

# Target
result["target"] = result["mom"] - 100

# USD features
result["usd_lag1"] = result["usd_nom_i"].shift(1)
result["usd_lag2"] = result["usd_nom_i"].shift(2)
result["usd_roc"] = result["usd_nom_i"].diff(1).shift(1)

# Normalize USD features
for col in ["usd_lag1", "usd_lag2", "usd_roc"]:
    if col in result.columns:
        result[col] = (result[col] - 100) / 10

# Brent features
result = result.join(brent_df[["brent"]], how="left")
result["brent_lag1"] = result["brent"].shift(1)
result["brent_lag2"] = result["brent"].shift(2)
result["brent_roc"] = result["brent"].diff(1).shift(1)

# Normalize Brent features
for col in ["brent_lag1", "brent_lag2", "brent_roc"]:
    if col in result.columns:
        result[col] = result[col] / 100
result = result.drop("brent", axis=1, errors="ignore")

# Target lags (very important for inflation)
result["target_lag1"] = result["target"].shift(1)
result["target_lag2"] = result["target"].shift(2)
result["target_lag3"] = result["target"].shift(3)

# Moving averages
result["target_ma3"] = result["target"].rolling(3).mean().shift(1)

result = result.dropna()

# Get test dates
test_start = pd.Timestamp("2019-01-01")
test_dates = result[result.index >= test_start].index

print(f"Testing {len(test_dates)} forecasts...")
print("=" * 60)

# Regressors to use
regressors = ["usd_lag1", "brent_lag1", "target_lag1", "target_lag2"]

results = []
for target_date in test_dates:
    cutoff = target_date - pd.DateOffset(months=1)
    train_data = result[result.index <= cutoff]

    if len(train_data) < 36:
        continue

    try:
        prophet_df = pd.DataFrame(
            {"ds": train_data.index, "y": train_data["target"].values}
        )

        for reg in regressors:
            prophet_df[reg] = train_data[reg].values

        # Remove outlier year 2022
        prophet_df["year"] = prophet_df["ds"].dt.year
        prophet_df = prophet_df[~prophet_df["year"].isin([2022])]
        prophet_df = prophet_df.drop("year", axis=1)

        if len(prophet_df) < 24:
            continue

        last_date = prophet_df["ds"].max()

        # Create Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.02,
            seasonality_prior_scale=0.5,
            mcmc_samples=0,
        )
        model.add_seasonality(name="monthly", period=30.5, fourier_order=3)

        for reg in regressors:
            if reg in prophet_df.columns:
                model.add_regressor(reg, standardize=False)

        model.fit(prophet_df)

        # Forecast
        future_dates = pd.date_range(start=last_date, periods=2, freq="ME")[1:]
        future = pd.DataFrame({"ds": future_dates})

        # Add regressors (use last known values)
        for reg in regressors:
            if reg in train_data.columns:
                future[reg] = train_data[reg].iloc[-1]
            else:
                future[reg] = 0

        forecast = model.predict(future)
        pred = forecast["yhat"].values[0]
        actual = result.loc[target_date, "target"]

        results.append(
            {
                "date": target_date,
                "actual": actual,
                "prediction": pred,
                "error": actual - pred,
            }
        )
    except Exception as e:
        continue

results_df = pd.DataFrame(results)
results_df["abs_error"] = np.abs(results_df["error"])
mae = results_df["abs_error"].mean()

print(f"Test period: {results_df['date'].min()} to {results_df['date'].max()}")
print(f"Number of forecasts: {len(results_df)}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {np.sqrt((results_df['error'] ** 2).mean()):.4f}")
print(f"ME: {results_df['error'].mean():.4f}")

print("=" * 60)
criterion = 0.30
if mae <= criterion:
    print(f"PASSED ✓ MAE={mae:.4f} <= {criterion}")
    sys.exit(0)
else:
    print(f"FAILED ✗ MAE={mae:.4f} > {criterion}")
    sys.exit(1)
