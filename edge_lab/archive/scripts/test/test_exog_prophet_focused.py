#!/usr/bin/env python3
"""
Focused Prophet test - USD only, optimized hyperparameters
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

from prophet import Prophet

# Load data
data_path = Path(__file__).parent.parent / "data" / "inflation_data.csv"

macro_df = pd.read_csv(data_path, sep=";", decimal=",")
macro_df["Date"] = pd.to_datetime(macro_df["Date"], format="%d.%m.%Y")
macro_df = macro_df.set_index("Date").sort_index()

# Simple feature engineering - USD lag 1
result = macro_df.copy()
result["target"] = result["mom"] - 100
result["usd_lag1"] = result["usd_nom_i"].shift(1)
result["usd_lag1"] = (result["usd_lag1"] - 100) / 10

# USD rate of change
result["usd_roc"] = result["usd_nom_i"].diff(1).shift(1)
result["usd_roc"] = result["usd_roc"] / 10

# Drop rows where target is NaN only (keep regressor NaNs)
result = result.dropna(subset=["target", "usd_lag1"])

# Test dates
test_start = pd.Timestamp("2019-01-01")
test_dates = result[result.index >= test_start].index

print(f"Testing {len(test_dates)} forecasts...")
print("=" * 60)

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
        prophet_df["usd_lag1"] = train_data["usd_lag1"].values

        # Remove outlier year 2022
        prophet_df["year"] = prophet_df["ds"].dt.year
        prophet_df = prophet_df[~prophet_df["year"].isin([2022])]
        prophet_df = prophet_df.drop("year", axis=1)

        if len(prophet_df) < 24:
            continue

        last_date = prophet_df["ds"].max()

        # Create Prophet model with very specific hyperparameters
        model = Prophet(
            yearly_seasonality=False,  # Disable, use monthly only
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.001,  # Very conservative
            seasonality_prior_scale=0.1,
            mcmc_samples=0,
        )

        # Add custom seasonality
        model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        model.add_seasonality(name="quarterly", period=91.25, fourier_order=3)

        # Add regressor
        model.add_regressor("usd_lag1", standardize=False)
        model.add_regressor("usd_roc", standardize=False)

        model.fit(prophet_df)

        # Forecast
        future_dates = pd.date_range(start=last_date, periods=2, freq="ME")[1:]
        future = pd.DataFrame({"ds": future_dates})
        future["usd_lag1"] = train_data["usd_lag1"].iloc[-1]
        future["usd_roc"] = train_data["usd_roc"].iloc[-1]

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
        print(f"Error at {target_date}: {e}")
        continue

if len(results) == 0:
    print("No results generated!")
    sys.exit(1)

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
