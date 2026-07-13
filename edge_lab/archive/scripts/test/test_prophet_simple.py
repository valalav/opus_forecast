#!/usr/bin/env python3
"""
Simple Prophet test - just USD lag1
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

from prophet import Prophet

# Load data
data_path = Path(__file__).parent.parent / "data" / "inflation_data.csv"

df = pd.read_csv(data_path, sep=";", decimal=",")
df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
df = df.set_index("Date").sort_index()

# Prepare target
df["y"] = df["mom"] - 100
df["usd_lag1"] = df["usd_nom_i"].shift(1)
df["usd_lag1"] = (df["usd_lag1"] - 100) / 10

# Drop NaNs
df = df.dropna(subset=["y", "usd_lag1"])

# Test
test_start = pd.Timestamp("2019-01-01")
test_dates = df[df.index >= test_start].index

print(f"Testing {len(test_dates)} forecasts...")
print("=" * 60)

results = []
for target_date in test_dates:
    cutoff = target_date - pd.DateOffset(months=1)
    train_data = df[df.index <= cutoff]

    if len(train_data) < 36:
        continue

    try:
        prophet_df = pd.DataFrame(
            {
                "ds": train_data.index,
                "y": train_data["y"].values,
                "usd_lag1": train_data["usd_lag1"].values,
            }
        )

        last_date = prophet_df["ds"].max()

        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.01,
            seasonality_prior_scale=0.5,
            mcmc_samples=0,
        )
        model.add_seasonality(name="monthly", period=30.5, fourier_order=4)
        model.add_regressor("usd_lag1", standardize=False)

        model.fit(prophet_df)

        future_dates = pd.date_range(start=last_date, periods=2, freq="ME")[1:]
        future = pd.DataFrame({"ds": future_dates})
        future["usd_lag1"] = train_data["usd_lag1"].iloc[-1]

        forecast = model.predict(future)
        pred = forecast["yhat"].values[0]
        actual = df.loc[target_date, "y"]

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

print("=" * 60)
criterion = 0.30
if mae <= criterion:
    print(f"PASSED ✓ MAE={mae:.4f} <= {criterion}")
    sys.exit(0)
else:
    print(f"FAILED ✗ MAE={mae:.4f} > {criterion}")
    sys.exit(1)
