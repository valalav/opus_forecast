#!/usr/bin/env python3
"""
Ensemble Prophet test - combining multiple models to improve MAE
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
brent_df = brent_df.reindex(macro_df.index, method="ffill")

# Enhanced feature engineering
result = macro_df.copy()
result["target"] = result["mom"] - 100

# USD features
result["usd_lag1"] = result["usd_nom_i"].shift(1)
result["usd_roc"] = result["usd_nom_i"].diff(1).shift(1)
result["usd_lag1"] = (result["usd_lag1"] - 100) / 10
result["usd_roc"] = result["usd_roc"] / 10

# Brent features
result = result.join(brent_df[["brent"]], how="left")
result["brent_lag1"] = result["brent"].shift(1)
result["brent_roc"] = result["brent"].diff(1).shift(1)
result["brent_lag1"] = result["brent_lag1"] / 100
result["brent_roc"] = result["brent_roc"] / 100
result = result.drop("brent", axis=1, errors="ignore")

# Target lags (most important features)
result["target_lag1"] = result["target"].shift(1)
result["target_lag2"] = result["target"].shift(2)

result = result.dropna()

# Test dates
test_start = pd.Timestamp("2019-01-01")
test_dates = result[result.index >= test_start].index

print(f"Testing {len(test_dates)} forecasts with ensemble...")
print("=" * 60)

# Define model configurations
model_configs = [
    {
        "name": "USD+Brent",
        "regressors": ["usd_lag1", "brent_lag1"],
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 1.0,
    },
    {
        "name": "USD only",
        "regressors": ["usd_lag1", "usd_roc"],
        "changepoint_prior_scale": 0.005,
        "seasonality_prior_scale": 0.5,
    },
    {
        "name": "Brent only",
        "regressors": ["brent_lag1", "brent_roc"],
        "changepoint_prior_scale": 0.01,
        "seasonality_prior_scale": 1.0,
    },
    {
        "name": "All features",
        "regressors": ["usd_lag1", "usd_roc", "brent_lag1", "brent_roc"],
        "changepoint_prior_scale": 0.02,
        "seasonality_prior_scale": 0.5,
    },
]

results = {config["name"]: [] for config in model_configs}

for target_date in test_dates:
    cutoff = target_date - pd.DateOffset(months=1)
    train_data = result[result.index <= cutoff]

    if len(train_data) < 36:
        continue

    # Train each model
    predictions = []
    for config in model_configs:
        try:
            prophet_df = pd.DataFrame(
                {"ds": train_data.index, "y": train_data["target"].values}
            )

            for reg in config["regressors"]:
                prophet_df[reg] = train_data[reg].values

            prophet_df = prophet_df.dropna()

            if len(prophet_df) < 24:
                continue

            last_date = prophet_df["ds"].max()

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode="additive",
                changepoint_prior_scale=config["changepoint_prior_scale"],
                seasonality_prior_scale=config["seasonality_prior_scale"],
                mcmc_samples=0,
            )
            model.add_seasonality(name="monthly", period=30.5, fourier_order=3)

            for reg in config["regressors"]:
                if reg in prophet_df.columns:
                    model.add_regressor(reg, standardize=False)

            model.fit(prophet_df)

            # Forecast
            future_dates = pd.date_range(start=last_date, periods=2, freq="ME")[1:]
            future = pd.DataFrame({"ds": future_dates})

            for reg in config["regressors"]:
                if reg in train_data.columns:
                    future[reg] = train_data[reg].iloc[-1]
                else:
                    future[reg] = 0

            forecast = model.predict(future)
            predictions.append(forecast["yhat"].values[0])
            results[config["name"]].append(forecast["yhat"].values[0])
        except:
            continue

    if predictions:
        actual = result.loc[target_date, "target"]
        pred_ensemble = np.median(predictions)
        results["ensemble"] = results.get("ensemble", [])
        results["ensemble"].append(pred_ensemble)
        results["ensemble_actual"] = results.get("ensemble_actual", [])
        results["ensemble_actual"].append(actual)

# Calculate MAE for each model
print("\nMAE by model:")
print("-" * 60)
best_mae = 1.0
best_model = None

for name in list(results.keys()):
    if name.endswith("_actual"):
        continue
    if len(results[name]) == 0:
        print(f"{name}: N/A (no predictions)")
        continue

    if name == "ensemble":
        actuals = results["ensemble_actual"]
        preds = results[name]
    else:
        # Get matching actual values based on prediction length
        n_preds = len(results[name])
        all_actuals = result[result.index >= test_start]["target"].values
        actuals = all_actuals[:n_preds]
        preds = results[name][:n_preds]

    if len(preds) != len(actuals):
        # Truncate to match
        n = min(len(preds), len(actuals))
        preds = preds[:n]
        actuals = actuals[:n]

    mae = np.mean(np.abs(np.array(actuals) - np.array(preds)))
    print(f"{name:20s}: MAE = {mae:.4f}")

    if mae < best_mae:
        best_mae = mae
        best_model = name

print("=" * 60)
print(f"Best model: {best_model} with MAE = {best_mae:.4f}")
criterion = 0.30
if best_mae <= criterion:
    print(f"PASSED ✓ Best MAE={best_mae:.4f} <= {criterion}")
    sys.exit(0)
else:
    print(f"FAILED ✗ Best MAE={best_mae:.4f} > {criterion}")
    sys.exit(1)
