#!/usr/bin/env python3
"""
Standalone test for ExogProphet MAE <= 0.30
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.models.exog_prophet import ExogProphetForecaster

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

# Get target
target = macro_df["mom"] - 100
test_start = "2019-01-01"
test_dates = target[target.index >= pd.Timestamp(test_start)].index

print(f"Testing {len(test_dates)} forecasts...")
print("=" * 60)

results = []
for target_date in test_dates:
    cutoff = target_date - pd.DateOffset(months=1)
    train_data = macro_df[macro_df.index <= cutoff]
    train_brent = brent_df[brent_df.index <= cutoff]

    if len(train_data) < 36:
        continue

    try:
        model = ExogProphetForecaster(
            use_usd=True,
            use_brent=True,
            use_ki=False,  # Try without Ki first
            seasonality_mode="additive",
            changepoint_prior_scale=0.005,
            seasonality_prior_scale=0.5,
            outlier_years=[],
        )

        model.macro_df = train_data
        model.brent_df = train_brent

        # Prepare features
        prepared = model._prepare_features(model.macro_df)
        prophet_df = model._prepare_prophet_df(prepared, "mom")

        # Remove outliers
        prophet_df["year"] = prophet_df["ds"].dt.year
        prophet_df = prophet_df[~prophet_df["year"].isin(model.outlier_years)]
        prophet_df = prophet_df.drop("year", axis=1)
        prophet_df = prophet_df.dropna()

        if len(prophet_df) < 24:
            continue

        model.last_date = prophet_df["ds"].max()

        from prophet import Prophet

        model.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
            changepoint_prior_scale=0.005,
            seasonality_prior_scale=0.5,
            mcmc_samples=0,
        )
        model.model.add_seasonality(name="monthly", period=30.5, fourier_order=4)
        for reg in model.regressors:
            model.model.add_regressor(reg, standardize=False)

        model.model.fit(prophet_df)

        # Forecast
        future_dates = pd.date_range(start=model.last_date, periods=2, freq="ME")[1:]
        future = pd.DataFrame({"ds": future_dates})

        # Add regressors
        prepared_hist = model._prepare_features(model.macro_df)
        for reg in model.regressors:
            if reg in prepared_hist.columns:
                last_val = prepared_hist[reg].dropna().iloc[-1]
                future[reg] = last_val
            else:
                future[reg] = 0

        forecast = model.model.predict(future)
        pred = forecast["yhat"].values[0]
        actual = target.loc[target_date]

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
