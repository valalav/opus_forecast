#!/usr/bin/env python3
"""
Seasonal Adjustment Impact Analysis
==================================

Compares forecasting accuracy on raw vs seasonally-adjusted (SA) data.

Models tested:
1. MIDAS (Mixed Data Sampling)
2. OPR Enhanced Ridge
3. Naive (last value persistence)
4. Simple Average
5. Seasonal Naive (same month last year)

Output: data/sa_impact_analysis.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.models.registry import ModelRegistry


def load_raw_data():
    """Load raw inflation data."""
    data_path = Path(__file__).parent.parent / "data" / "enhanced_inflation_data.csv"
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # MIDAS expects "Все товары и услуги" column, add mapping from "mom"
    if "Все товары и услуги" not in df.columns and "mom" in df.columns:
        df["Все товары и услуги"] = df["mom"]

    return df


def load_sa_data():
    """Load seasonally-adjusted data."""
    data_path = Path(__file__).parent.parent / "data" / "sa_fl.csv"
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


def prepare_sa_dataframe(sa_df):
    """Convert SA DataFrame to format expected by models."""
    # The SA data has mom_sa, Nonprod_sa, etc. We need to create
    # a DataFrame with a "mom" column containing SA values
    df = pd.DataFrame({"mom": sa_df["mom_sa"]}, index=sa_df.index)

    # MIDAS expects "Все товары и услуги" column
    df["Все товары и услуги"] = df["mom"]

    # Add other columns if available
    if "Nonprod_sa" in sa_df.columns:
        df["Nonprod"] = sa_df["Nonprod_sa"]
    if "Prod_sa" in sa_df.columns:
        df["Prod"] = sa_df["Prod_sa"]
    if "Serv_sa" in sa_df.columns:
        df["Serv"] = sa_df["Serv_sa"]

    # Add necessary columns from raw data (exogenous vars should stay same)
    raw_df = load_raw_data()
    for col in ["usd_nom_i", "Ki_i", "Ruonia", "Ki"]:
        if col in raw_df.columns:
            df[col] = raw_df[col]

    # Add OPR columns from raw data (needed for opr_ridge)
    opr_cols = [c for c in raw_df.columns if c.startswith("opr_")]
    for col in opr_cols:
        df[col] = raw_df[col]

    # Drop rows with NaN in target column (mom)
    df = df.dropna(subset=["mom"])

    return df


def calculate_mae(actuals, predictions):
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(actuals - predictions))


class NaiveForecaster:
    """Naive forecast: last value persistence."""

    name = "Naive"
    MIN_TRAIN_SIZE = 2

    def forecast(self, df, horizon=1):
        """Forecast using last observed value."""
        last_value = df["mom"].iloc[-1]
        return np.full(horizon, last_value)


class SimpleAverageForecaster:
    """Simple average forecast."""

    name = "SimpleAverage"
    MIN_TRAIN_SIZE = 2

    def forecast(self, df, horizon=1):
        """Forecast using historical mean."""
        mean_value = df["mom"].mean()
        return np.full(horizon, mean_value)


class SeasonalNaiveForecaster:
    """Seasonal naive: same month from previous year."""

    name = "SeasonalNaive"
    MIN_TRAIN_SIZE = 12

    def forecast(self, df, horizon=1):
        """Forecast using same month from previous year."""
        forecasts = []
        last_date = df.index[-1]

        for h in range(1, horizon + 1):
            target_date = last_date + pd.DateOffset(months=h)
            # Look for same month in previous year
            prev_year_date = target_date - pd.DateOffset(years=1)

            if prev_year_date in df.index:
                forecast = df.loc[prev_year_date, "mom"]
            else:
                # Fallback to last available value
                forecast = df["mom"].iloc[-1]

            forecasts.append(forecast)

        return np.array(forecasts)


class DriftForecaster:
    """Drift forecast: linear extrapolation from first to last value."""

    name = "Drift"
    MIN_TRAIN_SIZE = 2

    def forecast(self, df, horizon=1):
        """Forecast using drift model."""
        first_value = df["mom"].iloc[0]
        last_value = df["mom"].iloc[-1]
        n = len(df)
        drift_per_period = (last_value - first_value) / (n - 1)

        forecasts = []
        for h in range(1, horizon + 1):
            forecast = last_value + h * drift_per_period
            forecasts.append(forecast)

        return np.array(forecasts)


class MovingAverageForecaster:
    """Moving average forecast: average of last N periods."""

    name = "MovingAverage"
    MIN_TRAIN_SIZE = 12

    def __init__(self, window=3):
        self.window = window

    def forecast(self, df, horizon=1):
        """Forecast using moving average."""
        last_values = df["mom"].tail(self.window)
        forecast_value = last_values.mean()
        return np.full(horizon, forecast_value)


def run_backtest_simple(forecaster_class, df, start_date="2019-01-01"):
    """Run backtest for simple forecasters."""
    forecasts = []
    actuals = []

    df = df.copy()
    target_dates = df.loc[df.index >= start_date].index

    for target_date in target_dates:
        # Get data up to target_date (excluding target)
        train_df = df[df.index < target_date]

        if len(train_df) < forecaster_class.MIN_TRAIN_SIZE:
            continue

        # Generate forecast
        forecaster = forecaster_class()
        try:
            fc = forecaster.forecast(train_df, horizon=1)
        except:
            continue

        # Get actual value
        if target_date in df.index:
            actual = df.loc[target_date, "mom"]
            forecasts.append(fc[0])
            actuals.append(actual)

    if len(forecasts) == 0:
        return None

    return np.array(forecasts), np.array(actuals)


def run_backtest_sirena(model_name, df, start_date="2019-01-01"):
    """Run backtest for Sirena models."""
    try:
        model = ModelRegistry.get(model_name)

        # Debug info
        print(f"  DataFrame shape: {df.shape}, Columns: {list(df.columns)[:10]}")
        print(f"  Start date: {start_date}")

        # Check for target column
        if "Все товары и услуги" not in df.columns:
            print(f"  Warning: 'Все товары и услуги' column not found, using 'mom'")
            target_col = "mom"
        else:
            target_col = "Все товары и услуги"

        # Check for null values in target column
        null_count = df[target_col].isna().sum()
        print(f"  Null values in target column: {null_count}")

        # Check date range
        print(f"  Date range: {df.index.min()} to {df.index.max()}")

        # MIDAS uses different backtest signature
        if model_name == "midas":
            result = model.backtest(df=df, start_date=start_date, target_col=target_col)
        else:
            result = model.backtest(
                df=df, start_date=start_date, horizon=1, target_col=target_col
            )

        if result.empty:
            print(f"  Empty result DataFrame")
            return None

        return result["prediction"].values, result["actual"].values
    except Exception as e:
        import traceback

        print(f"  Error backtesting {model_name}: {e}")
        print(f"  Traceback: {traceback.format_exc()[:500]}")
        return None


def main():
    """Main analysis."""
    print("=" * 60)
    print("Seasonal Adjustment Impact Analysis")
    print("=" * 60)

    # Load data
    raw_df = load_raw_data()
    print(f"Loaded raw data: {len(raw_df)} observations")

    sa_df_full = load_sa_data()
    print(f"Loaded SA data: {len(sa_df_full)} observations")

    # Prepare SA data for models
    sa_df = prepare_sa_dataframe(sa_df_full)

    # Define models to test
    simple_models = [
        NaiveForecaster,
        SimpleAverageForecaster,
        SeasonalNaiveForecaster,
        DriftForecaster,
        MovingAverageForecaster,
    ]

    sirena_models = ["midas", "opr_ridge"]

    results = []

    # Test simple models
    print("\n--- Testing Simple Models ---")
    for model_class in simple_models:
        # Raw data
        print(f"\n{model_class.name} (Raw):")
        result_raw = run_backtest_simple(model_class, raw_df)
        if result_raw is not None:
            fc_raw, act_raw = result_raw
            mae_raw = calculate_mae(act_raw, fc_raw)
            print(f"  MAE: {mae_raw:.4f}, N: {len(act_raw)}")
        else:
            mae_raw = None
            print(f"  No valid forecasts")

        # SA data
        print(f"{model_class.name} (SA):")
        result_sa = run_backtest_simple(model_class, sa_df)
        if result_sa is not None:
            fc_sa, act_sa = result_sa
            mae_sa = calculate_mae(act_sa, fc_sa)
            print(f"  MAE: {mae_sa:.4f}, N: {len(act_sa)}")
        else:
            mae_sa = None
            print(f"  No valid forecasts")

        # Calculate delta
        if mae_raw is not None and mae_sa is not None:
            delta_pct = ((mae_sa - mae_raw) / mae_raw) * 100
        else:
            delta_pct = None

        results.append(
            {
                "Model": model_class.name,
                "MAE_raw": mae_raw,
                "MAE_SA": mae_sa,
                "Delta_pct": delta_pct,
            }
        )

    # Test Sirena models
    print("\n--- Testing Sirena Models ---")
    for model_name in sirena_models:
        # Raw data
        print(f"\n{model_name} (Raw):")
        result_raw = run_backtest_sirena(model_name, raw_df)
        if result_raw is not None:
            fc_raw, act_raw = result_raw
            mae_raw = calculate_mae(act_raw, fc_raw)
            print(f"  MAE: {mae_raw:.4f}, N: {len(act_raw)}")
        else:
            mae_raw = None
            print(f"  No valid forecasts")

        # SA data
        print(f"{model_name} (SA):")
        result_sa = run_backtest_sirena(model_name, sa_df)
        if result_sa is not None:
            fc_sa, act_sa = result_sa
            mae_sa = calculate_mae(act_sa, fc_sa)
            print(f"  MAE: {mae_sa:.4f}, N: {len(act_sa)}")
        else:
            mae_sa = None
            print(f"  No valid forecasts")

        # Calculate delta
        if mae_raw is not None and mae_sa is not None:
            delta_pct = ((mae_sa - mae_raw) / mae_raw) * 100
        else:
            delta_pct = None

        results.append(
            {
                "Model": model_name,
                "MAE_raw": mae_raw,
                "MAE_SA": mae_sa,
                "Delta_pct": delta_pct,
            }
        )

    # Save results
    results_df = pd.DataFrame(results)
    output_path = Path(__file__).parent.parent / "data" / "sa_impact_analysis.csv"
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("Summary Results:")
    print("=" * 60)
    print(results_df.to_string(index=False))

    print(f"\nSaved results to {output_path}")

    # Count models tested
    valid_models = results_df.dropna(subset=["MAE_raw", "MAE_SA"])
    print(f"\nModels tested successfully: {len(valid_models)}")

    # Interpretation
    print("\n--- Interpretation ---")
    for _, row in results_df.iterrows():
        delta = row["Delta_pct"]
        delta_float = float(delta) if pd.notna(delta) else None
        if delta_float is not None:
            if delta_float < -5:
                print(
                    f"{row['Model']}: SA IMPROVES accuracy by {abs(delta_float):.1f}%"
                )
            elif delta_float > 5:
                print(f"{row['Model']}: SA DEGRADES accuracy by {delta_float:.1f}%")
            else:
                print(f"{row['Model']}: SA has minimal impact ({delta_float:.1f}%)")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
