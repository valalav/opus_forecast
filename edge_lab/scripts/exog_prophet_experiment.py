import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import json

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
except ImportError:
    Prophet = None


def load_cpi_data():
    """Load CPI data."""
    cpi_path = Path("data/enhanced_inflation_data.csv")
    df = pd.read_csv(cpi_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    # Convert mom to deviation from 100
    df["y"] = df["mom"] - 100
    return df


def load_brent_data():
    """Load Brent data."""
    brent_path = Path("../data/brent_prices.csv")
    df = pd.read_csv(brent_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


def create_prophet_univariate(train_df, target_date):
    """Model A: Prophet (Univariate)."""
    prophet_df = train_df[["y"]].reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df = prophet_df.dropna()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        mcmc_samples=0,
    )

    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.fit(prophet_df)

    # Forecast for target date
    future = pd.DataFrame({"ds": [target_date]})
    forecast = model.predict(future)
    return forecast["yhat"].values[0]


def create_prophet_with_brent(train_df, brent_df, target_date, optimal_lag=7):
    """Model B: Prophet + Brent (shifted by optimal lag)."""
    # Merge Brent with CPI
    merged = train_df.join(brent_df[["brent"]], how="left")
    merged["brent"] = merged["brent"].fillna(method="ffill")

    # Create lagged Brent feature with optimal lag
    merged[f"brent_lag{optimal_lag}"] = merged["brent"].shift(optimal_lag)

    # Normalize (similar to ExogProphetForecaster)
    merged[f"brent_lag{optimal_lag}"] = merged[f"brent_lag{optimal_lag}"] / 100
    merged[f"brent_lag{optimal_lag}"] = merged[f"brent_lag{optimal_lag}"].fillna(0.7)

    # Prepare Prophet format
    prophet_df = merged[["y", f"brent_lag{optimal_lag}"]].reset_index()
    prophet_df.columns = ["ds", "y", f"brent_lag{optimal_lag}"]
    prophet_df = prophet_df.dropna()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        mcmc_samples=0,
    )

    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.add_regressor(f"brent_lag{optimal_lag}", standardize=False)
    model.fit(prophet_df)

    # Forecast for target date - need the lagged Brent value
    # Get last available lagged Brent value from training data
    last_brent_lag = merged[f"brent_lag{optimal_lag}"].dropna().iloc[-1]

    future = pd.DataFrame(
        {"ds": [target_date], f"brent_lag{optimal_lag}": [last_brent_lag]}
    )
    forecast = model.predict(future)
    return forecast["yhat"].values[0]


def run_backtest(cpi_df, brent_df, start_date="2024-01-01", horizon=12, optimal_lag=7):
    """Run comparative backtest."""
    results = []

    test_dates = cpi_df[cpi_df.index >= pd.Timestamp(start_date)].index[:horizon]

    for target_date in test_dates:
        cutoff = target_date - pd.DateOffset(months=1)
        train_df = cpi_df[cpi_df.index < cutoff].copy()

        if len(train_df) < 24:
            continue

        try:
            # Model A: Univariate Prophet
            pred_a = create_prophet_univariate(train_df, target_date)

            # Model B: Prophet with Brent
            pred_b = create_prophet_with_brent(
                train_df, brent_df, target_date, optimal_lag
            )

            actual = cpi_df.loc[target_date, "y"]

            results.append(
                {
                    "date": target_date.strftime("%Y-%m-%d"),
                    "actual": actual,
                    "prediction_univariate": pred_a,
                    "prediction_brent": pred_b,
                    "error_univariate": actual - pred_a,
                    "error_brent": actual - pred_b,
                }
            )
        except Exception as e:
            print(f"Error on {target_date}: {e}")
            continue

    return pd.DataFrame(results)


def main():
    """Main execution."""
    print("ExogProphet Controlled Experiment")
    print("=" * 50)

    # Load data
    print("Loading data...")
    cpi_df = load_cpi_data()
    brent_df = load_brent_data()

    # Load optimal lag from Task 241
    with open("data/brent_lag_analysis.json", "r") as f:
        lag_analysis = json.load(f)
    optimal_lag = lag_analysis["optimal_lag"]
    print(
        f"Optimal lag from Task 241: {optimal_lag} (correlation: {lag_analysis['correlation']:.4f})"
    )

    # Run backtest
    print(f"Running backtest (h=1, 12 months)...")
    results = run_backtest(
        cpi_df, brent_df, start_date="2024-01-01", horizon=12, optimal_lag=optimal_lag
    )

    # Calculate MAE
    mae_univariate = np.abs(results["error_univariate"]).mean()
    mae_brent = np.abs(results["error_brent"]).mean()

    print(f"\nResults:")
    print(f"Model A (Univariate Prophet): MAE = {mae_univariate:.4f}")
    print(f"Model B (Prophet + Brent lag{optimal_lag}): MAE = {mae_brent:.4f}")
    print(f"Improvement: {(1 - mae_brent / mae_univariate) * 100:.2f}%")

    # Save experiment results
    output_df = pd.DataFrame(
        [
            {"Model": "Prophet_Univariate", "MAE": mae_univariate},
            {"Model": f"Prophet_Brent_Lag{optimal_lag}", "MAE": mae_brent},
        ]
    )
    output_df.to_csv("data/exog_prophet_experiment.csv", index=False)

    # Save detailed predictions
    results.to_csv("data/exog_prophet_details.csv", index=False)

    print(f"\nSaved results to:")
    print(f"  - data/exog_prophet_experiment.csv")
    print(f"  - data/exog_prophet_details.csv")

    # Verify Model B actually uses the regressor
    print("\nVerification:")
    print("Model B uses Brent as regressor with lag:", optimal_lag)

    # Create one more model to show the regressor coefficient
    target_date = results.iloc[0]["date"]
    cutoff = pd.Timestamp(target_date) - pd.DateOffset(months=1)
    train_df = cpi_df[cpi_df.index < cutoff].copy()

    merged = train_df.join(brent_df[["brent"]], how="left")
    merged["brent"] = merged["brent"].ffill()
    merged[f"brent_lag{optimal_lag}"] = merged["brent"].shift(optimal_lag)
    merged[f"brent_lag{optimal_lag}"] = merged[f"brent_lag{optimal_lag}"] / 100
    merged[f"brent_lag{optimal_lag}"] = merged[f"brent_lag{optimal_lag}"].fillna(0.7)

    prophet_df = merged[["y", f"brent_lag{optimal_lag}"]].reset_index()
    prophet_df.columns = ["ds", "y", f"brent_lag{optimal_lag}"]
    prophet_df = prophet_df.dropna()

    print(f"  Training data shape: {prophet_df.shape}")
    print(
        f"  Regressor feature 'brent_lag{optimal_lag}' present in training data: {f'brent_lag{optimal_lag}' in prophet_df.columns}"
    )
    print(f"  Regressor mean value: {prophet_df[f'brent_lag{optimal_lag}'].mean():.4f}")
    print(f"  Regressor std value: {prophet_df[f'brent_lag{optimal_lag}'].std():.4f}")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        mcmc_samples=0,
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    model.add_regressor(f"brent_lag{optimal_lag}", standardize=False)
    model.fit(prophet_df)

    # Check regressor params
    print("  Model params keys:", sorted(model.params.keys()))
    if "beta" in model.params:
        beta = model.params["beta"]
        print(f"  Beta shape: {beta.shape}")
        print(f"  Beta mean: {beta.mean():.6f}")
        print(f"  Regressor coefficient (beta[0]): {beta[0].mean():.6f}")
        print(f"  Regressor used: YES (beta is not all zeros)")
        print(
            f"  Regressor magnitude: {abs(beta[0].mean())} {'(significant)' if abs(beta[0].mean()) > 1e-3 else '(small)'}"
        )
    else:
        print(f"  Checking all params for brent_lag{optimal_lag}:")
        for key in sorted(model.params.keys()):
            if "brent" in key.lower() or f"lag{optimal_lag}" in key or key == "beta":
                print(f"    {key}: mean={model.params[key].mean():.6f}")

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
