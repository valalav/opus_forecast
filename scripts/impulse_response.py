#!/usr/bin/env python3
"""
Impulse Response Functions for Macro Shocks on KBR Inflation

This script calculates and visualizes how KBR inflation (CPI) responds to
shocks in key macroeconomic variables using a Vector Autoregression (VAR) model.

Variables analyzed:
- CPI: Consumer Price Index (KBR inflation)
- Ki: Key Rate (Central Bank policy rate)
- USD: USD/RUB exchange rate
- Brent: Brent crude oil price

Output:
- IRF plots for 12-month horizon showing dynamic responses to each shock
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

warnings.filterwarnings("ignore")


def load_data(data_dir: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load and merge macroeconomic data for VAR analysis.

    Args:
        data_dir: Path to data directory

    Returns:
        Tuple of (merged_df, variable_labels)
    """
    data_path = Path(data_dir)

    # Load inflation data (CPI, Ki, USD)
    infl_df = pd.read_csv(
        data_path / "inflation_data.csv",
        sep=";",
        decimal=",",
        parse_dates=["Date"],
        dayfirst=True,
    )

    # Convert mom from 100.xx format to percentage (e.g., 101.49 -> 1.49%)
    infl_df["CPI"] = infl_df["mom"] - 100

    # Convert Key Rate from index to actual rate
    infl_df["Ki"] = infl_df["Ki_i"]

    # Use USD nominal index
    infl_df["USD"] = infl_df["usd_nom_i"]

    # Load Brent prices
    brent_df = pd.read_csv(
        data_path / "brent_prices.csv",
        parse_dates=["Date"],
    )

    # Create month key for merging (since dates don't match exactly)
    infl_df["month_key"] = infl_df["Date"].dt.to_period("M")
    brent_df["month_key"] = brent_df["Date"].dt.to_period("M")

    # Merge datasets on month_key
    merged_df = infl_df[["Date", "CPI", "Ki", "USD", "month_key"]].merge(
        brent_df[["brent", "month_key"]], on="month_key", how="inner"
    )

    # Rename brent column to match variable names
    merged_df = merged_df.rename(columns={"brent": "Brent"})

    merged_df = merged_df.set_index("Date")

    # Variable labels for plots
    variable_labels = {
        "CPI": "CPI Inflation (%)",
        "Ki": "Key Rate (%)",
        "USD": "USD/RUB",
        "Brent": "Brent Price ($/bbl)",
    }

    return merged_df, variable_labels


def prepare_var_data(
    df: pd.DataFrame, start_date: str = "2016-01-01", end_date: str = None
) -> pd.DataFrame:
    """
    Prepare data for VAR model by filtering period and handling missing values.

    Args:
        df: Merged macro data
        start_date: Start date for analysis (default: 2016, post-Crimea)
        end_date: End date for analysis (default: None, use all data)

    Returns:
        DataFrame ready for VAR model
    """
    # Filter by date range
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]

    # Select VAR variables
    var_data = df[["CPI", "Ki", "USD", "Brent"]].copy()

    # Check for missing values
    print(
        f"\nData period: {var_data.index[0].strftime('%Y-%m')} to {var_data.index[-1].strftime('%Y-%m')}"
    )
    print(f"Total observations: {len(var_data)}")
    print(f"\nMissing values check:")
    print(var_data.isnull().sum())

    # Forward-fill missing values (common for monthly data)
    var_data = var_data.ffill().bfill()

    return var_data


def fit_var_model(var_data: pd.DataFrame, maxlags: int = 6) -> VAR:
    """
    Fit VAR model and select optimal lag order using AIC.

    Args:
        var_data: DataFrame with variables for VAR
        maxlags: Maximum lag order to test

    Returns:
        Fitted VAR model with optimal lag order
    """
    print(f"\nFitting VAR model (max lags: {maxlags})...")

    model = VAR(var_data)

    # Select lag order using AIC
    lag_order = model.select_order(maxlags=maxlags)
    selected_lag = lag_order.aic

    print(f"Optimal lag order: {selected_lag} (AIC)")
    print(f"Lag order statistics:")
    print(lag_order.summary())

    # Fit model with optimal lag
    var_result = model.fit(selected_lag)

    # Check stability condition (all roots inside unit circle)
    roots = var_result.roots
    max_root = np.max(np.abs(roots))
    stability_status = "STABLE" if max_root < 1.0 else "UNSTABLE"
    print(f"\nVAR stability check: {stability_status} (max root = {max_root:.4f})")

    if max_root >= 1.0:
        warnings.warn("VAR model is unstable! Results may not be reliable.")

    return var_result


def generate_irf_plots(
    var_result: VAR, variable_labels: Dict[str, str], output_dir: str, periods: int = 12
) -> None:
    """
    Generate and save IRF plots for all variable shocks.

    Args:
        var_result: Fitted VAR model result
        variable_labels: Dictionary mapping variable names to display labels
        output_dir: Directory to save plots
        periods: Horizon for IRF (months)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Compute IRFs
    irf = var_result.irf(periods=periods)

    variables = ["CPI", "Ki", "USD", "Brent"]

    # Generate individual shock plots
    for shock_var in variables:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot response of each variable to the shock
        for response_var in variables:
            irf_values = irf.irfs[
                :, variables.index(shock_var), variables.index(response_var)
            ]
            months = np.arange(periods + 1)

            ax.plot(
                months,
                irf_values,
                label=variable_labels[response_var],
                linewidth=2,
                alpha=0.8,
            )

            # Mark zero line
            ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_xlabel("Months after shock", fontsize=11)
        ax.set_ylabel("Response", fontsize=11)
        ax.set_title(
            f"Impulse Response Functions\nShock to: {variable_labels[shock_var]}",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        filename = output_path / f"irf_{shock_var.lower()}_shock.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Saved: {filename}")

    # Generate combined plot with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, shock_var in enumerate(variables):
        ax = axes[idx]

        # Plot CPI response to each shock
        irf_values = irf.irfs[:, variables.index(shock_var), variables.index("CPI")]
        months = np.arange(periods + 1)

        ax.plot(
            months,
            irf_values,
            label="CPI Response",
            linewidth=2.5,
            color="red",
            alpha=0.8,
        )

        # Mark zero line
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_xlabel("Months after shock", fontsize=10)
        ax.set_ylabel("CPI Response (pp)", fontsize=10)
        ax.set_title(
            f"Shock to {variable_labels[shock_var]}", fontsize=11, fontweight="bold"
        )
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save combined plot
    combined_filename = output_path / "irf_combined.png"
    plt.savefig(combined_filename, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {combined_filename}")


def print_irf_summary(var_result: VAR, periods: int = 12) -> None:
    """
    Print summary statistics of IRFs.

    Args:
        var_result: Fitted VAR model result
        periods: Horizon for IRF
    """
    irf = var_result.irf(periods=periods)
    variables = ["CPI", "Ki", "USD", "Brent"]

    print("\n" + "=" * 70)
    print("IMPULSE RESPONSE SUMMARY (12-month horizon)")
    print("=" * 70)

    for shock_var in variables:
        print(f"\nShock to: {shock_var}")
        print("-" * 50)

        for response_var in variables:
            irf_values = irf.irfs[
                :, variables.index(shock_var), variables.index(response_var)
            ]

            # Calculate summary statistics
            peak = np.max(irf_values)
            peak_month = np.argmax(irf_values)
            trough = np.min(irf_values)
            trough_month = np.argmin(irf_values)
            cum_effect = np.sum(irf_values)

            print(f"  Response of {response_var}:")
            print(f"    Peak:      {peak:+.4f} at month {peak_month}")
            print(f"    Trough:    {trough:+.4f} at month {trough_month}")
            print(f"    Cumulative: {cum_effect:+.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Impulse Response Functions for macro shocks on KBR inflation"
    )
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument(
        "--output-dir",
        default="assets/charts",
        help="Path to output directory for IRF plots",
    )
    parser.add_argument(
        "--start-date",
        default="2016-01-01",
        help="Start date for analysis (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date", default=None, help="End date for analysis (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--horizon", type=int, default=12, help="IRF horizon in months (default: 12)"
    )
    parser.add_argument(
        "--maxlags", type=int, default=6, help="Maximum lag order for VAR (default: 6)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("IMPULSE RESPONSE FUNCTIONS: KBR INFLATION")
    print("=" * 70)

    # Load data
    merged_df, variable_labels = load_data(args.data_dir)

    # Prepare VAR data
    var_data = prepare_var_data(
        merged_df, start_date=args.start_date, end_date=args.end_date
    )

    # Fit VAR model
    var_result = fit_var_model(var_data, maxlags=args.maxlags)

    # Print IRF summary
    print_irf_summary(var_result, periods=args.horizon)

    # Generate IRF plots
    generate_irf_plots(
        var_result, variable_labels, args.output_dir, periods=args.horizon
    )

    print("\n" + "=" * 70)
    print("IRF analysis complete!")
    print(f"Plots saved to: {args.output_dir}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
