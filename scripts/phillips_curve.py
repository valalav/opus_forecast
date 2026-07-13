#!/usr/bin/env python3
"""
Phillips Curve Estimation for KBR

Analyzes the relationship between unemployment and inflation using:
1. Linear specification: π_t = α - β * u_t + ε_t
2. Accelerationist specification: π_t = π_t-1 - β * (u_t - u*) + ε_t
3. Non-linear specification: π_t = α - β * u_t + γ * u_t^2 + ε_t

Author: Worker Agent (Task 538)
Date: 2026-01-24
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import argparse
from scipy import stats
import warnings

warnings.filterwarnings("ignore")


def load_unemployment_data():
    """Load monthly unemployment data from SA file."""
    print("Loading unemployment data...")

    # Read SA unemployment file
    df = pd.read_excel(
        "assets/charts/ОПР_статистика/ОПР_статистика/ЗП_безработица/Уровень_безработицы_SA.xlsx",
        header=None,
    )

    # Find KBR column (region names in row 3)
    region_row = df.iloc[3, :]
    kbr_col = None
    for col_idx, name in enumerate(region_row):
        if pd.notna(name) and "Кабардин" in str(name):
            kbr_col = col_idx
            break

    if kbr_col is None:
        raise ValueError("KBR not found in unemployment data")

    # Extract monthly unemployment data (data starts from row 4)
    unemp_data = []
    dates_data = []

    for row_idx in range(4, len(df)):
        date = df.iloc[row_idx, 0]
        unemp = df.iloc[row_idx, kbr_col]

        if pd.notna(date) and pd.notna(unemp):
            # Parse date
            if isinstance(date, str):
                date = pd.to_datetime(date)
            dates_data.append(date)
            unemp_data.append(unemp)

    df_unemp = pd.DataFrame({"Date": dates_data, "unemployment_rate": unemp_data})

    df_unemp["Date"] = pd.to_datetime(df_unemp["Date"])
    # Normalize dates to first of month for merging with inflation data
    df_unemp["Date"] = df_unemp["Date"].dt.to_period("M").dt.to_timestamp()
    df_unemp = df_unemp.set_index("Date")

    print(f"Loaded {len(df_unemp)} monthly unemployment observations")
    print(f"Date range: {df_unemp.index.min()} to {df_unemp.index.max()}")
    print(f"Unemployment rate mean: {df_unemp['unemployment_rate'].mean():.2f}%")
    print(f"Unemployment rate std: {df_unemp['unemployment_rate'].std():.2f}%")

    return df_unemp


def load_inflation_data():
    """Load monthly inflation data."""
    print("\nLoading inflation data...")

    df = pd.read_csv(
        "data/inflation_data.csv",
        sep=";",
        decimal=",",
        parse_dates=["Date"],
        dayfirst=True,
    )

    # Calculate inflation rate (MoM - 100)
    df["inflation_rate"] = df["mom"] - 100

    # Normalize dates to first of month for merging with unemployment data
    df["Date"] = pd.to_datetime(df["Date"])
    df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    df = df.set_index("Date")

    print(f"Loaded {len(df)} monthly inflation observations")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Inflation rate mean: {df['inflation_rate'].mean():.3f}%")
    print(f"Inflation rate std: {df['inflation_rate'].std():.3f}%")

    return df


def merge_data(df_unemp, df_inf):
    """Merge unemployment and inflation data on date."""
    print("\nMerging data...")

    # Merge on date
    df = pd.merge(
        df_unemp,
        df_inf[["inflation_rate"]],
        left_index=True,
        right_index=True,
        how="inner",
    )

    print(f"Merged dataset: {len(df)} observations")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Check for missing values
    print(f"Missing values: {df.isna().sum().sum()}")

    return df


def estimate_linear_phillips(df):
    """
    Estimate linear Phillips Curve: π_t = α - β * u_t + ε_t

    Returns:
        dict: Results with coefficient, R-squared, p-values
    """
    print("\n" + "=" * 60)
    print("SPECIFICATION 1: Linear Phillips Curve")
    print("π_t = α - β * u_t + ε_t")
    print("=" * 60)

    X = df["unemployment_rate"].values.reshape(-1, 1)
    y = df["inflation_rate"].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot)

    # Calculate standard errors manually
    n = len(y)
    p = 2  # Number of parameters
    mse = ss_res / (n - p)
    X_with_const = np.column_stack([np.ones(n), X.ravel()])
    cov_matrix = mse * np.linalg.inv(X_with_const.T @ X_with_const)
    std_err = np.sqrt(np.diag(cov_matrix))

    # Calculate t-statistics and p-values
    t_stats = np.array([model.coef_[0], model.intercept_]) / std_err
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - 2))

    results = {
        "specification": "Linear Phillips Curve",
        "equation": "π_t = α - β * u_t + ε_t",
        "alpha": model.intercept_,
        "alpha_std": std_err[1],
        "alpha_pvalue": p_values[1],
        "beta": -model.coef_[0],
        "beta_std": std_err[0],
        "beta_pvalue": p_values[0],
        "r_squared": r_squared,
        "adj_r_squared": 1 - (1 - r_squared) * (n - 1) / (n - p),
        "n_obs": len(df),
    }

    print(
        f"α (intercept): {results['alpha']:.4f} ± {results['alpha_std']:.4f} (p={results['alpha_pvalue']:.4f})"
    )
    print(
        f"β (slope): {results['beta']:.4f} ± {results['beta_std']:.4f} (p={results['beta_pvalue']:.4f})"
    )
    print(f"R-squared: {results['r_squared']:.4f}")
    print(f"Adjusted R-squared: {results['adj_r_squared']:.4f}")

    return results


def estimate_accelerationist_phillips(df):
    """
    Estimate accelerationist Phillips Curve: Δπ_t = α - β * u_t + ε_t

    This specification says inflation acceleration depends on unemployment level.

    Returns:
        dict: Results
    """
    print("\n" + "=" * 60)
    print("SPECIFICATION 2: Accelerationist Phillips Curve")
    print("Δπ_t = α - β * u_t + ε_t")
    print("=" * 60)

    # Create lagged inflation
    df_temp = df.copy()
    df_temp["inflation_rate_lag1"] = df_temp["inflation_rate"].shift(1)
    df_temp = df_temp.dropna()

    X = df_temp["unemployment_rate"].values.reshape(-1, 1)
    y = (df_temp["inflation_rate"] - df_temp["inflation_rate_lag1"]).values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot)

    n = len(y)
    p = 2
    mse = ss_res / (n - p)
    X_with_const = np.column_stack([np.ones(n), X.ravel()])
    cov_matrix = mse * np.linalg.inv(X_with_const.T @ X_with_const)
    std_err = np.sqrt(np.diag(cov_matrix))

    t_stats = np.array([model.coef_[0], model.intercept_]) / std_err
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - 2))

    results = {
        "specification": "Accelerationist Phillips Curve",
        "equation": "Δπ_t = α - β * u_t + ε_t",
        "alpha": model.intercept_,
        "alpha_std": std_err[1],
        "alpha_pvalue": p_values[1],
        "beta": -model.coef_[0],
        "beta_std": std_err[0],
        "beta_pvalue": p_values[0],
        "r_squared": r_squared,
        "adj_r_squared": 1 - (1 - r_squared) * (n - 1) / (n - p),
        "n_obs": len(df_temp),
    }

    print(
        f"α (intercept): {results['alpha']:.4f} ± {results['alpha_std']:.4f} (p={results['alpha_pvalue']:.4f})"
    )
    print(
        f"β (slope): {results['beta']:.4f} ± {results['beta_std']:.4f} (p={results['beta_pvalue']:.4f})"
    )
    print(f"R-squared: {results['r_squared']:.4f}")
    print(f"Adjusted R-squared: {results['adj_r_squared']:.4f}")

    return results


def estimate_nonlinear_phillips(df):
    """
    Estimate non-linear Phillips Curve: π_t = α - β * u_t + γ * u_t^2 + ε_t

    Returns:
        dict: Results
    """
    print("\n" + "=" * 60)
    print("SPECIFICATION 3: Non-Linear (Quadratic) Phillips Curve")
    print("π_t = α - β * u_t + γ * u_t^2 + ε_t")
    print("=" * 60)

    X = np.column_stack(
        [df["unemployment_rate"].values, (df["unemployment_rate"] ** 2).values]
    )
    y = df["inflation_rate"].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1 - (ss_res / ss_tot)

    n = len(y)
    p = 3
    mse = ss_res / (n - p)
    X_with_const = np.column_stack([np.ones(n), X])
    cov_matrix = mse * np.linalg.inv(X_with_const.T @ X_with_const)
    std_err = np.sqrt(np.diag(cov_matrix))

    t_stats = np.array([model.coef_[0], model.coef_[1], model.intercept_]) / std_err
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - 3))

    results = {
        "specification": "Non-Linear Phillips Curve",
        "equation": "π_t = α - β * u_t + γ * u_t^2 + ε_t",
        "alpha": model.intercept_,
        "alpha_std": std_err[2],
        "alpha_pvalue": p_values[2],
        "beta": -model.coef_[0],
        "beta_std": std_err[0],
        "beta_pvalue": p_values[0],
        "gamma": model.coef_[1],
        "gamma_std": std_err[1],
        "gamma_pvalue": p_values[1],
        "r_squared": r_squared,
        "adj_r_squared": 1 - (1 - r_squared) * (n - 1) / (n - p),
        "n_obs": len(df),
    }

    print(
        f"α (intercept): {results['alpha']:.4f} ± {results['alpha_std']:.4f} (p={results['alpha_pvalue']:.4f})"
    )
    print(
        f"β (linear slope): {results['beta']:.4f} ± {results['beta_std']:.4f} (p={results['beta_pvalue']:.4f})"
    )
    print(
        f"γ (quadratic term): {results['gamma']:.4f} ± {results['gamma_std']:.4f} (p={results['gamma_pvalue']:.4f})"
    )
    print(f"R-squared: {results['r_squared']:.4f}")
    print(f"Adjusted R-squared: {results['adj_r_squared']:.4f}")

    return results


def calculate_nairu(df):
    """
    Estimate NAIRU (Non-Accelerating Inflation Rate of Unemployment)

    Using simple average method.

    Returns:
        float: Estimated NAIRU
    """
    print("\n" + "=" * 60)
    print("NAIRU ESTIMATION")
    print("=" * 60)

    # Method: Simple average of unemployment over period
    nairu = df["unemployment_rate"].mean()
    print(f"NAIRU (sample average): {nairu:.2f}%")

    return nairu


def save_results(results_list, df):
    """Save results to CSV."""
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Convert results to DataFrame
    df_results = pd.DataFrame(results_list)

    # Save detailed results
    output_path = "data/phillips_curve_results.csv"
    df_results.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Also save the merged data for reference
    data_path = "data/phillips_curve_data.csv"
    df.to_csv(data_path)
    print(f"Data saved to: {data_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(
        df_results[
            ["specification", "beta", "beta_pvalue", "r_squared", "adj_r_squared"]
        ].to_string(index=False)
    )


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Estimate Phillips Curve for KBR")
    parser.add_argument("--output-dir", default="data/", help="Output directory")
    args = parser.parse_args()

    print("=" * 60)
    print("PHILLIPS CURVE ESTIMATION FOR KABARDINO-BALKARIA")
    print("=" * 60)

    # Load data
    df_unemp = load_unemployment_data()
    df_inf = load_inflation_data()

    # Merge data
    df = merge_data(df_unemp, df_inf)

    # Estimate NAIRU
    nairu = calculate_nairu(df)

    # Estimate different specifications
    results_list = []

    results_list.append(estimate_linear_phillips(df))
    results_list.append(estimate_accelerationist_phillips(df.copy()))
    results_list.append(estimate_nonlinear_phillips(df))

    # Add NAIRU to results
    for res in results_list:
        res["nairu"] = nairu
        res["output_dir"] = args.output_dir

    # Save results
    save_results(results_list, df)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
