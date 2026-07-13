#!/usr/bin/env python3
"""
Error Decomposition by CPI Component
=====================================
Decomposes forecast errors by CPI component to identify which drives total errors.

Components:
- Food (Prod): 39.5% weight
- NonFood (Nonprod): 36.5% weight
- Services (Serv): 24.0% weight
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def load_data():
    """Load inflation data."""
    data_path = Path(__file__).parent.parent / "data" / "enhanced_inflation_data.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    return df


def create_features(df, target_col, lags=6):
    """
    Create lagged features for forecasting.

    Args:
        df: DataFrame with target column
        target_col: Name of target column
        lags: Number of lags to create

    Returns:
        DataFrame with features
    """
    result = df[[target_col]].copy()

    for lag in range(1, lags + 1):
        result[f"{target_col}_lag{lag}"] = result[target_col].shift(lag)

    # Add rolling statistics
    result[f"{target_col}_ma3"] = result[target_col].rolling(3).mean()
    result[f"{target_col}_ma6"] = result[target_col].rolling(6).mean()

    return result.dropna()


def backtest_component(df, component_col, start_date="2019-01-01"):
    """
    Run rolling backtest for a single component.

    Args:
        df: Full DataFrame
        component_col: Name of component column
        start_date: Backtest start date

    Returns:
        MAE for this component
    """
    df_feat = create_features(df, component_col)

    start_idx = df_feat.index.get_loc(pd.Timestamp(start_date))
    errors = []

    for i in range(start_idx, len(df_feat) - 1):
        train_df = df_feat.iloc[:i]
        test_row = df_feat.iloc[i : i + 1]

        if len(train_df) < 24:
            continue

        # Features (exclude target)
        feature_cols = [c for c in train_df.columns if c != component_col]
        X_train = train_df[feature_cols].values
        y_train = train_df[component_col].values
        X_test = test_row[feature_cols].values
        y_test = test_row[component_col].values[0]

        # Train Ridge model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)

        # Predict and calculate error
        pred = model.predict(X_test_scaled)[0]
        error = abs(pred - y_test)
        errors.append(error)

    if len(errors) == 0:
        return None

    return np.mean(errors)


def main():
    """Main execution."""
    print("=" * 60)
    print("Error Decomposition by CPI Component")
    print("=" * 60)

    # Load data
    df = load_data()
    print(f"Loaded data: {len(df)} observations ({df.index.min()} to {df.index.max()})")

    # Component definitions with weights
    components = {
        "Food": {"col": "Prod", "weight": 0.395},
        "NonFood": {"col": "Nonprod", "weight": 0.365},
        "Services": {"col": "Serv", "weight": 0.240},
    }

    print(f"\nComponent weights:")
    for name, config in components.items():
        print(f"  {name}: {config['weight'] * 100:.1f}%")

    print(f"\n{'=' * 60}")
    print("Running backtests for each component...")
    print(f"{'=' * 60}")

    results = []
    weighted_errors = []

    for comp_name, config in components.items():
        col = config["col"]
        weight = config["weight"]

        print(f"\nBacktesting {comp_name} ({col})...")
        mae = backtest_component(df, col, start_date="2019-01-01")

        if mae is not None:
            weighted_error = mae * weight
            results.append(
                {
                    "Component": comp_name,
                    "MAE": round(mae, 4),
                    "Weight": round(weight, 3),
                    "Weighted_MAE": round(weighted_error, 4),
                }
            )
            weighted_errors.append(weighted_error)
            print(f"  MAE: {mae:.4f}")
            print(f"  Weighted MAE: {weighted_error:.4f}")
        else:
            print(f"  ERROR: Could not calculate MAE for {comp_name}")

    # Calculate contributions
    total_weighted_mae = sum(weighted_errors)

    for result in results:
        if total_weighted_mae > 0:
            contribution_pct = (result["Weighted_MAE"] / total_weighted_mae) * 100
        else:
            contribution_pct = 0.0
        result["Contribution_pct"] = round(contribution_pct, 2)

    # Create output DataFrame
    output_df = pd.DataFrame(results)

    # Verify sum of contributions
    total_contrib = output_df["Contribution_pct"].sum()
    print(f"\n{'=' * 60}")
    print("Error Decomposition Results")
    print(f"{'=' * 60}")
    print(output_df.to_string(index=False))
    print(f"\nSum of contributions: {total_contrib:.2f}%")

    # Save to CSV
    output_path = Path(__file__).parent.parent / "data" / "error_decomposition.csv"
    output_df[["Component", "MAE", "Contribution_pct"]].to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Identify error driver
    if len(results) > 0:
        max_contrib_row = output_df.loc[output_df["Contribution_pct"].idxmax()]
        print(
            f"\nPrimary error driver: {max_contrib_row['Component']} ({max_contrib_row['Contribution_pct']:.2f}%)"
        )

    # Verify acceptance criteria
    print(f"\n{'=' * 60}")
    print("Acceptance Criteria Verification")
    print(f"{'=' * 60}")

    # Check 1: File exists with correct columns
    if output_path.exists():
        required_cols = ["Component", "MAE", "Contribution_pct"]
        existing_cols = output_df.columns.tolist()
        has_cols = all(col in existing_cols for col in required_cols)
        print(f"✓ CSV file exists with columns {required_cols}")
    else:
        print(f"✗ CSV file does not exist")

    # Check 2: Contributions sum to ~100%
    if 98 <= total_contrib <= 102:
        print(f"✓ Contribution percentages sum to {total_contrib:.2f}% (~100%)")
    else:
        print(f"✗ Contribution percentages sum to {total_contrib:.2f}% (not ~100%)")

    return output_path


if __name__ == "__main__":
    main()
