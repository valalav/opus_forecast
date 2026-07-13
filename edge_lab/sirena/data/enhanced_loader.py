#!/usr/bin/env python3
"""
Data loader that integrates OPR (Official Price Reporting) features into inflation data.

This script creates an enhanced dataset that combines:
- Base inflation data (mom, macro features)
- Monthly OPR-based features from regressor_priority_list.csv (Task 116)

For use with OPR-enhanced models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent sirena to path for model imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "sirena"))


def load_base_inflation_data():
    """
    Load base inflation data from main data file.

    Returns:
        DataFrame with columns: Date, mom, Prod, Serv, Nonprod, Ki_i, Ruonia, usd_nom_i, brent
    """
    # Data is in parent directory
    data_path = Path("../data/inflation_data.csv")

    if not data_path.exists():
        raise FileNotFoundError(f"Base data not found: {data_path}")

    df = pd.read_csv(data_path, sep=";")

    # Clean column names (strip whitespace)
    df.columns = [col.strip() for col in df.columns]

    # Convert date format from DD.MM.YYYY to datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")

    # Convert numeric columns from comma decimal to period decimal
    numeric_cols = [
        "mom",
        "Prod",
        "Serv",
        "Nonprod",
        "Ki_i",
        "Ruonia",
        "usd_nom_i",
        "brent",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

    # Normalize to month-start dates for consistency with OPR data
    df["Date"] = df["Date"].apply(lambda x: x.replace(day=1))

    df = df.set_index("Date")
    df = df.sort_index()

    return df


def load_monthly_opr_features(top_n=5):
    """
    Load MONTHLY (м/м) OPR features from regressor_priority_list.csv.

    Uses monthly features to avoid look-ahead bias in backtesting.

    Args:
        top_n: Number of top monthly features to load

    Returns:
        DataFrame with OPR features as columns, indexed by Date
    """
    priority_path = Path("data/regressor_priority_list.csv")

    if not priority_path.exists():
        raise FileNotFoundError(f"Priority list not found: {priority_path}")

    sectoral_path = Path("data/test_kbr_sectoral.csv")

    if not sectoral_path.exists():
        raise FileNotFoundError(f"Sectoral data not found: {sectoral_path}")

    # Load priority list
    priority_df = pd.read_csv(priority_path)

    # Filter for monthly features ONLY (avoid YoY CPI which is the target)
    monthly_features_df = priority_df[
        priority_df["feature_id"].str.contains("::м/м::", na=False)
    ]

    if len(monthly_features_df) == 0:
        raise ValueError("No monthly features found in priority list")

    # Get top N monthly features
    top_monthly = monthly_features_df.head(top_n)

    # Load sectoral data
    sectoral_df = pd.read_csv(sectoral_path)
    sectoral_df["Date"] = pd.to_datetime(sectoral_df["Date"])

    opr_features = []

    for _, row in top_monthly.iterrows():
        feature_id = row["feature_id"]

        # Parse feature_id: "Indicator::Metric_Type::Sheet"
        parts = feature_id.split("::")
        indicator = parts[0]
        metric_type = parts[1]
        sheet = int(parts[2])

        # Find matching data in sectoral_df
        matches = sectoral_df[
            (sectoral_df["Indicator"] == indicator)
            & (sectoral_df["Metric_Type"] == metric_type)
            & (sectoral_df["Sheet"] == sheet)
        ].copy()

        if len(matches) == 0:
            print(f"Warning: No data found for {feature_id}")
            continue

        # Rename for clarity - create clean column name
        clean_name = f"opr_{feature_id.replace(';', '_').replace(' ', '_').replace('::', '_').replace('/', '_')}"
        clean_name = clean_name[:60]  # Limit length

        matches = matches[["Date", "Value"]].copy()
        # Deduplicate: keep first value per date
        matches = matches.groupby("Date")["Value"].first().reset_index()
        matches = matches.rename(columns={"Value": clean_name})

        opr_features.append(matches)

    if not opr_features:
        raise ValueError("No OPR features found in data")

    # Merge all OPR features
    result = opr_features[0]
    for feat_df in opr_features[1:]:
        result = pd.merge(result, feat_df, on="Date", how="outer")

    result = result.set_index("Date")
    result = result.sort_index()

    return result


def load_enhanced_data(top_opr_features=5):
    """
    Load enhanced inflation data with monthly OPR features.

    Args:
        top_opr_features: Number of top OPR features to include

    Returns:
        DataFrame with all columns merged and indexed by Date
    """
    print("Loading base inflation data...")
    base_df = load_base_inflation_data()
    print(f"  - Base data: {len(base_df)} rows, {len(base_df.columns)} columns")

    print(f"\nLoading top {top_opr_features} monthly OPR features...")
    opr_df = load_monthly_opr_features(top_n=top_opr_features)
    print(f"  - OPR features: {len(opr_df)} rows, {len(opr_df.columns)} columns")
    print(f"  - OPR columns: {', '.join(opr_df.columns.tolist())}")

    print("\nMerging datasets...")
    # Merge on index (Date)
    enhanced_df = pd.merge(
        base_df, opr_df, left_index=True, right_index=True, how="outer"
    )
    enhanced_df = enhanced_df.sort_index()

    print(
        f"  - Merged data: {len(enhanced_df)} rows, {len(enhanced_df.columns)} columns"
    )

    # Report on OPR feature data coverage
    print("\nOPR Feature Coverage:")
    for col in opr_df.columns:
        non_null_pct = (enhanced_df[col].notna().sum() / len(enhanced_df)) * 100
        print(f"  - {col}: {non_null_pct:.1f}% coverage")

    return enhanced_df


def save_enhanced_data(df, output_path=None):
    """
    Save enhanced data to CSV.

    Args:
        df: Enhanced DataFrame
        output_path: Output file path (default: data/enhanced_inflation_data_monthly.csv)
    """
    if output_path is None:
        output_path = Path("data/enhanced_inflation_data_monthly.csv")
    else:
        output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path)
    print(f"\nSaved enhanced data to: {output_path}")

    return output_path


def main():
    """Main execution function."""
    print("=" * 60)
    print("Enhanced Data Loader - Monthly OPR Feature Integration")
    print("=" * 60)

    # Load enhanced data
    enhanced_df = load_enhanced_data(top_opr_features=5)

    # Save to file
    output_path = save_enhanced_data(enhanced_df)

    # Display sample data
    print("\n" + "=" * 60)
    print("Sample Data (last 5 rows)")
    print("=" * 60)
    print(enhanced_df.tail().to_string())

    print("\n✅ Enhanced data creation completed")

    return enhanced_df


if __name__ == "__main__":
    main()
