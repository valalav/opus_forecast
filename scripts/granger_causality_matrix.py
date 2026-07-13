#!/usr/bin/env python3
"""
Granger Causality Matrix for KBR CPI Microcomponents

Tests Granger causality between all pairs of top 20 CPI microcomponents
to identify leading indicators and causal relationships.

Uses micro_sprav.csv (538 items) and kbr_micro_full.csv (long format).
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

warnings.filterwarnings("ignore")


def load_micro_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load microcomponent data and reference information.

    Args:
        data_dir: Path to data directory

    Returns:
        Tuple of (mom_data_wide, reference_df)
    """
    data_path = Path(data_dir)

    # Load micro MoM data (long format)
    mom_df = pd.read_csv(
        data_path / "kbr_micro_full.csv",
    )

    # Filter to KBR region (Region_code=7)
    mom_df = mom_df[mom_df["Region_code"] == 7].copy()

    # The Date column contains monthly dates (e.g., 2024-01-01)
    # but there are daily observations (Day column has actual days)
    # Aggregate daily observations to monthly (take last observation)
    mom_df = mom_df.groupby(["Date", "Item_code"]).last().reset_index()

    # Convert Date to datetime
    mom_df["Date"] = pd.to_datetime(mom_df["Date"], dayfirst=True)

    # Pivot to wide format (Date index, Item_code columns)
    mom_wide = mom_df.pivot(index="Date", columns="Item_code", values="MoM")

    # Load micro reference (weights and names)
    ref_df = pd.read_csv(
        data_path / "micro_sprav.csv",
        sep=";",
    )

    # Clean up Item_code column (remove BOM if present)
    ref_df["Item_code"] = ref_df["Item_code"].astype(str).str.strip()

    # Parse weight as float (replace comma with dot)
    ref_df["Weight"] = (
        ref_df["Weight"].astype(str).str.replace(",", ".", regex=False).astype(float)
    )

    return mom_wide, ref_df


def get_top_microcomponents(ref_df: pd.DataFrame, n: int = 20) -> List[Dict]:
    """
    Get top N microcomponents by weight.

    Args:
        ref_df: Reference dataframe with weights
        n: Number of top components to return

    Returns:
        List of dicts with component info
    """
    # Sort by weight descending and get top N
    top = ref_df.nlargest(n, "Weight")

    components = []
    for _, row in top.iterrows():
        components.append(
            {
                "code": int(row["Item_code"]),
                "name": row["Товар"],
                "weight": row["Weight"],
                "component": row.get("Компонент", "Unknown"),
                "subcomponent": row.get("Субкомпонент", "Unknown"),
            }
        )

    return components


def prepare_data(mom_df: pd.DataFrame, components: List[Dict]) -> pd.DataFrame:
    """
    Prepare data matrix with selected microcomponents.

    Args:
        mom_df: MoM data in wide format (Date index, Item_code columns)
        components: List of component info dicts

    Returns:
        DataFrame with Date index and component columns
    """
    # Filter to top components
    selected_codes = [c["code"] for c in components if c["code"] in mom_df.columns]

    if len(selected_codes) < len(components):
        print(
            f"WARNING: Only {len(selected_codes)}/{len(components)} components have data"
        )

    # Extract data for selected components
    data = mom_df[selected_codes].copy()

    # Convert to numeric (already should be, but just in case)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Remove columns with >50% missing values (relaxed further)
    missing_pct = data.isnull().mean()
    valid_cols = missing_pct[missing_pct < 0.50].index.tolist()
    data = data[valid_cols]

    if len(valid_cols) < len(selected_codes):
        print(f"INFO: After removing >50% missing: {len(valid_cols)} columns remain")

    # Interpolate remaining missing values
    data = data.interpolate(method="time", limit_direction="both")

    # Forward fill any remaining NaNs
    data = data.ffill().bfill()

    return data


def make_stationary(series: pd.Series) -> pd.Series:
    """
    Make series stationary by differencing if needed.

    Uses Augmented Dickey-Fuller test to check stationarity.
    """
    if len(series.dropna()) < 20:
        return series

    try:
        # ADF test
        result = adfuller(series.dropna())
        p_value = result[1]

        # If not stationary (p > 0.05), difference once
        if p_value > 0.05:
            diff_series = series.diff().dropna()
            return diff_series
        else:
            return series
    except:
        return series.diff().dropna()


def granger_test_pair(
    x: pd.Series, y: pd.Series, maxlag: int = 2, verbose: bool = False
) -> float:
    """
    Run Granger causality test for a single pair.

    H0: x does NOT Granger-cause y
    If p-value < 0.05, reject H0 → x Granger-causes y

    Args:
        x: Potential cause series
        y: Effect series
        maxlag: Maximum lag to test
        verbose: Print test output

    Returns:
        p-value from F-test
    """
    # Align series
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 15:  # Need minimum observations (relaxed from 20)
        return np.nan

    # Make stationary
    df_x = make_stationary(df.iloc[:, 0])
    df_y = make_stationary(df.iloc[:, 1])

    # Realign after differencing
    df_aligned = pd.concat([df_x, df_y], axis=1).dropna()

    if len(df_aligned) < 15:
        return np.nan

    try:
        # Run Granger test
        result = grangercausalitytests(
            df_aligned.values, maxlag=maxlag, verbose=verbose
        )

        # Extract F-test p-value for maxlag
        # result is a dict: {1: {...}, 2: {...}}
        # Each entry has 'ssr_ftest', 'lrtest', 'params_ftest'
        p_value = result[maxlag][0]["ssr_ftest"][1]

        return p_value
    except Exception as e:
        if verbose:
            print(f"Error in Granger test: {e}")
        return np.nan


def build_granger_matrix(
    data: pd.DataFrame, components: List[Dict], maxlag: int = 2
) -> pd.DataFrame:
    """
    Build Granger causality matrix for all component pairs.

    Matrix[i, j] = p-value for test: does component i cause component j?

    Args:
        data: DataFrame with component columns
        components: List of component info
        maxlag: Maximum lag for Granger test

    Returns:
        DataFrame with p-value matrix
    """
    # Get codes that exist in data
    available_codes = [c["code"] for c in components if c["code"] in data.columns]
    available_names = [
        next((c["name"] for c in components if c["code"] == code), str(code))
        for code in available_codes
    ]

    n_components = len(available_codes)
    codes = [str(c) for c in available_codes]
    names = available_names

    # Initialize matrix
    matrix = pd.DataFrame(index=codes, columns=codes, dtype=float)

    print(f"\nBuilding {n_components}x{n_components} Granger causality matrix...")
    print(f"Total tests: {n_components * (n_components - 1)}\n")

    # Test all pairs (excluding self)
    for i, cause_code in enumerate(codes):
        for j, effect_code in enumerate(codes):
            if i == j:
                matrix.loc[cause_code, effect_code] = np.nan
                continue

            # Run test
            x = data[int(cause_code)]
            y = data[int(effect_code)]

            p_value = granger_test_pair(x, y, maxlag=maxlag)
            matrix.loc[cause_code, effect_code] = p_value

            # Progress indicator
            if (i * n_components + j) % 50 == 0:
                print(
                    f"Progress: {i * n_components + j}/{n_components * n_components} tests"
                )

    return matrix, codes, names


def identify_leading_components(
    matrix: pd.DataFrame, alpha: float = 0.05
) -> List[Dict]:
    """
    Identify components that cause many others.

    Args:
        matrix: P-value matrix
        alpha: Significance threshold

    Returns:
        List of components with statistics
    """
    leading = []

    for cause in matrix.index:
        # Count how many components this one causes
        significant = matrix.loc[cause, :] < alpha
        n_causes = significant.sum()
        avg_p = matrix.loc[cause, :].mean()

        leading.append(
            {
                "code": cause,
                "n_causes": n_causes,
                "avg_p_value": avg_p,
                "causes_list": matrix.columns[significant].tolist(),
            }
        )

    # Sort by number of causes
    leading = sorted(leading, key=lambda x: x["n_causes"], reverse=True)

    return leading


def main():
    parser = argparse.ArgumentParser(
        description="Build Granger causality matrix for KBR CPI microcomponents"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Path to data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/granger_matrix.csv",
        help="Output path for p-value matrix",
    )
    parser.add_argument(
        "--top-n", type=int, default=20, help="Number of top components to analyze"
    )
    parser.add_argument(
        "--maxlag", type=int, default=1, help="Maximum lag for Granger test"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="Significance threshold"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2016,
        help="Start year for analysis (post-Crimea period)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Granger Causality Matrix for KBR CPI Microcomponents")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    mom_df, ref_df = load_micro_data(args.data_dir)
    print(f"   MoM data shape (wide): {mom_df.shape}")

    # Filter by date range
    start_date = pd.Timestamp(f"{args.start_year}-01-01")
    mom_df = mom_df[mom_df.index >= start_date]
    print(f"   Filtered to >= {start_date}: {mom_df.shape}")
    print(f"   Reference rows: {len(ref_df)}")

    # Get top components
    print(f"\n2. Selecting top {args.top_n} microcomponents by weight...")
    components = get_top_microcomponents(ref_df, n=args.top_n)
    for i, c in enumerate(components[:10]):
        print(
            f"   {i + 1:2d}. {c['code']:3d}: {c['name'][:40]:40s} (weight: {c['weight']:.4f})"
        )
    if len(components) > 10:
        print(f"   ... and {len(components) - 10} more")

    # Prepare data
    print("\n3. Preparing data...")
    data = prepare_data(mom_df, components)
    print(f"   Prepared data shape: {data.shape}")
    print(f"   Date range: {data.index.min()} to {data.index.max()}")
    print(f"   Columns (microcomponents): {data.columns.tolist()}")

    if data.shape[1] < args.top_n:
        print(
            f"\nWARNING: Only {data.shape[1]} components available (requested {args.top_n})"
        )

    # Build matrix
    print("\n4. Running Granger causality tests...")
    matrix, codes, names = build_granger_matrix(data, components, maxlag=args.maxlag)

    # Save matrix
    print("\n5. Saving results...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with component codes as row/column labels
    matrix.to_csv(output_path)
    print(f"   Matrix saved to: {output_path}")

    # Create mapping file (code -> name)
    mapping_df = pd.DataFrame(
        {
            "code": codes,
            "name": names,
        }
    )
    mapping_path = output_path.parent / "granger_matrix_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)
    print(f"   Code->name mapping saved to: {mapping_path}")

    # Identify leading components
    print("\n6. Identifying leading components...")
    leading = identify_leading_components(matrix, alpha=args.alpha)

    print("\n" + "=" * 70)
    print("LEADING MICROCOMPONENTS (by number of components they cause)")
    print("=" * 70)
    for i, c in enumerate(leading[:10]):
        name = next(
            (comp["name"] for comp in components if comp["code"] == int(c["code"])),
            c["code"],
        )
        print(f"\n{i + 1}. {name}")
        print(f"   Code: {c['code']}")
        print(f"   Causes: {c['n_causes']} components (p < {args.alpha})")
        print(f"   Avg p-value: {c['avg_p_value']:.4f}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    n_significant = (matrix < args.alpha).sum().sum()
    n_total = matrix.count().sum()
    pct_significant = 100 * n_significant / n_total if n_total > 0 else 0

    print(f"Matrix size: {len(matrix)}x{len(matrix)}")
    print(f"Total tests run: {n_total}")
    print(f"Significant (p < {args.alpha}): {n_significant} ({pct_significant:.1f}%)")

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
