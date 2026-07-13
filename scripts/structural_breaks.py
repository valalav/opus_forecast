#!/usr/bin/env python3
"""
Structural Breaks Detection for KBR Inflation

Detects structural breaks in KBR inflation time series using:
- Bai-Perron test (multiple unknown breakpoints)
- Chow test (known potential breakpoints)

Outputs detected break dates with confidence levels.
"""

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")


def load_inflation_data(data_path: str) -> pd.DataFrame:
    """
    Load inflation data and prepare time series.

    Args:
        data_path: Path to inflation_data.csv

    Returns:
        DataFrame with Date index and mom column (MoM inflation - 100)
    """
    df = pd.read_csv(
        data_path,
        sep=";",
        decimal=",",
        parse_dates=["Date"],
        dayfirst=True,
    )
    df = df.sort_values("Date").reset_index(drop=True)

    # Convert MoM from 100.xx format to percent (e.g., 101.49 -> 1.49)
    df["mom_pct"] = df["mom"] - 100

    return df


def calculate_chow_statistic(
    y: np.ndarray, X: np.ndarray, breakpoint: int
) -> Tuple[float, float, float]:
    """
    Calculate Chow test statistic for structural break at specific point.

    Args:
        y: Dependent variable
        X: Independent variables (with constant)
        breakpoint: Index of potential break point

    Returns:
        Tuple of (F_statistic, p_value, critical_value)
    """
    n = len(y)

    # Fitted model on full sample
    coef_full, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals_full = y - X @ coef_full
    sse_full = np.sum(residuals_full**2)

    # Fitted models on subsamples
    y1 = y[:breakpoint]
    X1 = X[:breakpoint]
    y2 = y[breakpoint:]
    X2 = X[breakpoint:]

    coef1, _, _, _ = np.linalg.lstsq(X1, y1, rcond=None)
    coef2, _, _, _ = np.linalg.lstsq(X2, y2, rcond=None)

    residuals1 = y1 - X1 @ coef1
    residuals2 = y2 - X2 @ coef2
    sse_split = np.sum(residuals1**2) + np.sum(residuals2**2)

    # Chow F-statistic
    k = X.shape[1]  # Number of parameters
    numerator = (sse_full - sse_split) / k
    denominator = sse_split / (n - 2 * k)

    F_stat = numerator / denominator if denominator > 0 else 0

    # Degrees of freedom
    df1 = k
    df2 = n - 2 * k

    # P-value
    p_value = 1 - stats.f.cdf(F_stat, df1, df2)

    # Critical value at 5% significance
    critical_value = stats.f.ppf(0.95, df1, df2)

    return F_stat, p_value, critical_value


def chow_test_breakpoints(
    df: pd.DataFrame, candidate_dates: List[str] = None
) -> List[Dict]:
    """
    Perform Chow test at candidate break dates.

    Args:
        df: DataFrame with Date index and mom_pct column
        candidate_dates: List of candidate break dates (YYYY-MM-DD format)

    Returns:
        List of dicts with test results
    """
    results = []

    if candidate_dates is None:
        candidate_dates = [
            "2014-12-01",  # Currency crisis
            "2015-12-01",  # Post-crisis recovery
            "2020-03-01",  # COVID start
            "2022-02-01",  # Sanctions start
        ]

    y = df["mom_pct"].values
    # Add trend term
    X = np.column_stack([np.ones(len(y)), np.arange(len(y))])

    for date_str in candidate_dates:
        try:
            break_date = pd.to_datetime(date_str)
            # Find closest date in data
            if break_date not in df["Date"].values:
                # Get nearest date
                idx = (df["Date"] - break_date).abs().idxmin()
            else:
                idx = df[df["Date"] == break_date].index[0]

            F_stat, p_value, critical_value = calculate_chow_statistic(y, X, idx)
        except Exception as e:
            print(f"Warning: Could not test date {date_str}: {e}")

    return results


def detect_bai_perron_breaks(
    y: np.ndarray, max_breaks: int = 5, min_obs_between: int = 24
) -> List[int]:
    """
    Detect multiple structural breakpoints using Bai-Perron approach.

    Uses a simplified version based on minimizing RSS.

    Args:
        y: Time series
        max_breaks: Maximum number of breaks to detect
        min_obs_between: Minimum observations between breaks

    Returns:
        List of breakpoint indices
    """
    n = len(y)
    breakpoints = []

    # Trend component
    X = np.column_stack([np.ones(n), np.arange(n)])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    trend = X @ coef
    detrended = y - trend

    # Dynamic programming to find best breakpoints
    def calculate_rss(segment_start: int, segment_end: int) -> float:
        """Calculate RSS for a segment."""
        segment = y[segment_start:segment_end]
        if len(segment) == 0:
            return float("inf")
        mean_segment = np.mean(segment)
        return np.sum((segment - mean_segment) ** 2)

    # Greedy approach: iteratively add breakpoints
    remaining_obs = list(range(min_obs_between, n - min_obs_between))
    total_rss = calculate_rss(0, n)

    for _ in range(max_breaks):
        best_break = None
        best_reduction = 0

        for candidate in remaining_obs:
            if any(abs(candidate - bp) < min_obs_between for bp in breakpoints):
                continue

            # Calculate RSS with new breakpoint
            candidate_breaks = sorted(breakpoints + [candidate])
            segments = [0] + candidate_breaks + [n]
            new_rss = sum(
                calculate_rss(segments[i], segments[i + 1])
                for i in range(len(segments) - 1)
            )

            reduction = total_rss - new_rss
            if reduction > best_reduction:
                best_reduction = reduction
                best_break = candidate

        if best_break is not None and best_reduction / total_rss > 0.05:
            breakpoints.append(best_break)
            total_rss -= best_reduction
        else:
            break

    return sorted(breakpoints)


def bai_perron_analysis(df: pd.DataFrame, max_breaks: int = 5) -> List[Dict]:
    """
    Perform Bai-Perron structural break analysis.

    Args:
        df: DataFrame with Date index and mom_pct column
        max_breaks: Maximum number of breaks to detect

    Returns:
        List of dicts with break information
    """
    results = []
    y = df["mom_pct"].values

    # Detect breakpoints
    breakpoints = detect_bai_perron_breaks(y, max_breaks=max_breaks)

    # Calculate statistics for each break
    for i, bp_idx in enumerate(breakpoints):
        break_date = df.iloc[bp_idx]["Date"]

        # Calculate mean and variance before/after break
        pre_break = y[:bp_idx]
        post_break = y[bp_idx:]

        mean_before = np.mean(pre_break)
        mean_after = np.mean(post_break)
        std_before = np.std(pre_break)
        std_after = np.std(post_break)

        # T-test for mean shift
        t_stat, p_value = stats.ttest_ind(pre_break, post_break)

        results.append(
            {
                "break_date": break_date.strftime("%Y-%m-%d"),
                "test_type": "Bai-Perron",
                "break_index": bp_idx,
                "mean_before": round(mean_before, 4),
                "mean_after": round(mean_after, 4),
                "mean_shift": round(mean_after - mean_before, 4),
                "std_before": round(std_before, 4),
                "std_after": round(std_after, 4),
                "std_change": round(std_after - std_before, 4),
                "t_statistic": round(t_stat, 4),
                "p_value": round(p_value, 6),
                "significant": p_value < 0.05,
                "description": f"Breakpoint {i + 1} detected via Bai-Perron",
            }
        )

    return results


def detect_variance_breaks(df: pd.DataFrame, window: int = 12) -> List[Dict]:
    """
    Detect volatility/structural breaks using rolling variance.

    Args:
        df: DataFrame with Date index and mom_pct column
        window: Rolling window size for variance

    Returns:
        List of dicts with variance break information
    """
    results = []
    y = df["mom_pct"].values

    # Calculate rolling variance
    rolling_var = pd.Series(y).rolling(window=window).var()

    # Threshold for variance breaks (3 sigma)
    var_mean = rolling_var.mean()
    var_std = rolling_var.std()
    threshold = var_mean + 3 * var_std

    for idx, var in enumerate(rolling_var):
        if pd.notna(var) and var > threshold:
            if idx >= window:
                break_date = df.iloc[idx]["Date"]
                results.append(
                    {
                        "break_date": break_date.strftime("%Y-%m-%d"),
                        "test_type": "Variance",
                        "variance": round(var, 4),
                        "threshold": round(threshold, 4),
                        "description": f"Variance spike detected (window={window})",
                    }
                )

    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Detect structural breaks in KBR inflation series"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/inflation_data.csv",
        help="Path to inflation data CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/structural_breaks.csv",
        help="Path to output CSV",
    )
    parser.add_argument(
        "--max-breaks",
        type=int,
        default=5,
        help="Maximum number of breaks for Bai-Perron",
    )
    parser.add_argument(
        "--candidate-dates",
        type=str,
        nargs="+",
        default=None,
        help="Candidate break dates for Chow test (YYYY-MM-DD format)",
    )

    args = parser.parse_args()

    print(f"Loading inflation data from {args.input}...")
    df = load_inflation_data(args.input)

    print(f"Data range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Total observations: {len(df)}")

    all_results = []

    # 1. Chow Test at candidate dates
    print("\n[1/3] Running Chow test at candidate dates...")
    chow_results = chow_test_breakpoints(df, args.candidate_dates)
    all_results.extend(chow_results)
    print(f"  Found {len(chow_results)} candidate breakpoints")

    # 2. Bai-Perron Test (unknown breakpoints)
    print("\n[2/3] Running Bai-Perron test (unknown breakpoints)...")
    bp_results = bai_perron_analysis(df, max_breaks=args.max_breaks)
    all_results.extend(bp_results)
    print(f"  Detected {len(bp_results)} structural breaks")

    # 3. Variance Breaks Detection
    print("\n[3/3] Detecting variance breaks...")
    var_results = detect_variance_breaks(df)
    all_results.extend(var_results)
    print(f"  Found {len(var_results)} variance breaks")

    # Save results
    output_df = pd.DataFrame(all_results)
    output_df = output_df.sort_values("break_date").reset_index(drop=True)

    print(f"\nSaving results to {args.output}...")
    output_df.to_csv(args.output, index=False, sep=",")
    print(f"  Total breaks detected: {len(output_df)}")

    # Summary
    print("\n=== SUMMARY ===")
    if "significant" in output_df.columns:
        significant_breaks = output_df[output_df["significant"] == True]
        if len(significant_breaks) > 0:
            print(f"Significant breaks (p < 0.05): {len(significant_breaks)}")
            print("\nBreak dates:")
            for _, row in significant_breaks.iterrows():
                print(f"  - {row['break_date']}: {row.get('description', 'N/A')}")
        else:
            print("No statistically significant breaks detected at 5% level.")
    else:
        print("No significance tests performed (variance breaks only).")

    print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
