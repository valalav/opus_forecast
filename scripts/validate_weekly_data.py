#!/usr/bin/env python3
"""
Weekly Data Quality Validator
============================

Scans weekly price data for gaps and outliers.
Generates quality report in markdown format.

Usage:
    python3 scripts/validate_weekly_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime


def load_data(data_path: str = "data/kbr_weekly_prices_2008_2026.csv") -> pd.DataFrame:
    """Load weekly prices data."""
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def check_missing_weeks(df: pd.DataFrame, max_gap_weeks: int = 4) -> Dict:
    """
    Check for missing weeks in time series.

    Args:
        df: Weekly price data
        max_gap_weeks: Maximum allowed gap (default: 4 weeks = 1 month)

    Returns:
        Dictionary with gap statistics
    """
    results = {
        "total_products": df["product_code"].nunique(),
        "products_with_gaps": 0,
        "max_gap_weeks": max_gap_weeks,
        "gaps_by_product": [],
        "critical_gaps": [],  # Gaps > max_gap_weeks
    }

    all_dates = df["date"].unique()

    for code, group in df.groupby("product_code"):
        product_dates = sorted(group["date"].unique())
        gaps = []

        for i in range(len(product_dates) - 1):
            gap_weeks = (product_dates[i + 1] - product_dates[i]).days // 7
            if gap_weeks > 1:
                gaps.append(
                    {
                        "start_date": product_dates[i],
                        "end_date": product_dates[i + 1],
                        "gap_weeks": gap_weeks,
                    }
                )

        if gaps:
            results["products_with_gaps"] += 1
            product_name = group["product_name"].iloc[0]

            # Find maximum gap
            max_gap = max(gaps, key=lambda x: x["gap_weeks"])

            results["gaps_by_product"].append(
                {
                    "product_code": code,
                    "product_name": product_name,
                    "total_gaps": len(gaps),
                    "max_gap_weeks": max_gap["gap_weeks"],
                    "max_gap_period": f"{max_gap['start_date'].date()} to {max_gap['end_date'].date()}",
                }
            )

            # Track critical gaps
            if max_gap["gap_weeks"] > max_gap_weeks:
                results["critical_gaps"].append(
                    {
                        "product_code": code,
                        "product_name": product_name,
                        "gap_weeks": max_gap["gap_weeks"],
                        "period": f"{max_gap['start_date'].date()} to {max_gap['end_date'].date()}",
                    }
                )

    return results


def check_outliers(df: pd.DataFrame, z_threshold: float = 5.0) -> Dict:
    """
    Check for outliers using z-score method.

    Args:
        df: Weekly price data
        z_threshold: Z-score threshold (default: 5.0)

    Returns:
        Dictionary with outlier statistics
    """
    # Filter to non-null wow_growth values
    growth_df = df[df["wow_growth"].notna()].copy()

    results = {
        "z_threshold": z_threshold,
        "total_records": len(growth_df),
        "outliers_found": 0,
        "outliers_by_product": [],
        "extreme_outliers": [],  # z-score > 10
    }

    for code, group in growth_df.groupby("product_code"):
        # Calculate z-scores for this product
        mean = group["wow_growth"].mean()
        std = group["wow_growth"].std()

        if std == 0 or np.isnan(std):
            continue

        z_scores = np.abs((group["wow_growth"] - mean) / std)

        # Find outliers
        outlier_mask = z_scores > z_threshold
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            results["outliers_found"] += outlier_count

            outliers = group[outlier_mask].copy()
            outliers["z_score"] = z_scores[outlier_mask]

            # Find maximum outlier
            max_outlier = outliers.loc[outliers["z_score"].idxmax()]

            results["outliers_by_product"].append(
                {
                    "product_code": code,
                    "product_name": group["product_name"].iloc[0],
                    "outlier_count": outlier_count,
                    "max_z_score": max_outlier["z_score"],
                    "max_wow_growth": max_outlier["wow_growth"],
                    "max_outlier_date": max_outlier["date"].date(),
                }
            )

            # Track extreme outliers
            extreme_mask = z_scores > (z_threshold * 2)
            if extreme_mask.any():
                extreme_outliers = group[extreme_mask].copy()
                extreme_outliers["z_score"] = z_scores[extreme_mask]
                results["extreme_outliers"].extend(
                    [
                        {
                            "product_code": code,
                            "product_name": group["product_name"].iloc[0],
                            "date": row["date"].date(),
                            "wow_growth": row["wow_growth"],
                            "z_score": row["z_score"],
                        }
                        for _, row in extreme_outliers.iterrows()
                    ]
                )

    return results


def check_null_coverage(df: pd.DataFrame) -> Dict:
    """Check null value coverage by product."""
    results = {
        "total_products": df["product_code"].nunique(),
        "products_with_nulls": 0,
        "null_stats_by_product": [],
    }

    for code, group in df.groupby("product_code"):
        total_records = len(group)
        null_prices = group["price"].isna().sum()
        null_growth = group["wow_growth"].isna().sum()

        if null_prices > 0:
            results["products_with_nulls"] += 1
            results["null_stats_by_product"].append(
                {
                    "product_code": code,
                    "product_name": group["product_name"].iloc[0],
                    "total_records": total_records,
                    "null_prices": null_prices,
                    "null_growth": null_growth,
                    "null_pct": (null_prices / total_records) * 100,
                }
            )

    # Sort by null percentage
    results["null_stats_by_product"].sort(key=lambda x: x["null_pct"], reverse=True)

    return results


def generate_markdown_report(
    gaps: Dict,
    outliers: Dict,
    nulls: Dict,
    output_path: str = "data/weekly_quality_report.md",
) -> None:
    """Generate markdown quality report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Weekly Data Quality Report",
        f"",
        f"**Generated:** {timestamp}",
        f"",
        f"---",
        f"",
        f"## Executive Summary",
        f"",
        f"- **Total Products:** {gaps['total_products']}",
        f"- **Products with Gaps:** {gaps['products_with_gaps']}",
        f"- **Critical Gaps (>4 weeks):** {len(gaps['critical_gaps'])}",
        f"- **Products with Nulls:** {nulls['products_with_nulls']}",
        f"- **Total Outliers (|z| > 5):** {outliers['outliers_found']}",
        f"- **Extreme Outliers (|z| > 10):** {len(outliers['extreme_outliers'])}",
        f"",
    ]

    # Missing Weeks Section
    lines.extend(
        [
            f"## Missing Weeks Analysis",
            f"",
            f"**Maximum Allowed Gap:** {gaps['max_gap_weeks']} weeks (1 month)",
            f"",
            f"### Products with Time Gaps ({len(gaps['gaps_by_product'])})",
            f"",
            f"| Product Code | Product Name | Total Gaps | Max Gap (weeks) | Period |",
            f"|--------------|--------------|-------------|-----------------|--------|",
        ]
    )

    for gap in gaps["gaps_by_product"][:20]:  # Top 20
        lines.append(
            f"| {gap['product_code']} | {gap['product_name']} | "
            f"{gap['total_gaps']} | {gap['max_gap_weeks']} | {gap['max_gap_period']} |"
        )

    if len(gaps["gaps_by_product"]) > 20:
        lines.append(f"| ... | ... | ... | ... | ... |")
        lines.append(f"| | **{len(gaps['gaps_by_product'])} total** | | | |")

    # Critical Gaps Section
    if gaps["critical_gaps"]:
        lines.extend(
            [
                f"",
                f"### Critical Gaps (> {gaps['max_gap_weeks']} weeks) - {len(gaps['critical_gaps'])}",
                f"",
                f"**WARNING:** The following products have gaps exceeding {gaps['max_gap_weeks']} weeks:",
                f"",
                f"| Product Code | Product Name | Gap (weeks) | Period |",
                f"|--------------|--------------|--------------|--------|",
            ]
        )

        for gap in gaps["critical_gaps"]:
            lines.append(
                f"| {gap['product_code']} | {gap['product_name']} | "
                f"{gap['gap_weeks']} | {gap['period']} |"
            )
    else:
        lines.extend(
            [
                f"",
                f"### Critical Gaps",
                f"",
                f"✅ **No critical gaps found.** All gaps are within {gaps['max_gap_weeks']} weeks.",
                f"",
            ]
        )

    # Outliers Section
    lines.extend(
        [
            f"---",
            f"",
            f"## Outlier Analysis",
            f"",
            f"**Z-Score Threshold:** {outliers['z_threshold']}",
            f"**Total Outliers Detected:** {outliers['outliers_found']}",
            f"",
            f"### Products with Outliers ({len(outliers['outliers_by_product'])})",
            f"",
            f"| Product Code | Product Name | Outlier Count | Max |z|-score | Max WoW Growth | Date |",
            f"|--------------|--------------|---------------|--------------|----------------|------|",
        ]
    )

    for outlier in outliers["outliers_by_product"][:20]:  # Top 20
        lines.append(
            f"| {outlier['product_code']} | {outlier['product_name']} | "
            f"{outlier['outlier_count']} | {outlier['max_z_score']:.2f} | "
            f"{outlier['max_wow_growth']:.2f}% | {outlier['max_outlier_date']} |"
        )

    if len(outliers["outliers_by_product"]) > 20:
        lines.append(f"| ... | ... | ... | ... | ... | ... |")
        lines.append(f"| | **{len(outliers['outliers_by_product'])} total** | | | |")

    # Extreme Outliers Section
    if outliers["extreme_outliers"]:
        lines.extend(
            [
                f"",
                f"### Extreme Outliers (|z| > {outliers['z_threshold'] * 2}) - {len(outliers['extreme_outliers'])}",
                f"",
                f"**CRITICAL:** Extreme outliers detected:",
                f"",
                f"| Product Code | Product Name | Date | WoW Growth | Z-Score |",
                f"|--------------|--------------|------|------------|---------|",
            ]
        )

        for ext in outliers["extreme_outliers"][:50]:  # Top 50
            lines.append(
                f"| {ext['product_code']} | {ext['product_name']} | "
                f"{ext['date']} | {ext['wow_growth']:.2f}% | {ext['z_score']:.2f} |"
            )

        if len(outliers["extreme_outliers"]) > 50:
            lines.append(f"| ... | ... | ... | ... | ... |")
            lines.append(f"| | **{len(outliers['extreme_outliers'])} total** | | | |")
    else:
        lines.extend(
            [
                f"",
                f"### Extreme Outliers",
                f"",
                f"✅ **No extreme outliers found.** All outliers have |z| < {outliers['z_threshold'] * 2}.",
                f"",
            ]
        )

    # Null Values Section
    lines.extend(
        [
            f"---",
            f"",
            f"## Null Values Coverage",
            f"",
            f"**Products with Null Prices:** {nulls['products_with_nulls']} / {nulls['total_products']}",
            f"",
            f"### Top 20 Products by Null Percentage",
            f"",
            f"| Product Code | Product Name | Null Prices | Null Growth | Null % |",
            f"|--------------|--------------|-------------|-------------|---------|",
        ]
    )

    for null in nulls["null_stats_by_product"][:20]:
        lines.append(
            f"| {null['product_code']} | {null['product_name']} | "
            f"{null['null_prices']} | {null['null_growth']} | {null['null_pct']:.1f}% |"
        )

    if len(nulls["null_stats_by_product"]) > 20:
        lines.append(f"| ... | ... | ... | ... | ... |")
        lines.append(f"| | **{nulls['products_with_nulls']} total** | | | |")

    # Recommendations
    lines.extend(
        [
            f"---",
            f"",
            f"## Recommendations",
            f"",
        ]
    )

    recommendations = []

    if gaps["critical_gaps"]:
        recommendations.append(
            f"⚠️ **Critical Data Gaps:** {len(gaps['critical_gaps'])} products have gaps > {gaps['max_gap_weeks']} weeks. "
            f"Consider data imputation or exclusion of affected periods."
        )

    if nulls["products_with_nulls"] > nulls["total_products"] * 0.5:
        recommendations.append(
            f"⚠️ **High Null Rate:** {nulls['products_with_nulls']}/{nulls['total_products']} products have null prices. "
            f"Review data collection process."
        )

    if outliers["outliers_found"] > 100:
        recommendations.append(
            f"⚠️ **High Outlier Count:** {outliers['outliers_found']} outliers detected. "
            f"Review extreme values and consider robust outlier handling (e.g., winsorization)."
        )

    if outliers["extreme_outliers"]:
        recommendations.append(
            f"🚨 **Extreme Outliers:** {len(outliers['extreme_outliers'])} extreme outliers found. "
            f"Manual review recommended for these data points."
        )

    if not recommendations:
        recommendations.append(
            f"✅ **Good Data Quality:** No critical issues detected. "
            f"Data appears suitable for analysis."
        )

    lines.extend(recommendations)
    lines.extend([f"", f"---", f"", f"*End of Report*"])

    # Write to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report saved to: {output_path}")


def main():
    """Main validation pipeline."""
    print("Loading weekly data...")
    df = load_data()

    print(f"Total records: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Products: {df['product_code'].nunique()}")

    print("\nChecking for missing weeks...")
    gaps = check_missing_weeks(df, max_gap_weeks=4)
    print(f"  Products with gaps: {gaps['products_with_gaps']}")
    print(f"  Critical gaps (>4 weeks): {len(gaps['critical_gaps'])}")

    print("\nChecking for outliers...")
    outliers = check_outliers(df, z_threshold=5.0)
    print(f"  Total outliers (|z| > 5): {outliers['outliers_found']}")
    print(f"  Extreme outliers (|z| > 10): {len(outliers['extreme_outliers'])}")

    print("\nChecking null coverage...")
    nulls = check_null_coverage(df)
    print(f"  Products with nulls: {nulls['products_with_nulls']}")

    print("\nGenerating report...")
    generate_markdown_report(gaps, outliers, nulls)

    print("\nValidation complete!")


if __name__ == "__main__":
    main()
