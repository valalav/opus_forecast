#!/usr/bin/env python3
"""
Weekly Price Correlation Analysis
================================

Analyze cross-correlations between weekly price movements of different products
to identify correlated groups and potential minimal basket.

Methodology:
1. Load weekly price data for all products
2. Calculate pairwise Pearson correlation matrices
3. Apply hierarchical clustering to group correlated products
4. Identify high-correlation clusters for minimal basket selection

Output:
- data/product_correlation_matrix.csv: Full correlation matrix
- data/product_clusters.csv: Clustered products by correlation groups
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import argparse

warnings.filterwarnings("ignore")

# Add parent directory to path for imports
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from sirena.data.weekly_loader import load_weekly_prices, HIGH_QUALITY_PRODUCTS

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform


def pivot_weekly_to_products(
    weekly_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pivot weekly data from long to wide format (products as columns).

    Args:
        weekly_df: DataFrame with columns date, product_code, wow_growth

    Returns:
        DataFrame with dates as index, product_codes as columns
    """
    # Use WoW growth for correlation (stationary)
    pivoted = weekly_df.pivot(index="date", columns="product_code", values="wow_growth")

    # Filter to products with sufficient data
    min_obs_ratio = 0.8  # Require 80% non-null
    product_counts = pivoted.notna().mean()
    valid_products = product_counts[product_counts >= min_obs_ratio].index
    pivoted = pivoted[valid_products]

    return pivoted


def calculate_correlation_matrix(
    product_pivot: pd.DataFrame,
    min_periods: int = 52,
) -> pd.DataFrame:
    """
    Calculate pairwise Pearson correlation between products.

    Args:
        product_pivot: Pivoted DataFrame with products as columns
        min_periods: Minimum observations required for correlation

    Returns:
        Correlation matrix (N x N)
    """
    corr_matrix = product_pivot.corr(min_periods=min_periods)

    # Remove NaN rows/cols (products with insufficient data)
    corr_matrix = corr_matrix.dropna(how="all").dropna(axis=1, how="all")

    return corr_matrix


def cluster_products(
    corr_matrix: pd.DataFrame,
    n_clusters: int = 10,
    distance_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Perform hierarchical clustering based on correlation distance.

    Distance = 1 - correlation (high correlation = low distance)

    Args:
        corr_matrix: Correlation matrix
        n_clusters: Target number of clusters
        distance_threshold: Max distance to merge clusters

    Returns:
        DataFrame with product_code, cluster_id, and cluster_size
    """
    # Convert correlation to distance
    distance_matrix = 1 - corr_matrix.fillna(0)

    # Handle NaN values in distance matrix
    distance_matrix = distance_matrix.fillna(1.0)  # Max distance for missing

    # Ensure diagonal is zero (self-correlation = 1)
    np.fill_diagonal(distance_matrix.values, 0.0)

    # Condense to pairwise distance vector
    condensed_dist = squareform(distance_matrix.values)

    # Perform hierarchical clustering
    Z = linkage(condensed_dist, method="average", metric="euclidean")

    # Cut dendrogram to get clusters
    cluster_labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Create cluster DataFrame
    clusters = pd.DataFrame(
        {
            "product_code": corr_matrix.index,
            "cluster_id": cluster_labels,
        }
    )

    # Add product names
    product_names = {}
    for code, info in HIGH_QUALITY_PRODUCTS.items():
        product_names[code] = info["name"]

    clusters["product_name"] = clusters["product_code"].map(
        lambda x: product_names.get(x, f"Product_{x}")
    )

    # Calculate cluster statistics
    cluster_stats = clusters.groupby("cluster_id").size().rename("cluster_size")
    clusters = clusters.merge(cluster_stats, on="cluster_id")

    # Sort by cluster_id and cluster_size
    clusters = clusters.sort_values(
        ["cluster_id", "cluster_size"], ascending=[True, False]
    )

    return clusters


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze correlations between weekly product prices"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2016-01-01",
        help="Start date for analysis (default: 2016-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for analysis (default: None = all available)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=10,
        help="Number of clusters for hierarchical clustering (default: 10)",
    )
    parser.add_argument(
        "--min-periods",
        type=int,
        default=52,
        help="Minimum observations required for correlation (default: 52)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory (default: data)",
    )

    args = parser.parse_args()

    print(f"Loading weekly price data from {args.start_date}...")
    weekly_df = load_weekly_prices(start_date=args.start_date, end_date=args.end_date)

    print(f"Pivoting to product matrix...")
    product_pivot = pivot_weekly_to_products(weekly_df)

    print(f"Calculating correlation matrix ({len(product_pivot.columns)} products)...")
    corr_matrix = calculate_correlation_matrix(
        product_pivot, min_periods=args.min_periods
    )

    print(f"Clustering products into {args.n_clusters} groups...")
    clusters = cluster_products(corr_matrix, n_clusters=args.n_clusters)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Save correlation matrix
    corr_path = output_dir / "product_correlation_matrix.csv"
    corr_matrix.to_csv(corr_path)
    print(f"Saved correlation matrix to {corr_path}")

    # Save clusters
    clusters_path = output_dir / "product_clusters.csv"
    clusters.to_csv(clusters_path, index=False)
    print(f"Saved clusters to {clusters_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"Products analyzed: {len(product_pivot.columns)}")
    print(f"Valid correlations: {corr_matrix.notna().sum().sum()}")
    print(f"Clusters created: {clusters['cluster_id'].nunique()}")
    print("\nTop 5 products by cluster size:")
    print(clusters.drop_duplicates("cluster_id").nlargest(5, "cluster_size"))


if __name__ == "__main__":
    main()
