#!/usr/bin/env python3
"""
Minimal Basket Analysis (Simplified)
=====================================

Analyze correlation clusters to define a minimal product basket for weekly nowcasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def main():
    print("=" * 60)
    print("MINIMAL BASKET ANALYSIS")
    print("=" * 60)

    # Load cluster data and correlation matrix
    clusters = pd.read_csv("data/product_clusters.csv")
    corr_matrix = pd.read_csv("data/product_correlation_matrix.csv", index_col=0)

    # Filter to intersection
    corr_index_list = corr_matrix.index.tolist()
    valid_mask = clusters["product_code"].isin(corr_index_list)
    clusters = clusters[valid_mask].copy()

    print(f"✓ Loaded {len(clusters)} products from clusters")
    print(f"✓ Loaded {len(corr_matrix)} products from correlation matrix")

    # Analyze each cluster and select representative
    minimal_basket = []

    for cluster_id in sorted(clusters["cluster_id"].unique()):
        cluster_products = clusters[clusters["cluster_id"] == cluster_id][
            "product_code"
        ].tolist()
        cluster_products = [p for p in cluster_products if p in corr_index_list]
        cluster_size = len(cluster_products)

        if cluster_size == 0:
            continue

        if cluster_size == 1:
            # Single product in cluster - select it
            selected_product = int(cluster_products[0])
            avg_corr = np.nan
            reason = "Single product in cluster"
        else:
            # Multiple products - select one with highest avg intra-cluster correlation
            best_product = None
            best_avg_corr = -np.inf

            for p in cluster_products:
                other_products = [
                    x for x in cluster_products if x != p and x in corr_index_list
                ]

                if not other_products:
                    continue

                # Get correlations to other cluster members
                corrs = []
                for other in other_products:
                    if other in corr_index_list and p in corr_index_list:
                        corr_val = corr_matrix.loc[p, other]
                        if pd.notna(corr_val):
                            corrs.append(float(corr_val))

                if corrs:
                    avg_corr = float(np.mean(corrs))
                    if avg_corr > best_avg_corr:
                        best_avg_corr = avg_corr
                        best_product = int(p)

            selected_product = best_product
            avg_corr = best_avg_corr
            reason = f"Highest avg correlation to cluster members ({best_avg_corr:.3f})"

        # Get product name
        product_row = clusters[clusters["product_code"] == selected_product]
        if len(product_row) > 0:
            product_name = product_row.iloc[0]["product_name"]
        else:
            product_name = f"Code {selected_product}"

        minimal_basket.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": cluster_size,
                "product_code": selected_product,
                "product_name": product_name,
                "avg_intra_corr": avg_corr,
                "reason": reason,
            }
        )

        print(
            f"  Cluster {cluster_id}: {cluster_size} products → Selected {product_name} ({selected_product})"
        )

    # Create DataFrame and save
    basket_df = pd.DataFrame(minimal_basket)
    output_file = Path("data/minimal_basket.csv")
    basket_df.to_csv(output_file, index=False)
    print(f"\n✓ Minimal basket saved to: {output_file}")

    # Display summary table
    print("\n" + "=" * 60)
    print("MINIMAL BASKET (one product per cluster)")
    print("=" * 60)
    print(f"\n| Cluster | Product Code | Product Name | Reason |")
    print("-" * 80)
    for _, row in basket_df.iterrows():
        cluster_id = int(row["cluster_id"])
        product_code = int(row["product_code"])
        product_name = row["product_name"]
        reason = row["reason"]
        print(
            f"| {cluster_id:7} | {product_code:12} | {product_name:35} | {reason:20} |"
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total products analyzed: {len(clusters)}")
    print(f"Number of clusters: {len(basket_df)}")
    print(f"Minimal basket size: {len(basket_df)} products")
    print(
        f"Reduction ratio: {len(basket_df)}/{len(clusters)} = {len(basket_df) / len(clusters) * 100:.1f}%"
    )


if __name__ == "__main__":
    main()
