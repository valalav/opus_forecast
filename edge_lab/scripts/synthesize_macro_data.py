#!/usr/bin/env python3
"""
Task 121: Synthesis - The Ultimate Macro Dataset
Combines ALL extracted data (Tasks 114, 115, 118, 119, 120) into a single master dataset.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings("ignore")

print("=" * 60)
print("Task 121: The Ultimate Macro Dataset Synthesis")
print("=" * 60)

# Step 1: Load all datasets
print("\n[1/7] Loading datasets...")

# Task 114: Macro Monolith
df_macro = pd.read_csv("data/kbr_macro_monolith.csv")
print(f"  - kbr_macro_monolith.csv: {df_macro.shape}")

# Task 115: Sectoral Details
df_sectoral = pd.read_csv("data/kbr_sectoral_details.csv")
print(f"  - kbr_sectoral_details.csv: {df_sectoral.shape}")

# Task 118: Labor Market
df_labor = pd.read_csv("data/kbr_labor_market.csv")
print(f"  - kbr_labor_market.csv: {df_labor.shape}")

# Task 119: PPI
df_ppi = pd.read_csv("data/kbr_ppi_detailed.csv")
df_ppi.rename(columns={"date": "Date", "value": "Value"}, inplace=True)
print(f"  - kbr_ppi_detailed.csv: {df_ppi.shape}")

# Task 120: GRP Forecast
df_grp = pd.read_csv("data/kbr_grp_forecast.csv")
df_grp.rename(columns={"date": "Date"}, inplace=True)
print(f"  - kbr_grp_forecast.csv: {df_grp.shape}")

# Inflation target
df_inflation = pd.read_csv("data/enhanced_inflation_data.csv")
print(f"  - enhanced_inflation_data.csv (target): {df_inflation.shape}")

# Step 2: Pivot long-format datasets to wide format
print("\n[2/7] Pivoting long-format datasets...")

# Pivot macro monolith - create unique column names (handle duplicates)
df_macro["feature_name"] = df_macro["Indicator"] + "_" + df_macro["Metric_Type"]
df_macro_wide = (
    df_macro.groupby(["Date", "feature_name"])["Value"].mean().unstack().reset_index()
)
print(f"  - Macro monolith pivoted: {df_macro_wide.shape[1] - 1} features")

# Pivot sectoral details (handle duplicates)
df_sectoral["feature_name"] = (
    "S_"
    + df_sectoral["Indicator"]
    + "_"
    + df_sectoral["Metric_Type"]
    + "_S"
    + df_sectoral["Sheet"].astype(str)
)
df_sectoral_wide = (
    df_sectoral.groupby(["Date", "feature_name"])["Value"]
    .mean()
    .unstack()
    .reset_index()
)
print(f"  - Sectoral pivoted: {df_sectoral_wide.shape[1] - 1} features")

# Pivot labor market (handle duplicates)
df_labor["feature_name"] = (
    "L_" + df_labor["Series_Name"] + "_" + df_labor["Indicator_Type"]
)
df_labor_wide = (
    df_labor.groupby(["Date", "feature_name"])["Value"].mean().unstack().reset_index()
)
print(f"  - Labor market pivoted: {df_labor_wide.shape[1] - 1} features")

# Pivot PPI (handle duplicates)
df_ppi["feature_name"] = "PPI_" + df_ppi["indicator"] + "_" + df_ppi["metric_type"]
df_ppi_wide = (
    df_ppi.groupby(["Date", "feature_name"])["Value"].mean().unstack().reset_index()
)
print(f"  - PPI pivoted: {df_ppi_wide.shape[1] - 1} features")

# Step 3: Start with inflation as base and merge
print("\n[3/7] Merging datasets on Date...")

# Start with inflation data
master_df = df_inflation[["Date", "mom"]].copy()
master_df.rename(columns={"mom": "CPI_mom"}, inplace=True)

# Merge each dataset
for df, name in [
    (df_macro_wide, "macro"),
    (df_sectoral_wide, "sectoral"),
    (df_labor_wide, "labor"),
    (df_ppi_wide, "ppi"),
    (df_grp, "grp"),
]:
    master_df = pd.merge(master_df, df, on="Date", how="left")
    print(
        f"  - After merging {name}: {master_df.shape[1] - 1} features, {master_df.shape[0]} rows"
    )

print(f"\n  - Master dataset shape: {master_df.shape}")
print(f"  - Total features (excluding Date and CPI): {master_df.shape[1] - 2}")

# Step 4: Handle missing values
print("\n[4/7] Handling missing values...")

# Calculate missing percentage before
missing_before = (
    master_df.isnull().sum().sum() / (master_df.shape[0] * master_df.shape[1]) * 100
)
print(f"  - Missing data before: {missing_before:.2f}%")

# Remove columns with >50% missing data
non_date_cols = [c for c in master_df.columns if c != "Date" and c != "CPI_mom"]
missing_per_col = master_df[non_date_cols].isnull().sum() / len(master_df) * 100
cols_to_keep = ["Date", "CPI_mom"] + [
    c for c in non_date_cols if missing_per_col[c] <= 50
]
master_df = master_df[cols_to_keep]
print(
    f"  - Removed {len(non_date_cols) - len(cols_to_keep) + 2} columns with >50% missing"
)

# Fill remaining missing values with interpolation
numeric_cols = [c for c in master_df.columns if c != "Date"]
for col in numeric_cols:
    if master_df[col].isnull().any():
        master_df[col] = master_df[col].interpolate(
            method="linear", limit_direction="both"
        )
        # Fill any remaining NaN with forward fill
        master_df[col] = master_df[col].fillna(method="ffill").fillna(method="bfill")

# Calculate missing percentage after
missing_after = (
    master_df.isnull().sum().sum() / (master_df.shape[0] * master_df.shape[1]) * 100
)
print(f"  - Missing data after: {missing_after:.2f}%")
print(f"  - Final dataset shape: {master_df.shape}")

# Step 5: Calculate correlation with target
print("\n[5/7] Calculating correlation with CPI...")

# Prepare feature columns
feature_cols = [c for c in master_df.columns if c not in ["Date", "CPI_mom"]]
print(f"  - Analyzing {len(feature_cols)} features")

# Calculate correlations
correlations = []
for col in feature_cols:
    try:
        corr = master_df[[col, "CPI_mom"]].corr().iloc[0, 1]
        if not np.isnan(corr):
            correlations.append((col, abs(corr), corr))
    except:
        pass

# Sort by absolute correlation
correlations.sort(key=lambda x: x[1], reverse=True)
print(f"\n  Top 20 features by correlation:")
for i, (col, abs_corr, corr) in enumerate(correlations[:20]):
    print(f"    {i + 1:2d}. {col[:50]:50s} | Corr: {corr:7.4f}")

# Step 6: Lasso feature selection
print("\n[6/7] Running Lasso feature selection...")

# Prepare data for Lasso - exclude non-numeric columns
numeric_feature_cols = [c for c in feature_cols if master_df[c].dtype != "object"]
print(f"  - Using {len(numeric_feature_cols)} numeric features for Lasso")

X = master_df[numeric_feature_cols].values
y = master_df["CPI_mom"].values

# Impute any remaining NaN
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Remove NaN from y
mask = ~np.isnan(y)
X_scaled = X_scaled[mask]
y = y[mask]

# Run LassoCV
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_scaled, y)

# Get feature importance
feature_importance = []
for i, col in enumerate(numeric_feature_cols):
    coef = lasso.coef_[i] if i < len(lasso.coef_) else 0
    if coef != 0:
        feature_importance.append((col, abs(coef), coef))

feature_importance.sort(key=lambda x: x[1], reverse=True)
print(f"  - Lasso selected {len(feature_importance)} non-zero features")
print(f"  - Alpha: {lasso.alpha_:.6f}")

# Step 7: Select Top-10 features using combined ranking
print("\n[7/7] Selecting Top-10 features...")

# Combine rankings (weight correlation 0.6, lasso 0.4)
correlation_rank = {col: i for i, (col, _, _) in enumerate(correlations)}
lasso_rank = {col: i for i, (col, _, _) in enumerate(feature_importance)}

combined_scores = []
all_features = set([c for c, _, _ in correlations]) | set(
    [c for c, _, _ in feature_importance]
)

for col in all_features:
    corr_score = len(correlations) - correlation_rank.get(col, len(correlations))
    lasso_score = len(feature_importance) - lasso_rank.get(col, len(feature_importance))
    combined = 0.6 * corr_score + 0.4 * lasso_score
    combined_scores.append((col, combined, corr_score, lasso_score))

combined_scores.sort(key=lambda x: x[1], reverse=True)

top_10 = combined_scores[:10]
print(f"\n  Top-10 Features (Combined Ranking):")
print("-" * 80)
for i, (col, combined, corr_score, lasso_score) in enumerate(top_10):
    print(f"  {i + 1:2d}. {col[:55]:55s}")
    print(
        f"      Corr Rank: {len(correlations) - corr_score:3d} | Lasso Rank: {len(feature_importance) - lasso_score:3d} | Score: {combined:6.2f}"
    )

# Save full master dataset with all features
master_df.to_csv("data/master_macro_dataset.csv", index=False)
print(f"\n  - Master dataset saved: data/master_macro_dataset.csv")
print(f"  - Shape: {master_df.shape}")

# Create a separate dataset with top-10 features for easy model training
final_cols = ["Date", "CPI_mom"] + [col for col, _, _, _ in top_10]
final_df = master_df[final_cols].copy()
final_df.to_csv("data/master_macro_dataset_top10.csv", index=False)
print(f"  - Top-10 subset saved: data/master_macro_dataset_top10.csv")
print(f"  - Shape: {final_df.shape}")

# Generate rationale document
rationale = f"""# Top-10 Feature Selection Rationale

## Methodology

We combined two complementary feature selection approaches:

1. **Correlation Analysis (60% weight)**: Pearson correlation with CPI inflation target
2. **Lasso Regression (40% weight)**: L1 regularization to identify predictive features

### Datasets Integrated

| Source | Task | Features Extracted |
|--------|------|-------------------|
| Macro Monolith | 114 | {df_macro_wide.shape[1] - 1} macro indicators |
| Sectoral Details | 115 | {df_sectoral_wide.shape[1] - 1} sectoral breakdowns |
| Labor Market | 118 | {df_labor_wide.shape[1] - 1} employment/wage metrics |
| PPI | 119 | {df_ppi_wide.shape[1] - 1} producer price indices |
| GRP Forecast | 120 | GRP index and forecasts |

### Data Processing

- **Missing Data**: Columns with >50% missing values removed ({len(non_date_cols) - len(cols_to_keep) + 2} columns)
- **Imputation**: Linear interpolation for remaining gaps
- **Final Features**: {master_df.shape[1] - 2} columns before selection

### Top-10 Selected Features

"""

for i, (col, combined, corr_score, lasso_score) in enumerate(top_10):
    corr_val = next((c2 for c1, c2, _ in correlations if c1 == col), None)
    lasso_val = next((c2 for c1, c2, _ in feature_importance if c1 == col), 0)
    rationale += f"{i + 1}. **{col}**\n"
    rationale += f"   - Correlation with CPI: {corr_val:+.4f}\n"
    rationale += f"   - Lasso coefficient: {lasso_val:+.6f}\n"
    rationale += f"   - Combined score: {combined:.2f}\n\n"

rationale += f"""
## Summary

- **Total Features Available**: {master_df.shape[1] - 2}
- **Features Selected**: 10
- **Missing Data Handled**: {missing_before:.2f}% → {missing_after:.2f}%
- **Model-Ready**: Yes - all features are numeric and imputed

## Ready for Model Training

The `data/master_macro_dataset.csv` file contains:
- Date column for temporal ordering
- `CPI_mom` as the target variable
- Top-10 selected features
- No missing values
- Ready for time series cross-validation
"""

with open("data/top10_feature_rationale.md", "w") as f:
    f.write(rationale)

print(f"\n  - Rationale documented: data/top10_feature_rationale.md")

print("\n" + "=" * 60)
print("Task 121 Complete: Master macro dataset synthesized!")
print("=" * 60)
print(
    f"\n  - Master dataset (full): data/master_macro_dataset.csv ({master_df.shape[1] - 1} features)"
)
print(
    f"  - Master dataset (top-10): data/master_macro_dataset_top10.csv ({final_df.shape[1] - 1} features)"
)
print(f"  - Rationale: data/top10_feature_rationale.md")
