#!/usr/bin/env python3
"""
Task 119: Deep Dive - Producer Prices (PPI)
Extract manufacturing and agricultural PPI for KBR (using SKFO proxy)
and analyze lag correlation with CPI.
"""

import openpyxl
import pandas as pd
import numpy as np
from pathlib import Path

MONTH_NAMES = [
    "янв",
    "фев",
    "мар",
    "апр",
    "май",
    "июн",
    "июл",
    "авг",
    "сен",
    "окт",
    "ноя",
    "дек",
]


def process_sheet_ppi(ws, target_codes=None, region_target=None):
    """Process a PPI sheet and extract relevant data."""
    # Read all data at once for efficiency
    all_data = list(ws.iter_rows(values_only=True))

    # Find header positions
    row_labels_idx = None
    year_row_idx = None
    month_row_idx = None

    for i, row in enumerate(all_data):
        if i < 10 and row[0] == "Row Labels":
            row_labels_idx = i
            year_row_idx = i - 2
            month_row_idx = i - 1
            break

    if year_row_idx is None or month_row_idx is None:
        return []

    # Build date mapping
    date_map = {}
    current_year = None
    year_row = all_data[year_row_idx]
    month_row = all_data[month_row_idx]

    for col_idx in range(6, len(year_row)):
        year_val = year_row[col_idx]
        month_val = month_row[col_idx]

        if year_val and isinstance(year_val, (int, str)) and str(year_val).isdigit():
            current_year = int(str(year_val))

        if month_val and current_year:
            month_str = str(month_val).lower()
            if month_str in MONTH_NAMES:
                month_num = MONTH_NAMES.index(month_str) + 1
                date_map[col_idx] = f"{current_year}-{month_num:02d}-01"

    if not date_map:
        return []

    results = []
    target_set = {c for c, _ in target_codes} if target_codes else None

    # Process data rows
    for row_idx in range(row_labels_idx + 1, len(all_data)):
        row = all_data[row_idx]

        if len(row) < 3:
            continue

        # Check if this row matches our target
        if target_codes:
            code = str(row[1]) if row[1] else ""
            if code not in target_set:
                continue
            name = next((n for c, n in target_codes if c == code), "")
        elif region_target:
            okato_code = str(row[1]) if row[1] else ""
            region_name = str(row[2]) if row[2] else ""
            if okato_code != "038" or "Северо - Кавказский" not in region_name:
                continue
            name = "SKFO (proxy for KBR)"
            code = "SKFO"
        else:
            continue

        # Extract monthly values
        for col_idx, date_str in date_map.items():
            if col_idx < len(row) and row[col_idx] is not None:
                val = row[col_idx]
                if val != "..." and str(val) != "" and str(val) != "nan":
                    try:
                        results.append(
                            {
                                "date": date_str,
                                "code": code,
                                "indicator": name,
                                "metric_type": ws.title,
                                "value": float(val),
                            }
                        )
                    except (ValueError, TypeError):
                        pass

    return results


def load_cpi_data():
    """Load CPI data."""
    cpi_file = Path(
        "/home/valalav/_projects/sirena-kbr/edge_lab/data/enhanced_inflation_data.csv"
    )
    if not cpi_file.exists():
        return None

    df = pd.read_csv(cpi_file)
    df["date"] = pd.to_datetime(df["Date"])
    df["cpi_growth"] = df["mom"] - 100
    return df[["date", "cpi_growth"]]


def calculate_lag_correlations(ppi_df, cpi_df, max_lag=6):
    """Calculate PPI vs CPI lag correlations."""
    correlations = []

    merged = pd.merge(ppi_df, cpi_df, on="date", how="inner")
    if merged.empty:
        return correlations

    for indicator in ppi_df["indicator"].unique():
        ind_data = merged[merged["indicator"] == indicator]
        if len(ind_data) < 12:
            continue

        ppi_vals = ind_data["value"].to_numpy()
        cpi_vals = ind_data["cpi_growth"].to_numpy()

        best_corr = 0
        best_lag = 0

        for lag in range(max_lag + 1):
            if len(ppi_vals) > lag:
                ppi_lagged = ppi_vals[lag:]
                cpi_aligned = cpi_vals[:-lag] if lag > 0 else cpi_vals

                if len(ppi_lagged) > 10:
                    corr = np.corrcoef(ppi_lagged, cpi_aligned)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(
                            {
                                "indicator": indicator,
                                "metric_type": ind_data["metric_type"].iloc[0],
                                "lag_months": lag,
                                "correlation": corr,
                            }
                        )
                        if abs(corr) > abs(best_corr):
                            best_corr = corr
                            best_lag = lag

        print(f"  {indicator[:50]}: best lag = {best_lag}mo, corr = {best_corr:.4f}")

    return correlations


def main():
    """Main execution."""
    print("=" * 60)
    print("Task 119: Deep Dive - Producer Prices (PPI)")
    print("=" * 60)

    ppi_file = Path(
        "/home/valalav/_projects/sirena-kbr/edge_lab/assets/charts/ОПР_статистика/Цены производителей/ОКВЭДОКПД2_цены_производителей_полный.xlsx"
    )

    if not ppi_file.exists():
        print(f"Error: File not found: {ppi_file}")
        return

    # Industry codes of interest
    target_codes = [
        ("В.ppi_prom.intermediate.manufacturing", "Manufacturing - Intermediate goods"),
        ("В.ppi_prom_less_FEC", "Industry excluding Fuel-Energy Complex"),
        ("В.10+11.07", "Food production (agricultural products)"),
        ("В.ppi_prom.consumer.nondurable", "Consumer goods non-durable"),
        ("В.05+06+09", "Fuel-Energy Complex"),
        ("В.ppi_prom.investment", "Investment goods"),
        ("В.ppi_prom.intermediate.mining", "Mining - Intermediate"),
    ]

    # Load workbook
    print("Loading PPI workbook (73MB)...")
    wb = openpyxl.load_workbook(str(ppi_file), read_only=True)

    all_data = []

    # Extract industry PPI
    print("\nExtracting industry PPI data...")
    for sheet_name in ["MoM_отр", "YoY_отр", "Рублей_отр"]:
        print(f"  Sheet: {sheet_name}...", end="", flush=True)
        data = process_sheet_ppi(wb[sheet_name], target_codes=target_codes)
        all_data.extend(data)
        print(f" {len(data)} records")

    # Extract regional PPI (SKFO as KBR proxy)
    print("\nExtracting regional PPI (SKFO as KBR proxy)...")
    for sheet_name in ["MoM_рег", "YoY_рег"]:
        print(f"  Sheet: {sheet_name}...", end="", flush=True)
        data = process_sheet_ppi(wb[sheet_name], region_target="SKFO")
        all_data.extend(data)
        print(f" {len(data)} records")

    wb.close()

    if not all_data:
        print("Error: No data extracted")
        return

    ppi_df = pd.DataFrame(all_data)
    ppi_df["date"] = pd.to_datetime(ppi_df["date"])

    print(f"\nTotal PPI records: {len(ppi_df)}")
    print(f"Unique indicators: {ppi_df['indicator'].nunique()}")
    print(f"Date range: {ppi_df['date'].min()} to {ppi_df['date'].max()}")

    # Load CPI and calculate correlations
    cpi_df = load_cpi_data()

    if cpi_df is not None:
        print(f"\nCPI data: {len(cpi_df)} records")
        print(f"Date range: {cpi_df['date'].min()} to {cpi_df['date'].max()}")

        print("\n=== Lag Correlation Analysis (PPI vs CPI) ===")
        correlations = calculate_lag_correlations(ppi_df, cpi_df, max_lag=6)

        if correlations:
            corr_df = pd.DataFrame(correlations)
            corr_file = Path(
                "/home/valalav/_projects/sirena-kbr/edge_lab/data/ppi_cpi_correlations.csv"
            )
            corr_df.to_csv(corr_file, index=False)
            print(f"\nCorrelations saved to: {corr_file}")

            # Show best correlations
            print("\nTop correlations by absolute value:")
            best_idx = corr_df.groupby("indicator")["correlation"].apply(
                lambda x: x.abs().idxmax()
            )
            best = corr_df.loc[best_idx]
            best_sorted = best.sort_values("correlation", ascending=False, key=abs)
            print(
                best_sorted[["indicator", "lag_months", "correlation"]].to_string(
                    index=False
                )
            )

    # Save outputs
    output_file = Path("/home/valalav/_projects/sirena-kbr/edge_lab/data/kbr_ppi_detailed.csv")
    ppi_df.to_csv(output_file, index=False)
    print(f"\nPPI data saved to: {output_file}")

    if cpi_df is not None:
        merged = ppi_df.merge(cpi_df, on="date", how="left")
        merged_file = Path(
            "/home/valalav/_projects/sirena-kbr/edge_lab/data/kbr_ppi_detailed_with_cpi.csv"
        )
        merged.to_csv(merged_file, index=False)
        print(f"PPI + CPI saved to: {merged_file}")

    print("\n" + "=" * 60)
    print("Task 119 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
