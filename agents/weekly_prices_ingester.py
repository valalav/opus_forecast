import pandas as pd
import numpy as np
import re
import os
import sys

def parse_date_range(date_str):
    """
    Parses a date range string like 'с 07.01.2008 по 14.01.2008'
    Returns the end date as a datetime object.
    """
    if not isinstance(date_str, str):
        return None
    
    # Try pattern "с start по end"
    match = re.search(r'по\s+(\d{2}\.\d{2}\.\d{4})', date_str)
    if match:
        return pd.to_datetime(match.group(1), dayfirst=True)
    
    # Fallback: try parsing just the date if it's not a range
    try:
        return pd.to_datetime(date_str, dayfirst=True)
    except:
        return None

def clean_product_code(code):
    """
    Cleans product code (e.g. 1001.0 -> 1001, '1001' -> 1001)
    """
    try:
        return int(float(code))
    except (ValueError, TypeError):
        return None

def ingest_weekly_prices(input_path, output_path, report_path):
    print(f"Loading {input_path}...")
    try:
        # Load 'Отчет' sheet, header is row 1 (0-indexed)
        df_raw = pd.read_excel(input_path, sheet_name='Отчет', header=1)
    except Exception as e:
        print(f"Error loading Excel: {e}")
        return

    print(f"Raw shape: {df_raw.shape}")
    
    # Analyze columns
    # Col 0: Name, Col 1: Code. Rest are dates.
    
    # Extract date columns
    date_cols = []
    metadata_cols = []
    
    date_mapping = {}
    
    for col in df_raw.columns:
        parsed_date = parse_date_range(col)
        if parsed_date is not None:
            date_cols.append(col)
            date_mapping[col] = parsed_date
        else:
            # Assume first two columns are metadata regardless of name
            pass

    # Assume first two valid columns are Product Name and code
    # The file structure description said:
    # Column 0: Product Name
    # Column 1: Product Code
    # Let's verify by position if names are ambiguous
    
    prod_name_col = df_raw.columns[0]
    prod_code_col = df_raw.columns[1]
    
    print(f"Product Name Column: {prod_name_col}")
    print(f"Product Code Column: {prod_code_col}")
    
    # Rename for clarity
    df_raw = df_raw.rename(columns={prod_name_col: 'product_name', prod_code_col: 'product_code_raw'})
    
    # Filter valid rows (must have product code)
    df_raw['product_code'] = df_raw['product_code_raw'].apply(clean_product_code)
    df_clean = df_raw.dropna(subset=['product_code']).copy()
    
    print(f"Rows with valid codes: {len(df_clean)}")
    
    # Melt to long format
    print("Reshaping to long format...")
    melted = df_clean.melt(
        id_vars=['product_code', 'product_name'], 
        value_vars=date_cols,
        var_name='date_raw',
        value_name='price'
    )
    
    # Map dates
    melted['date'] = melted['date_raw'].map(date_mapping)
    
    # Sort
    melted = melted.sort_values(['product_code', 'date'])
    
    # Interpolation and Growth calc
    print("Processing time series (interpolation and growth)...")
    
    final_dfs = []
    quality_issues = []
    
    for code, group in melted.groupby('product_code'):
        group = group.sort_values('date').set_index('date')
        
        # Resample to weekly to identify gaps properly, though raw data should be widely regular
        # But let's work with the raw series first to interpolation NaNs
        
        total_points = len(group)
        missing_initial = group['price'].isna().sum()
        
        # Interpolate limit=4 (4 weeks)
        group['price_interp'] = group['price'].interpolate(method='linear', limit=3)
        
        # Check remaining NaNs (large gaps)
        missing_final = group['price_interp'].isna().sum()
        
        if missing_final / total_points > 0.1:
            quality_issues.append({
                'product_code': code, 
                'product_name': group['product_name'].iloc[0],
                'missing_pct': (missing_final / total_points) * 100
            })
            
        group['price'] = group['price_interp'] # Use interpolated
        
        # Calc growth
        group['price_prev_week'] = group['price'].shift(1)
        group['wow_growth'] = ((group['price'] / group['price_prev_week']) - 1) * 100
        
        # Reset index
        group = group.reset_index()[['date', 'product_code', 'product_name', 'price', 'price_prev_week', 'wow_growth']]
        final_dfs.append(group)
        
    result_df = pd.concat(final_dfs, ignore_index=True)
    
    # Final checks
    print(f"Final shape: {result_df.shape}")
    print(f"Date range: {result_df['date'].min()} to {result_df['date'].max()}")
    
    # Validation: Outliers (> 20% jump)
    outliers = result_df[abs(result_df['wow_growth']) > 20]
    print(f"Potential outliers (>20% WoW change): {len(outliers)}")
    
    # Save CSV
    print(f"Saving to {output_path}...")
    result_df.to_csv(output_path, index=False)
    
    # Save Report
    print(f"Generating report {report_path}...")
    with open(report_path, 'w') as f:
        f.write("# Weekly Data Quality Report\n\n")
        f.write(f"- **Total Rows:** {len(result_df)}\n")
        f.write(f"- **Products:** {result_df['product_code'].nunique()}\n")
        f.write(f"- **Date Range:** {result_df['date'].min().date()} to {result_df['date'].max().date()}\n")
        f.write(f"- **Outliers Detected (>20% WoW):** {len(outliers)}\n\n")
        
        if quality_issues:
            f.write("## ⚠️ Products with >10% Missing Data\n")
            f.write("| Code | Product | Missing % |\n")
            f.write("|------|---------|-----------|\n")
            for issue in quality_issues:
                f.write(f"| {issue['product_code']} | {issue['product_name']} | {issue['missing_pct']:.1f}% |\n")
        else:
            f.write("## ✅ No high-missingess products found.\n")
            
    print("Done.")

if __name__ == "__main__":
    INPUT_FILE = "data/raw/11511100200030200004.xlsx"
    OUTPUT_FILE = "data/kbr_weekly_prices_2008_2026.csv"
    REPORT_FILE = "weekly_data_quality.md"
    
    ingest_weekly_prices(INPUT_FILE, OUTPUT_FILE, REPORT_FILE)
