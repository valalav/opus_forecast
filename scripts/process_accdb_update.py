"""
Process updated db_cpi_store.accdb — extract all critical data.

Steps:
1. Export small tables (regions_names, items_names, items_structure, region_structure, z_max_day)
2. Export weights table  
3. Stream-extract aggregate indices (Items 1,2,3,4,33) for ALL regions
4. Stream-extract micro-data for ALL regions
5. Filter KBR-specific data
6. Update infl_kbr_detailed.csv
7. Print summary of January 2026 data for verification
"""
import pandas as pd
import subprocess
import sys
import os
import io

DB_PATH = "data/db_cpi_store.accdb"
KBR_CODE = 7  # КБР region code (verified from previous analysis)
RF_CODE = 0   # РФ aggregate

def export_small_table(table_name, output_path):
    """Export a small table directly."""
    print(f"  Exporting {table_name}...")
    result = subprocess.run(
        ["mdb-export", DB_PATH, table_name],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return None
    
    with open(output_path, 'w') as f:
        f.write(result.stdout)
    
    lines = result.stdout.strip().split('\n')
    print(f"  → {output_path} ({len(lines)-1} rows)")
    return output_path


def stream_extract_indices(target_items, output_path, label="indices"):
    """Stream-extract data_indices filtering by Item_code set."""
    print(f"  Streaming data_indices for {label} ({len(target_items)} item codes)...")
    sys.stdout.flush()
    
    cmd = ["mdb-export", DB_PATH, "data_indices"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    
    chunks = []
    row_count = 0
    
    try:
        for chunk in pd.read_csv(process.stdout, chunksize=200_000):
            filtered = chunk[chunk['Item_code'].isin(target_items)].copy()
            if not filtered.empty:
                subset = filtered[['Day', 'Region_code', 'Item_code', 'MoM', 'YoY']]
                chunks.append(subset)
                row_count += len(subset)
    except Exception as e:
        print(f"  ERROR reading: {e}")
    
    process.wait()
    
    if not chunks:
        print(f"  No data found!")
        return None
    
    df = pd.concat(chunks, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Day'], format='%m/%d/%y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['Date'])
    df.to_csv(output_path, index=False)
    print(f"  → {output_path} ({len(df)} rows)")
    
    # Show date range
    print(f"    Date range: {df['Date'].min().strftime('%Y-%m')} — {df['Date'].max().strftime('%Y-%m')}")
    return df


def extract_kbr_data(all_regions_df, output_path):
    """Filter KBR data from all regions DataFrame."""
    kbr = all_regions_df[all_regions_df['Region_code'] == KBR_CODE].copy()
    kbr.to_csv(output_path, index=False)
    print(f"  → {output_path} ({len(kbr)} rows)")
    return kbr


def update_infl_kbr_detailed(kbr_indices_df):
    """Create/update infl_kbr_detailed.csv from KBR indices."""
    target_codes = {
        1: 'Все товары и услуги',
        3: 'Продовольственные товары',
        2: 'Непродовольственные товары',
        4: 'Услуги',
        33: 'Плодоовощная продукция',
    }
    
    df = kbr_indices_df[kbr_indices_df['Item_code'].isin(target_codes.keys())].copy()
    df['Товар'] = df['Item_code'].map(target_codes)
    
    pivot = df.pivot_table(index='Date', columns='Товар', values='MoM', aggfunc='first')
    pivot = pivot.sort_index()
    pivot.to_csv('data/infl_kbr_detailed.csv', sep=';', decimal='.')
    print(f"  → data/infl_kbr_detailed.csv ({len(pivot)} rows)")
    return pivot


def update_inflation_data(kbr_indices_df):
    """Update inflation_data.csv with real January 2026 MoM values from KBR."""
    # Read current file
    infl = pd.read_csv('data/inflation_data.csv', sep=';', decimal=',', encoding='utf-8-sig')
    
    # Get January 2026 values from ACCDB
    jan_2026 = kbr_indices_df[
        (kbr_indices_df['Date'].dt.year == 2026) & 
        (kbr_indices_df['Date'].dt.month == 1)
    ]
    
    if jan_2026.empty:
        print("  WARNING: No January 2026 data found in KBR indices!")
        return
    
    # Item code mapping to inflation_data columns
    code_to_col = {
        1: 'mom',    # Все товары и услуги
        2: 'Nonprod',  # Непродовольственные
        3: 'Prod',     # Продовольственные
        4: 'Serv',     # Услуги
    }
    
    print("  January 2026 KBR MoM values from ACCDB:")
    for item_code, col_name in code_to_col.items():
        row = jan_2026[jan_2026['Item_code'] == item_code]
        if not row.empty:
            val = row['MoM'].values[0]
            print(f"    Item {item_code} ({col_name}): {val}")
    
    # Check if last row is Jan 2026
    last_date = infl['Date'].iloc[-1] if 'Date' in infl.columns else infl.iloc[-1, 0]
    print(f"  Current last row date: {last_date}")
    
    # Update the last row if it's January 2026
    if '31.01.2026' in str(last_date):
        for item_code, col_name in code_to_col.items():
            row = jan_2026[jan_2026['Item_code'] == item_code]
            if not row.empty:
                val = row['MoM'].values[0]
                # Format: comma decimal
                infl.iloc[-1, infl.columns.get_loc(col_name)] = str(val).replace('.', ',')
        
        infl.to_csv('data/inflation_data.csv', sep=';', index=False, encoding='utf-8-sig')
        print("  → data/inflation_data.csv updated (Jan 2026 row)")
    else:
        print("  NOTE: Last row is not Jan 2026. Manual update may be needed.")


def print_jan_2026_summary(all_indices_df):
    """Print summary of January 2026 data for verification."""
    jan = all_indices_df[
        (all_indices_df['Date'].dt.year == 2026) &
        (all_indices_df['Date'].dt.month == 1)
    ]
    
    if jan.empty:
        print("  ⚠️ No January 2026 data!")
        return
    
    print(f"\n{'='*60}")
    print(f"JANUARY 2026 SUMMARY")
    print(f"{'='*60}")
    
    # RF aggregate
    rf = jan[jan['Region_code'] == RF_CODE]
    if not rf.empty:
        print("\n  РФ:")
        for _, r in rf.sort_values('Item_code').iterrows():
            item_name = {1: 'Все', 2: 'Непрод', 3: 'Прод', 4: 'Услуги', 33: 'Плодоовощи'}.get(r['Item_code'], f'Item_{r["Item_code"]}')
            print(f"    {item_name}: MoM={r['MoM']:.2f}")
    
    # KBR
    kbr = jan[jan['Region_code'] == KBR_CODE]
    if not kbr.empty:
        print("\n  КБР:")
        for _, r in kbr.sort_values('Item_code').iterrows():
            item_name = {1: 'Все', 2: 'Непрод', 3: 'Прод', 4: 'Услуги', 33: 'Плодоовощи'}.get(r['Item_code'], f'Item_{r["Item_code"]}')
            print(f"    {item_name}: MoM={r['MoM']:.2f}")
    
    # Count unique regions
    n_regions = jan['Region_code'].nunique()
    print(f"\n  Уникальных регионов: {n_regions}")


def print_weights_summary():
    """Print summary of updated weights for KBR."""
    print(f"\n{'='*60}")
    print(f"WEIGHTS UPDATE SUMMARY (January 2026)")
    print(f"{'='*60}")
    
    # Read the exported weights
    try:
        # Stream read for large file
        cmd = ["mdb-export", DB_PATH, "weights"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        
        jan_2026_chunks = []
        for chunk in pd.read_csv(process.stdout, chunksize=100_000):
            # Filter Jan 2026 for RF (code 0) + KBR (code 7), only main items
            mask = (
                chunk['Day'].str.startswith('01/01/26') & 
                chunk['Region_code'].isin([0, KBR_CODE]) &
                chunk['Item_code'].isin([1, 2, 3, 4])
            )
            filtered = chunk[mask]
            if not filtered.empty:
                jan_2026_chunks.append(filtered)
        
        process.wait()
        
        if jan_2026_chunks:
            weights_df = pd.concat(jan_2026_chunks)
            
            for region in [0, KBR_CODE]:
                region_name = "РФ" if region == 0 else "КБР"
                print(f"\n  {region_name} (Region {region}):")
                region_w = weights_df[weights_df['Region_code'] == region]
                for _, r in region_w.sort_values('Item_code').iterrows():
                    item_name = {1: 'Все', 2: 'Непрод', 3: 'Прод', 4: 'Услуги'}.get(r['Item_code'], f'Item_{r["Item_code"]}')
                    print(f"    {item_name}: Weight_vertical={r['Weight_vertical']:.5f} ({r['Weight_vertical']*100:.2f}%)")
        else:
            print("  No January 2026 weights found!")
            
    except Exception as e:
        print(f"  Error reading weights: {e}")


def main():
    print("="*60)
    print("PROCESSING UPDATED db_cpi_store.accdb")
    print("="*60)
    
    # Step 1: Export small tables
    print("\n[1/7] Exporting reference tables...")
    export_small_table("regions_names", "data/regions_names.csv")
    export_small_table("items_names", "data/items_names.csv")
    export_small_table("items_structure", "data/items_structure.csv")
    export_small_table("region_structure", "data/region_structure.csv")
    export_small_table("z_max_day", "data/z_max_day.csv")
    export_small_table("items_set", "data/items_set.csv")
    
    # Step 2: Export weights
    print("\n[2/7] Exporting weights table (this may take a while)...")
    export_small_table("weights", "data/access_weights.csv")
    
    # Step 3: Stream extract aggregate indices
    print("\n[3/7] Extracting aggregate indices (Items 1,2,3,4,33)...")
    aggregate_items = {1, 2, 3, 4, 33}
    all_indices = stream_extract_indices(aggregate_items, "data/all_regions_indices.csv", "aggregates")
    
    if all_indices is None:
        print("FATAL: No aggregate indices extracted!")
        sys.exit(1)
    
    # Step 4: Extract KBR data
    print("\n[4/7] Filtering KBR data...")
    kbr_indices = extract_kbr_data(all_indices, "data/kbr_aggregates.csv")
    
    # Step 5: Update infl_kbr_detailed.csv
    print("\n[5/7] Updating infl_kbr_detailed.csv...")
    update_infl_kbr_detailed(kbr_indices)
    
    # Step 6: Update inflation_data.csv
    print("\n[6/7] Updating inflation_data.csv with Jan 2026 real values...")
    update_inflation_data(kbr_indices)
    
    # Step 7: Summary
    print("\n[7/7] Generating summaries...")
    print_jan_2026_summary(all_indices)
    print_weights_summary()
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
