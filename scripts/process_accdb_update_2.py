"""
Fix inflation_data.csv: standardize decimal separators and extract full KBR data.
Also update micro_sprav.csv weights from the new ACCDB weights table.
"""
import pandas as pd
import subprocess
import sys

DB_PATH = "data/db_cpi_store.accdb"
KBR_CODE = 7

def fix_inflation_data():
    """Fix decimal separator inconsistency in inflation_data.csv."""
    print("[1] Fixing inflation_data.csv decimal separators...")
    
    # Read as raw text and fix the last line
    with open('data/inflation_data.csv', 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    
    # The file uses dots as decimals and semicolons as separators
    # Fix the last line that has commas
    last_line = lines[-1]
    if ',' in last_line:
        # Replace commas with dots (only in numeric fields, not the date)
        parts = last_line.strip().split(';')
        fixed_parts = []
        for i, part in enumerate(parts):
            if i == 0:  # Date field
                fixed_parts.append(part)
            else:
                fixed_parts.append(part.replace(',', '.'))
        lines[-1] = ';'.join(fixed_parts) + '\n'
    
    with open('data/inflation_data.csv', 'w', encoding='utf-8-sig') as f:
        f.writelines(lines)
    
    # Verify
    with open('data/inflation_data.csv', 'r', encoding='utf-8-sig') as f:
        all_lines = f.readlines()
    print(f"  Last 3 rows:")
    for line in all_lines[-3:]:
        print(f"    {line.strip()}")


def extract_full_kbr():
    """Extract ALL item codes for KBR from data_indices."""
    print("\n[2] Extracting full KBR data (all items)...")
    sys.stdout.flush()
    
    cmd = ["mdb-export", DB_PATH, "data_indices"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    
    chunks = []
    for chunk in pd.read_csv(process.stdout, chunksize=200_000):
        filtered = chunk[chunk['Region_code'] == KBR_CODE]
        if not filtered.empty:
            chunks.append(filtered)
    
    process.wait()
    
    if not chunks:
        print("  ERROR: No KBR data found!")
        return
    
    df = pd.concat(chunks, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Day'], format='%m/%d/%y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['Date'])
    
    df.to_csv('data/kbr_indices.csv', index=False)
    print(f"  → data/kbr_indices.csv ({len(df)} rows)")
    print(f"    Date range: {df['Date'].min().strftime('%Y-%m')} — {df['Date'].max().strftime('%Y-%m')}")
    print(f"    Unique Item_codes: {df['Item_code'].nunique()}")
    
    # The production micro models require every item in the canonical KBR basket,
    # not only a small aggregate proxy list.
    canonical_basket = pd.read_csv(
        "data/micro_sprav.csv",
        sep=";",
        encoding="utf-8-sig",
        usecols=["Item_code"],
    )
    micro_codes = set(
        pd.to_numeric(canonical_basket["Item_code"], errors="raise").astype(int)
    )
    micro = df[df["Item_code"].isin(micro_codes)].copy()
    if micro.empty:
        raise ValueError("No canonical micro-basket observations found for KBR")
    micro.to_csv("data/kbr_micro_full.csv", index=False)
    print(
        f"  → data/kbr_micro_full.csv ({len(micro)} rows, "
        f"{micro['Item_code'].nunique()} basket items)"
    )
    
    return df


def update_micro_sprav():
    """Update micro_sprav.csv with January 2026 weights from ACCDB."""
    print("\n[3] Updating micro_sprav.csv with new KBR weights...")
    
    # Read current micro_sprav
    sprav = pd.read_csv('data/micro_sprav.csv', sep=';', encoding='utf-8-sig')
    print(f"  Current micro_sprav: {len(sprav)} items")
    
    # Read items_names for mapping
    items = pd.read_csv('data/items_names.csv')
    
    # Read Jan 2026 weights for KBR from access_weights.csv
    weights = pd.read_csv('data/access_weights.csv')
    
    # Parse dates
    weights['Date'] = pd.to_datetime(weights['Day'], format='%m/%d/%y %H:%M:%S', errors='coerce')
    
    # Get Jan 2026 KBR weights
    jan26_kbr = weights[
        (weights['Region_code'] == KBR_CODE) &
        (weights['Date'].dt.year == 2026) &
        (weights['Date'].dt.month == 1)
    ].copy()
    
    if jan26_kbr.empty:
        print("  WARNING: No Jan 2026 KBR weights found!")
        return
    
    print(f"  Found {len(jan26_kbr)} weight entries for КБР Jan 2026")
    
    # Update weights in micro_sprav
    weight_map = dict(zip(jan26_kbr['Item_code'], jan26_kbr['Weight_vertical']))
    
    updated_count = 0
    for idx, row in sprav.iterrows():
        item_code = row['Item_code']
        if item_code in weight_map:
            old_weight = row['Weight']
            new_weight = weight_map[item_code]
            if abs(old_weight - new_weight) > 0.00001:
                sprav.at[idx, 'Weight'] = new_weight
                updated_count += 1
    
    sprav.to_csv('data/micro_sprav.csv', sep=';', index=False, encoding='utf-8-sig')
    print(f"  Updated {updated_count} weights in micro_sprav.csv")
    
    # Show top 10 by weight
    print(f"\n  Top 10 items by weight (Jan 2026, КБР):")
    top10 = sprav.nlargest(10, 'Weight')
    for _, r in top10.iterrows():
        print(f"    {r['Товар'][:40]:40s} Weight={r['Weight']:.5f}")


def extract_all_regions_micro():
    """Extract micro-data for ALL regions."""
    print("\n[4] Extracting micro-data for ALL regions...")
    sys.stdout.flush()
    
    target_items = {10, 1100, 21, 42, 1700, 4700, 9400, 7400}
    
    cmd = ["mdb-export", DB_PATH, "data_indices"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    
    chunks = []
    for chunk in pd.read_csv(process.stdout, chunksize=200_000):
        filtered = chunk[chunk['Item_code'].isin(target_items)]
        if not filtered.empty:
            chunks.append(filtered[['Day', 'Region_code', 'Item_code', 'MoM']])
    
    process.wait()
    
    if not chunks:
        print("  ERROR: No micro data found!")
        return
    
    df = pd.concat(chunks, ignore_index=True)
    df['Date'] = pd.to_datetime(df['Day'], format='%m/%d/%y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['Date'])
    
    df.to_csv('data/all_regions_micro.csv', index=False)
    print(f"  → data/all_regions_micro.csv ({len(df)} rows)")
    print(f"    Date range: {df['Date'].min().strftime('%Y-%m')} — {df['Date'].max().strftime('%Y-%m')}")


def main():
    fix_inflation_data()
    extract_full_kbr()
    update_micro_sprav()
    extract_all_regions_micro()
    print("\n" + "="*60)
    print("ALL UPDATES COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
