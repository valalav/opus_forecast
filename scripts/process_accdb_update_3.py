"""Fix micro_sprav.csv weights and extract all-regions micro data."""
import pandas as pd
import subprocess
import sys

DB_PATH = "data/db_cpi_store.accdb"
KBR_CODE = 7

def update_micro_sprav():
    """Update micro_sprav.csv with January 2026 weights from ACCDB."""
    print("[1] Updating micro_sprav.csv with new KBR weights...")
    
    sprav = pd.read_csv('data/micro_sprav.csv', sep=';', encoding='utf-8-sig')
    # Fix Weight column type - may have comma decimals
    sprav['Weight'] = sprav['Weight'].astype(str).str.replace(',', '.').astype(float)
    print(f"  Current micro_sprav: {len(sprav)} items")
    
    weights = pd.read_csv('data/access_weights.csv')
    weights['Date'] = pd.to_datetime(weights['Day'], format='%m/%d/%y %H:%M:%S', errors='coerce')
    
    jan26_kbr = weights[
        (weights['Region_code'] == KBR_CODE) &
        (weights['Date'].dt.year == 2026) &
        (weights['Date'].dt.month == 1)
    ].copy()
    
    if jan26_kbr.empty:
        print("  WARNING: No Jan 2026 KBR weights found!")
        return
    
    print(f"  Found {len(jan26_kbr)} weight entries for КБР Jan 2026")
    
    weight_map = dict(zip(jan26_kbr['Item_code'], jan26_kbr['Weight_vertical']))
    
    updated_count = 0
    missing_count = 0
    for idx, row in sprav.iterrows():
        item_code = row['Item_code']
        if item_code in weight_map:
            old_weight = float(row['Weight'])
            new_weight = float(weight_map[item_code])
            if abs(old_weight - new_weight) > 0.000001:
                sprav.at[idx, 'Weight'] = new_weight
                updated_count += 1
        else:
            missing_count += 1
    
    # Save with proper format
    sprav.to_csv('data/micro_sprav.csv', sep=';', index=False, encoding='utf-8-sig')
    print(f"  Updated {updated_count} weights, {missing_count} items not found in new weights")
    
    # Show top 10 by weight
    print(f"\n  Top 10 items by weight (Jan 2026, КБР):")
    top10 = sprav.nlargest(10, 'Weight')
    for _, r in top10.iterrows():
        name = str(r['Товар'])[:45]
        print(f"    {name:45s} Weight={r['Weight']:.5f}")
    
    # Show aggregate weights
    print(f"\n  Aggregate weights by component:")
    for comp in sprav['Компонент'].unique():
        comp_weight = sprav[sprav['Компонент'] == comp]['Weight'].sum()
        if comp_weight > 0.01:
            print(f"    {str(comp):35s} {comp_weight:.5f} ({comp_weight*100:.2f}%)")


def extract_all_regions_micro():
    """Extract micro-data for ALL regions."""
    print("\n[2] Extracting micro-data for ALL regions...")
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
    update_micro_sprav()
    extract_all_regions_micro()
    print("\n" + "="*60)
    print("ALL UPDATES COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
